# import matrix math libraries
import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd, jit, vmap, random
from jax.scipy.linalg import block_diag
from jax.scipy.stats.norm import cdf, pdf

# import ODE solver
from jax.experimental.ode import odeint

# import optimizer for parameter estimation
from scipy.optimize import minimize

import matplotlib.pyplot as plt

# define function that returns model sensitivity vector
def runODE(t_eval, x0, params, dX_dt):
    # solve ODE model
    y = odeint(dX_dt, x0, t_eval, params)
    return jnp.nan_to_num(y)

# define function to integrate adjoint sensitivity equations backwards
def runODEA(t_eval, xt, at, params, dXA_dt):
    # initialize gradient of loss w.r.t. parameters
    lt = np.zeros(len(params))

    # concatenate final condition
    xal = (xt, at, lt)

    # solve ODE model backwards
    x0, a0, l0 = odeint(dXA_dt, xal, t_eval, params)

    # return parameter gradient
    return l0[-1]

# define function that returns model sensitivity vector
def runODEZ(t_eval, x0, params, dXZ_dt):
    # check dimensions

    # set initial condition of sensitivity matrix
    xz = (x0, jnp.zeros([len(x0), len(params)]))

    # solve augmented ODE model
    y, Z = odeint(dXZ_dt, xz, t_eval, params)

    return jnp.nan_to_num(y), jnp.nan_to_num(Z)

### Function to process dataframes ###
def process_df(df, species):

    # store measured datasets for quick access
    data = []
    for treatment, comm_data in df.groupby("Treatments"):

        # make sure comm_data is sorted in chronological order
        comm_data = comm_data.sort_values(by='Time', ascending=True).copy()

        # pull evaluation times
        t_eval = np.array(comm_data['Time'].values, float)

        # pull species data
        Y_measured = np.clip(np.array(comm_data[species].values, float), 0., np.inf)

        # append t_eval and Y_measured to data list
        data.append([treatment, t_eval, Y_measured])

    return data

class gLV:
    def __init__(self, species, dataframe, verbose=True):

        # set species names
        self.species_names = species

        # process dataframe
        self.dataset = process_df(dataframe, species)

        # dimension of model output
        self.n_s = len(species)

        # initialize parameters
        self.n_params = self.n_s + self.n_s**2
        self.params = -.1*np.ones(self.n_params)

        # set small positive growth rate
        self.params[:self.n_s] = .3

        # set negative self interaction
        k = len(species)
        for i in range(self.n_s):
            for j in range(self.n_s):
                if i == j:
                    self.params[k] = -1.
                k += 1

        # set prior
        self.prior = np.copy(self.params)

        # runtime parameters
        self.verbose = verbose

        # set parameters of precision hyper-priors
        self.a = 1e-4
        self.b = 1e-4

        # set posterior parameter precision and covariance to None
        self.A = None
        self.Ainv = None

        # jit compiled derivative of self.dX_dt
        def dX_dt(x, t, params):

            # reshape params to growth rates and interaction matrix
            r = params[:self.n_s]
            A = jnp.reshape(params[self.n_s:self.n_s+self.n_s**2], [self.n_s, self.n_s])

            return x*(r + A@x)
        self.dX_dt = jit(dX_dt)

        # adjoint sensitivity derivative
        def dXA_dt(xa, t, params):

            # unpack state, adjoint, and gradient of loss w.r.t. parameters
            x, a, _ = xa
            # vjp returns self.dX_dt evaluated at x,t,params and
            # vjpfun, which evaluates a^T Jx, a^T Jp
            # where Jx is the gradient of the self.dX_dt w.r.t. x
            # and   Jp is the gradient of the self.dX_dt w.r.t. parameters
            y_dot, vjpfun = jax.vjp(lambda x, params: self.dX_dt(x,t,params), x, params)
            vjps = vjpfun(a)

            return (-y_dot, *vjps)
        self.dXA_dt = jit(dXA_dt)

        # if not vectorized, xz will be 1-D
        def dXZ_dt(xZ, t, params):
            # split up x, z, and z0
            x, Z = xZ

            # compute derivatives
            dxdt  = self.dX_dt(x, t, params)

            # time derivative of initial condition sensitivity
            # Jacobian-vector-product approach is surprisingly slow
            # JxV  = vmap(lambda z: jax.jvp(lambda x: self.dX_dt(x,t,params), (x,), (z,))[1], (1), (1))
            Jx = jacfwd(self.dX_dt, 0)(x, t, params)

            # time derivative of parameter sensitivity
            JxZ = Jx@Z # JxV(Z)

            # compute gradient of model w.r.t. parameters
            Jp = jacfwd(self.dX_dt, 2)(x, t, params)

            # return derivatives
            return (dxdt, JxZ + Jp)
        self.dXZ_dt = jit(dXZ_dt)

        # jit compile function to integrate ODE
        self.runODE  = jit(lambda t_eval, x, params: runODE(t_eval, x[0], params, self.dX_dt))

        # jit compile function to integrate forward sensitivity equations
        self.runODEZ = jit(lambda t_eval, x, params: runODEZ(t_eval, x[0], params, self.dXZ_dt))

        # jit compile function to integrate adjoint sensitivity equations
        self.adjoint = jit(vmap(jacfwd(lambda xt, yt, B: jnp.einsum("i,ij,j", yt-xt, B, yt-xt)/2.), (0, 0, None)))
        self.runODEA = jit(lambda t, xt, at, params: runODEA(t, xt, at, params, self.dXA_dt))

        # JIT compile matrix operations
        def GAinvG(G, Ainv):
            return jnp.einsum("tki,ij,tlj->tkl", G, Ainv, G)
        self.GAinvG = jit(GAinvG)

        def yCOV_next(Y_error, G, Ainv):
            # sum over time dimension
            return jnp.einsum('tk,tl->kl', Y_error, Y_error) + jnp.sum(self.GAinvG(G, Ainv), 0)
        self.yCOV_next = jit(yCOV_next)

        def A_next(G, Beta):
            A_n = jnp.einsum('tki, kl, tlj->ij', G, Beta, G)
            A_n = (A_n + A_n.T)/2.
            return A_n
        self.A_next = jit(A_next)

        # jit compile inverse Hessian computation step
        def Ainv_next(G, Ainv, BetaInv):
            GAinv = G@Ainv # [n_t, n_p]
            Ainv_step = GAinv.T@jnp.linalg.inv(BetaInv + GAinv@G.T)@GAinv
            # Ainv_step = jnp.einsum("ti,tk,kj->ij", GAinv, jnp.linalg.inv(BetaInv + jnp.einsum("ti,ki->tk", GAinv, G)), GAinv)
            Ainv_step = (Ainv_step + Ainv_step.T)/2.
            return Ainv_step
        self.Ainv_next = jit(Ainv_next)

        # jit compile inverse Hessian computation step
        def Ainv_prev(G, Ainv, BetaInv):
            GAinv = G@Ainv
            Ainv_step = GAinv.T@jnp.linalg.inv(GAinv@G.T - BetaInv)@GAinv
            Ainv_step = (Ainv_step + Ainv_step.T)/2.
            return Ainv_step
        self.Ainv_prev = jit(Ainv_prev)

        # jit compile function to compute log of determinant of a matrix
        def log_det(A):
            L = jnp.linalg.cholesky(A)
            return 2*jnp.sum(jnp.log(jnp.diag(L)))
        self.log_det = jit(log_det)

        # approximate inverse of A, where A = LL^T, Ainv = Linv^T Linv
        def compute_Ainv(A):
            Linv = jnp.linalg.inv(jnp.linalg.cholesky(A))
            Ainv = Linv.T@Linv
            return Ainv
        self.compute_Ainv = jit(compute_Ainv)

        def eval_grad_NLP(Y_error, Beta, G):
            return jnp.einsum('tk,kl,tli->i', Y_error, Beta, G)
        self.eval_grad_NLP = jit(eval_grad_NLP)

        # jit compile prediction covariance computation
        def compute_searchCOV(Beta, G, Ainv):
            # dimensions of sample
            n_t, n_y, n_theta = G.shape
            # stack G over time points [n, n_t, n_out, n_theta]--> [n, n_t*n_out, n_theta]
            Gaug = jnp.concatenate(G, 0)
            return jnp.eye(n_t*n_y) + jnp.einsum("kl,li,ij,mj->km", block_diag(*[Beta]*n_t), Gaug, Ainv, Gaug)
        self.compute_searchCOV = jit(compute_searchCOV)

        # jit compile prediction covariance computation
        def compute_forgetCOV(Beta, G, Ainv):
            # dimensions of sample
            n_t, n_y, n_theta = G.shape
            # stack G over time points [n, n_t, n_out, n_theta]--> [n, n_t*n_out, n_theta]
            Gaug = jnp.concatenate(G, 0)
            return jnp.eye(n_t*n_y) - jnp.einsum("kl,li,ij,mj->km", block_diag(*[Beta]*n_t), Gaug, Ainv, Gaug)
        self.compute_forgetCOV = jit(compute_forgetCOV)

        # compute utility of each experiment
        def utility(searchCOV):
            # predicted objective + log det of prediction covariance over time series
            # searchCOV has shape [n_out, n_out]
            # log eig predCOV has shape [n_out]
            # det predCOV is a scalar
            return jnp.nansum(jnp.log(jnp.linalg.eigvalsh(searchCOV)))
        self.utility = jit(utility)

    def fit(self, evidence_tol=1e-3, nlp_tol=None, alpha_0=1., patience=1, max_fails=2, beta=1e-3):
        # set initial regularization
        self.alpha_0 = alpha_0

        # estimate parameters using gradient descent
        self.itr = 0
        passes = 0
        fails = 0
        convergence = np.inf
        previdence  = -np.inf

        # initialize hyper parameters
        self.init_hypers()

        while passes < patience and fails < max_fails:
            # update Alpha and Beta hyper-parameters
            if self.itr>0:
                self.update_hypers()
                nlp_tol = 1e-3

            # fit using updated Alpha and Beta
            self.res = minimize(fun=self.objective,
                                jac=self.jacobian_fwd,
                                hess=self.hessian,
                                x0=self.params,
                                tol=nlp_tol,
                                method='Newton-CG',
                                callback=self.callback)
            if self.verbose:
                print(self.res.message)
                # assert self.res.success, "optimizer did not converge"
            self.params = self.res.x

            # update precision
            if self.itr==0:
                self.alpha = self.alpha_0
                self.Alpha = self.alpha_0*np.ones(self.n_params)
            self.update_precision()
            # update covariance (Hessian inverse)
            self.update_covariance()

            # update evidence
            self.update_evidence()
            assert not np.isnan(self.evidence), "Evidence is NaN! Something went wrong."

            # check convergence
            convergence = np.abs(previdence - self.evidence) / np.max([1.,np.abs(self.evidence)])

            # update pass count
            if convergence < evidence_tol:
                passes += 1
                print("Pass count ", passes)
            else:
                passes = 0

            # increment fails if convergence is negative
            if self.evidence < previdence:
                fails += 1
                print("Fail count ", fails)
            else:
                fails = 0

            # update evidence
            previdence = np.copy(self.evidence)
            self.itr += 1

    def init_hypers(self):

        # count number of samples
        self.N = 0

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured in self.dataset:

            # count effective number of uncorrelated observations
            k = 0 # number of outputs
            for series in Y_measured.T:
                # check if there is any variation in the series
                if np.std(series) > 0:
                    # count number of outputs that vary over time
                    k += 1
            assert k > 0, f"There are no time varying outputs in sample {treatment}"

            # adjust N to account for unmeasured outputs
            self.N += (len(series) - 1) * k / self.n_s

        # init output precision
        self.Beta = np.eye(self.n_s)
        self.BetaInv = np.eye(self.n_s)

        # initial guess of parameter precision
        self.alpha = 1e-3
        self.Alpha = self.alpha*np.ones(self.n_params)
        # self.alpha = 0.
        # self.Alpha = np.zeros(self.n_params)

        if self.verbose:
            print("Total samples: {:.0f}, Initial regularization: {:.2e}".format(self.N, self.alpha))

    # EM algorithm to update hyper-parameters
    def update_hypers(self):
        print("Updating hyper-parameters...")

        # init yCOV
        yCOV = 0.

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured in self.dataset:

            # for each output
            output, G = self.runODEZ(t_eval, Y_measured, self.params)

            # Determine SSE
            Y_error = output[1:] - Y_measured[1:]
            yCOV += self.yCOV_next(Y_error, G[1:], self.Ainv)

        ### M step: update hyper-parameters ###

        # maximize complete data log-likelihood w.r.t. alpha and beta
        Ainv_ii = np.diag(self.Ainv)
        self.alpha = self.n_params/(np.sum((self.params-self.prior)**2) + np.sum(Ainv_ii) + 2.*self.a)
        # self.Alpha = self.alpha*np.ones(self.n_params)
        self.Alpha = 1./((self.params-self.prior)**2 + Ainv_ii + 2.*self.a)

        # update output precision
        self.Beta = self.N*np.linalg.inv(yCOV + 2.*self.b*np.eye(self.n_s))
        self.Beta = (self.Beta + self.Beta.T)/2.

        # make sure that precision is positive definite (algorithm 3.3 in Numerical Optimization)
        self.Beta = self.make_pos_def(self.Beta, jnp.ones(self.n_s))

        # compute covariance of data distribution
        self.BetaInv = np.linalg.inv(self.Beta)

        if self.verbose:
            print("Total samples: {:.0f}, Updated regularization: {:.2e}".format(self.N, self.alpha))

    def objective(self, params):
        # compute negative log posterior (NLP)
        self.NLP = np.sum(self.Alpha * (params-self.prior)**2) / 2.
        # compute residuals
        self.RES = 0.

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured in self.dataset:

            # for each output
            output = np.nan_to_num(self.runODE(t_eval, Y_measured, params))

            # Determine error
            Y_error = output[1:] - Y_measured[1:]

            # Determine SSE and gradient of SSE
            self.NLP += np.einsum('tk,kl,tl->', Y_error, self.Beta, Y_error)/2.
            self.RES += np.sum(Y_error)/self.N

        # return NLP
        return self.NLP

    def jacobian_adj(self, params):

        # compute gradient of negative log posterior
        grad_NLP = self.Alpha*(params-self.prior)

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured in self.dataset:

            ### Adjoint sensitivity method can be slower or return NaN depending on stiffness of ODE
            output = np.nan_to_num(self.runODE(t_eval, Y_measured, params))

            # adjoint at measured time points
            at = self.adjoint(output, Y_measured, self.Beta)

            # gradient of NLP
            for t, out, a in zip(t_eval[1:], output[1:], at[1:]):
                grad_NLP += self.runODEA(t, out, a, params)

        # return gradient of NLP
        return grad_NLP

    def jacobian_fwd(self, params):

        # compute gradient of negative log posterior
        grad_NLP = self.Alpha*(params-self.prior)

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured in self.dataset:

            # for each output
            output, G = self.runODEZ(t_eval, Y_measured, params)

            # Determine error
            Y_error = output[1:] - Y_measured[1:]

            # sum over time and outputs to get gradient w.r.t params
            grad_NLP += self.eval_grad_NLP(Y_error, self.Beta, G[1:])

        # return gradient of NLP
        return grad_NLP

    def hessian(self, params):

        # compute Hessian of NLP
        self.A = np.diag(self.Alpha)

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured in self.dataset:

            # for each output
            output, G = self.runODEZ(t_eval, Y_measured, params)

            # compute Hessian
            self.A += self.A_next(G[1:], self.Beta)

        # make sure precision is symmetric
        self.A = (self.A + self.A.T)/2.

        # return Hessian
        return self.A

    def update_precision(self):
        # update parameter covariance matrix given current parameter estimate
        self.A = np.diag(self.Alpha)

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured in self.dataset:

            # for each output
            output, G = self.runODEZ(t_eval, Y_measured, self.params)

            # compute Hessian
            self.A += self.A_next(G[1:], self.Beta)

        # Laplace approximation of posterior precision
        self.A = (self.A + self.A.T)/2.
        self.A = self.make_pos_def(self.A, self.Alpha)

    def update_covariance(self):
        ### Approximate / fast method to compute inverse ###
        self.Ainv = self.make_pos_def(self.compute_Ainv(self.A), jnp.ones(self.n_params))

    # compute the log marginal likelihood
    def update_evidence(self):
        # compute evidence
        self.evidence = self.N/2*self.log_det(self.Beta)  + \
                        1/2*np.nansum(np.log(self.Alpha)) - \
                        1/2*self.log_det(self.A) - self.NLP

        # print evidence
        if self.verbose:
            print("Evidence {:.3f}".format(self.evidence))

    # make sure that precision is positive definite (algorithm 3.3 in Numerical Optimization)
    def make_pos_def(self, A, Alpha, beta=1e-3):

        # initial amount to add to matrix
        if jnp.min(jnp.diag(A)) > 0:
            tau = 0.
        else:
            tau = beta - jnp.min(jnp.diag(A))

        # use cholesky decomposition to check positive-definiteness of A
        while jnp.isnan(jnp.linalg.cholesky(A)).any():

            # increase precision of prior until posterior precision is positive definite
            A += tau*jnp.diag(Alpha)

            # increase prior precision
            tau = np.max([2*tau, beta])

        return A

    def callback(self, xk, res=None):
        if self.verbose:
            print("Loss: {:.3f}, Residuals: {:.3f}".format(self.NLP, self.RES))
        return True

    def predict_point(self, x_test, teval):

        # make predictions given initial conditions and evaluation times
        Y_predicted = np.nan_to_num(self.runODE(teval, np.atleast_2d(x_test), self.params))

        return Y_predicted

    def predict(self, x_test, t_eval, n_std=1.):
        # check if precision has been computed

        # simulate
        output, G = self.runODEZ(t_eval, np.atleast_2d(x_test), self.params)

        # calculate covariance (dimension = [steps, n_out, n_out])
        COV = self.BetaInv + self.GAinvG(G, self.Ainv)

        # determine confidence interval for species
        stdv = n_std*jnp.sqrt(vmap(jnp.diag)(COV))

        return np.array(output), np.array(stdv)

    def get_params(self,):
        # return expected value and standard deviation of model parameters

        # reshape params to growth rates and interaction matrix
        r = self.params[:self.n_s]
        A = np.reshape(self.params[self.n_s:self.n_s + self.n_s ** 2], [self.n_s, self.n_s])

        # parameter stdv
        params_stdv = np.sqrt(np.diag(self.Ainv))

        r_stdv = params_stdv[:self.n_s]
        A_stdv = np.reshape(params_stdv[self.n_s:self.n_s + self.n_s ** 2], [self.n_s, self.n_s])

        return r, A, r_stdv, A_stdv

    def design(self, df_design, N, batch_size=512, Ainv_q=None):
        # process dataframe
        if self.verbose:
            print("Processing design dataframe...")
        design_space = process_df(df_design, self.species_names)

        # total number of possible experimental conditions
        n_samples = len(design_space)
        batch_size = np.min([batch_size, n_samples])

        # init parameter covariance
        if Ainv_q is None:
            Ainv_q = jnp.copy(self.Ainv)

        # store sensitivity to each condition
        if self.verbose:
            print("Computing sensitivies...")
        Gs = {}
        exp_names = []
        for treatment, t_eval, Y_measured in design_space:
            exp_names.append(treatment)

            # run model using current parameters, output = [n_time, n_sys_vars]
            output, G = self.runODEZ(t_eval, Y_measured, self.params)

            # store in hash table of sensitivies (ignoring initial condition)
            Gs[treatment] = G[1:]

        # # randomly select experiments
        # best_experiments = list(np.random.choice(exp_names, 10, replace=False))
        # for exp in best_experiments:
        #     # update parameter covariance given selected experiment
        #     for Gt in Gs[exp]:
        #         Ainv_q -= self.Ainv_next(Gt, Ainv_q, self.BetaInv)

        # search for new experiments
        best_experiments = []
        N_selected = 0
        while N_selected < N:

            # compute information content of each observation
            f_I = []
            for treatment in exp_names:
                # predCOV has shape [n_time, n_out, n_out]
                searchCOV = self.compute_searchCOV(self.Beta, Gs[treatment], Ainv_q)
                f_I.append(self.utility(searchCOV))

            # concatenate utilities
            utilities = np.array(f_I).ravel()
            # print("Top 5 utilities:, ", np.sort(utilities)[::-1][:5])

            # sort utilities from best to worst
            exp_sorted = jnp.argsort(utilities)[::-1]
            for exp in exp_sorted:
                treatment, t_eval, Y_measured = design_space[exp]
                if treatment not in best_experiments:
                    best_experiments.append(treatment)
                    N_selected +=  1

                    # update parameter covariance given selected experiment
                    for Gt in Gs[treatment]:
                        Ainv_q -= self.Ainv_next(Gt, Ainv_q, self.BetaInv)

                    # make sure Ainv_q is positive definite
                    Ainv_q = self.make_pos_def(Ainv_q, jnp.ones(self.n_params))

                    # select next sample
                    print(f"Picked {treatment}")
                    break
                else:
                    print("Picked duplicate!")

        ### Algorithm to continue improving design ###
        while True:
            # Find point that, when dropped, results in smallest decrease in EIG
            f_L = []
            for treatment in best_experiments:
                # compute impact of losing this point
                # | A - G' B G |
                forgetCOV = self.compute_forgetCOV(self.Beta, Gs[treatment], Ainv_q)
                f_L.append(self.utility(forgetCOV))
            worst_exp = best_experiments[np.argmax(f_L)]

            # update parameter covariance given selected experiment
            for Gt in Gs[worst_exp]:
                Ainv_q -= self.Ainv_prev(Gt, Ainv_q, self.BetaInv)
            print(f"Dropped {worst_exp}")

            # Find next most informative point
            f_I = []
            for treatment in exp_names:
                # compute impact of gaining new point
                # | A + G' B G |
                searchCOV = self.compute_searchCOV(self.Beta, Gs[treatment], Ainv_q)
                f_I.append(self.utility(searchCOV))
            best_exp = design_space[np.argmax(f_I)][0]
            # update parameter covariance given selected experiment
            for Gt in Gs[best_exp]:
                Ainv_q -= self.Ainv_next(Gt, Ainv_q, self.BetaInv)
            print(f"Picked {best_exp}")

            # make sure Ainv_q is positive definite
            Ainv_q = self.make_pos_def(Ainv_q, jnp.ones(self.n_params))

            # If the dropped point is the same as the added point,
            # or if the same point selected again, stop
            if worst_exp == best_exp or best_exp in best_experiments:
                return best_experiments
            else:
                best_experiments.remove(worst_exp)
                best_experiments.append(best_exp)

        return best_experiments

    def design_opt(self, df_design, s_ind, N, Ainv_q=None):
        # Bayesian optimization: select for a batch of N conditions that
        # maximize species abundance indexed by s_ind and
        # maximize information content

        # start by determining expected value of best previous observation
        s_vals = []
        for treatment, t_eval, Y_measured in self.dataset:
            # run model using current parameters, output = [n_time, n_sys_vars]
            s_vals.append(Y_measured[-1, s_ind])

        # need to compare to log of best previous observation
        treatment, t_eval, Y_measured = self.dataset[np.argmax(s_vals)]
        s_best = Y_measured[-1, s_ind]
        print(s_best)

        # Bayesian optimization utility function
        @jit
        def utility_opt(pred, G_next, Ainv):

            # determine n_t and n_s
            n_t, n_s, n_p = G_next.shape

            # epistemic stdv of prediction
            stdv = jnp.sqrt(self.GAinvG(G_next, Ainv)[-1, s_ind, s_ind])
            # L = jnp.nan_to_num(jnp.linalg.cholesky(self.GAinvG(G_next, Ainv)))
            # stdv = jnp.einsum("tkl,tl->tk", L, jnp.ones([n_t, n_s]))[-1, s_ind]

            # expected improvement
            improvement = pred - s_best
            z = improvement/stdv
            return improvement*cdf(z) + stdv*pdf(z)

            # UCB
            # return pred + stdv

            # Exploitation
            # return pred

        # process dataframe and determine mean of best predicted sample
        if self.verbose:
            print("Processing design dataframe...")
        design_space = process_df(df_design, self.species_names)

        # store sensitivity to each condition
        if self.verbose:
            print("Computing sensitivies...")
        outputs = {}
        Gs = {}
        exp_names = []
        for treatment, t_eval, Y_measured in design_space:
            exp_names.append(treatment)

            # run model using current parameters, output = [n_time, n_sys_vars]
            output, G = self.runODEZ(t_eval, Y_measured, self.params)

            # make sure NaNs are ignored
            output = np.nan_to_num(output)
            G = np.nan_to_num(G)

            # store in hash table of outputs
            outputs[treatment] = output[-1, s_ind]

            # store in hash table of sensitivies
            # need full matrix for inverse Hessian update
            Gs[treatment] = G[1:]

        # convert exp_names to numpy array to enable logical indexing
        exp_names = np.array(exp_names)

        # init parameter covariance
        if Ainv_q is None:
            Ainv_q = jnp.copy(self.Ainv)

        # search for new experiments
        best_experiments = []
        N_selected = 0
        while N_selected < N:

            # compute information content of each observation
            # utilities = np.array([utility_opt(outputs[treatment], Gs[treatment], Ainv_q) for treatment in exp_names])
            utilities = np.array([utility_opt(outputs[treatment], Gs[treatment], Ainv_q) for treatment in exp_names])
            print("Top 5 utilities:, ", np.sort(utilities)[::-1][:5])

            # sort through utilities from best to worst
            for treatment in exp_names[np.argsort(utilities)[::-1]]:

                # update parameter covariance given selected experiment
                for Gt in Gs[treatment]:
                    Ainv_q -= self.Ainv_next(Gt, Ainv_q, self.BetaInv)

                # make sure Ainv_q is positive definite
                Ainv_q = self.make_pos_def(Ainv_q, jnp.ones(self.n_params))

                if treatment not in best_experiments:
                    best_experiments.append(treatment)
                    # number of selected observations is the evaluation time
                    # minus 1 to ignore initial condition
                    N_selected += 1

                    # select next sample
                    print(f"Picked {treatment}")
                    break
                else:
                    print("Picked duplicate!")

        return best_experiments
