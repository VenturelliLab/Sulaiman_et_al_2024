import numpy as np
import pandas as pd

# from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize

# import pytorch libraries to compute gradients
from jax import vjp, jacfwd, jit, vmap
from jax.nn import tanh, sigmoid
import jax.numpy as jnp
from jax.experimental.ode import odeint

# for troubleshooting
import matplotlib.pyplot as plt

### Function to process dataframes ###
def process_df(df, species, inputs):

    # store measured datasets for quick access
    data = []
    for treatment, comm_data in df.groupby("Treatments"):

        # make sure comm_data is sorted in chronological order
        comm_data.sort_values(by='Time', ascending=True, inplace=True)

        # pull evaluation times
        t_eval = np.array(comm_data['Time'].values, np.float32)

        # pull system data
        Y_measured = np.array(comm_data[species].values, np.float32)

        # condition specific inputs
        Y_inputs = np.array(comm_data[inputs].values, np.float32)[0]

        # append t_eval and Y_measured to data list
        data.append([treatment, t_eval, Y_measured, Y_inputs])

    return data

class NSM:
    def __init__(self, dataframe, species, inputs=[], n_r=2, n_h=5, verbose=True):

        # dimensions
        self.n_s = len(species)
        self.n_r = n_r
        self.n_x = self.n_s + self.n_r + len(inputs)
        self.n_h = n_h

        # initialize consumer resource parameters

        # death rate
        d = -3.*np.ones(self.n_s)
        # [C]_ij = rate that species j consumes resource i
        C = np.random.uniform(-1.,  0., [self.n_r, self.n_s])
        # [P]_ij = rate that species j produces resource i
        P = np.random.uniform(-5., -1., [self.n_r, self.n_s])
        # carrying capacity of species
        ks = np.ones(self.n_s)
        # carrying capacity of resources
        kr = np.ones(self.n_r)

        # initialize neural network parameters
        p_std = 1./np.sqrt(self.n_x)
        # map state to hidden dimension
        W1 = p_std*np.random.randn(self.n_h, self.n_x)
        b1 = np.random.randn(self.n_h)
        # map hidden dimension to efficiencies
        p_std = 1./np.sqrt(self.n_h)
        W2 = p_std*np.random.randn(self.n_r+2*self.n_s, self.n_h)
        b2 = np.random.randn(self.n_r+2*self.n_s)

        # initial resource concentration (log)
        self.r0 = np.random.uniform(-2., 0., self.n_r)

        # concatenate parameter initial guess
        self.params = (d, C, P, ks, kr, W1, b1, W2, b2)

        # determine shapes of parameters
        self.shapes = []
        self.k_params = []
        self.n_params = 0
        for param in self.params:
            self.shapes.append(param.shape)
            self.k_params.append(self.n_params)
            self.n_params += param.size
        self.k_params.append(self.n_params)

        # set prior so that C is sparse
        r0 = -5.*np.ones(self.n_r)
        C0 = -5.*np.ones([self.n_r, self.n_s])
        P0 = -5.*np.ones([self.n_r, self.n_s])
        W10 = np.zeros_like(W1)
        b10 = np.zeros_like(b1)
        W20 = np.zeros_like(W2)
        b20 = np.zeros_like(b2)

        # concatenate prior
        prior = [r0, d, C0, P0, ks, kr, W10, b10, W20, b20]
        self.prior = np.concatenate([p.ravel() for p in prior])

        # set up data
        self.dataset = process_df(dataframe, species, inputs)

        # for additional output messages
        self.verbose = verbose

        # set parameters of precision hyper-priors
        self.a = 1e-4
        self.b = 1e-4

        # set posterior parameter precision and covariance to None
        self.A = None
        self.Ainv = None

        ### JIT compiled helper functions to integrate ODEs ###

        # function to integrate NSM model
        self.runODE  = jit(lambda t_eval, x, r0, params, inputs: odeint(self.system, jnp.concatenate((x[0], r0)), t_eval, params, inputs))

        # function to integrate forward sensitivity equations
        Y0 = np.eye(self.n_s + self.n_r)[:, self.n_s:]
        Z0 = [np.zeros([self.n_s + self.n_r] + list(param.shape)) for param in self.params]
        self.runODEZ = jit(lambda t_eval, x, r0, params, inputs: odeint(self.aug_system, (jnp.concatenate((x[0], r0)), Y0, *Z0), t_eval, params, inputs))

        # function to integrate adjoint sensitivity equations
        self.adjoint = jit(vmap(jacfwd(lambda zt, yt, B: jnp.einsum("i,ij,j", yt-zt[:self.n_s], B, yt-zt[:self.n_s])/2.), (0, 0, None)))
        # define function to integrate adjoint sensitivity equations backwards
        Lt = [jnp.zeros(param.shape) for param in self.params]
        def runODEA(t_eval, zt, at, cr_params, inputs):

            # concatenate final condition
            xal = (zt, at, Lt)

            # solve ODE model
            xaL0 = odeint(self.adj_system, xal, t_eval, cr_params, inputs)

            # adjoint (gradient of loss w.r.t. initial condition)
            a0 = xaL0[1]

            # gradient of loss w.r.t. parameters
            L0 = xaL0[2]

            # concatenate gradients
            grads = jnp.concatenate([a0[-1, self.n_s:]]+[p[-1].ravel() for p in L0])
            return grads
        self.runODEA = jit(lambda t, xt, at, cr_params, inputs: runODEA(jnp.array([0., t]), xt, at, cr_params, inputs))

        ### JIT compiled matrix operations ###

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

    # define neural consumer resource model
    def system(self, x, t, params, inputs):

        # species
        s = x[:self.n_s]

        # resources
        r = jnp.exp(x[self.n_s:])

        # compute state
        state = jnp.concatenate((s, r, inputs))

        # ujnpack params
        d, Cmax, Pmax, ks, kr, W1, b1, W2, b2 = params

        # take exp of strictly positive params
        d = jnp.exp(d)
        Cmax = jnp.exp(Cmax)
        Pmax = jnp.exp(Pmax)
        ks = jnp.exp(ks)
        kr = jnp.exp(kr)

        # map to hidden layer
        h1 = tanh(W1@state + b1)
        h2 = sigmoid(W2@h1 + b2)

        # divide hidden layer into resource availability, species growth efficiency, resource production efficiency
        f = h2[:self.n_r]
        g = h2[self.n_r:self.n_r+self.n_s]
        h = h2[self.n_r+self.n_s:]

        # update consumption matrix according to resource attractiveness
        C = jnp.einsum("i,ij->ij", f, Cmax)

        # scaled production rate
        P = jnp.einsum("ij,j->ij", Pmax, h)

        # rate of change of species
        dsdt = s*(g*(C.T@r) - d)*(1. - s/ks)

        # rate of change of log of resources
        dlrdt = (1. - r/kr)*(P@s) - C@s

        return jnp.concatenate((dsdt, dlrdt))

    # augmented system for forward sensitivity equations
    def aug_system(self, aug_x, t, params, inputs):

        # ujnpack augmented state
        x = aug_x[0]
        Y = aug_x[1]
        Z = aug_x[2:]

        # time derivative of state
        dxdt = self.system(x, t, params, inputs)

        # system jacobian
        Jx_i = jacfwd(self.system, 0)(x, t, params, inputs)

        # time derivative of grad(state, initial condition)
        dYdt = Jx_i@Y

        # time derivative of parameter sensitivity
        dZdt = [jnp.einsum("ij,j...->i...", Jx_i, Z_i) + Jp_i for Z_i, Jp_i in zip(Z, jacfwd(self.system, 2)(x, t, params, inputs))]

        return (dxdt, dYdt, *dZdt)

    # augmented system for adjoint sensitivity method
    def adj_system(self, aug_x, t, params, inputs):

        # unpack state and adjoint
        x = aug_x[0]
        a = aug_x[1]

        # vjp returns self.dX_dt evaluated at x,t,params and
        # vjpfun, which evaluates a^T Jx, a^T Jp
        # where Jx is the gradient of the self.dX_dt w.r.t. x
        # and   Jp is the gradient of the self.dX_dt w.r.t. parameters
        y_dot, vjpfun = vjp(lambda x, params: self.system(x, t, params, inputs), x, params)
        vjps = vjpfun(a)

        return (-y_dot, *vjps)

    # reshape parameters into weight matrices and bias vectors
    def reshape(self, params):
        return [np.array(np.reshape(params[k1:k2], shape), dtype=np.float32) for k1,k2,shape in zip(self.k_params, self.k_params[1:], self.shapes)]

    def fit(self, evidence_tol=1e-2, nlp_tol=1e-3, alpha_0=1e-3, patience=1, max_fails=2, beta=1e-3):
        # estimate parameters using gradient descent
        self.alpha_0 = alpha_0
        self.itr = 0
        passes = 0
        fails = 0
        convergence = np.inf
        previdence  = -np.inf

        # scipy minimize works with a numpy vector of parameters
        params = np.concatenate([self.r0]+[p.ravel() for p in self.params])

        # initialize hyper parameters
        self.init_hypers()

        while passes < patience and fails < max_fails:

            # update Alpha and Beta hyper-parameters
            if self.itr>0: self.update_hypers()

            # fit using updated Alpha and Beta
            self.res = minimize(fun=self.objective,
                                jac=self.jacobian_adj,
                                hess=self.hessian,
                                x0 = params,
                                tol = nlp_tol,
                                method='Newton-CG',
                                callback=self.callback)
            if self.verbose:
                print(self.res.message)
            params = self.res.x
            self.r0 = np.array(params[:self.n_r], dtype=np.float32)
            self.params = self.reshape(params[self.n_r:])

            # update covariance
            self.update_precision()

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

        # finally compute covariance (Hessian inverse)
        self.update_covariance()

    def init_hypers(self):

        # count number of samples
        self.N = 0

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured, inputs in self.dataset:

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
        self.alpha = self.alpha_0
        self.Alpha = self.alpha_0*np.ones(self.n_params+self.n_r)

        if self.verbose:
            print("Total samples: {:.0f}, Updated regularization: {:.2e}".format(self.N, self.alpha))

    # EM algorithm to update hyper-parameters
    def update_hypers(self):
        print("Updating precision...")

        # compute inverse of cholesky decomposed precision matrix
        Ainv = self.compute_Ainv(self.A)

        # make sure precision is positive definite
        Ainv = self.make_pos_def(Ainv, jnp.ones(Ainv.shape[0]))

        # init yCOV
        yCOV = 0.

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured, inputs in self.dataset:

            # integrate forward sensitivity equations
            xYZ = self.runODEZ(t_eval, Y_measured, self.r0, self.params, inputs)
            output = np.nan_to_num(xYZ[0])
            Y = xYZ[1]
            Z = xYZ[2:]

            # collect gradients and reshape
            Z = np.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], -1) for Z_i in Z], -1)

            # stack gradient matrices
            G = np.concatenate((Y, Z), axis=-1)

            # Determine SSE of log of Y
            Y_error = output[1:, :self.n_s] - Y_measured[1:]
            yCOV += self.yCOV_next(Y_error, G[1:, :self.n_s, :], Ainv)

        ### M step: update hyper-parameters ###

        # maximize complete data log-likelihood w.r.t. alpha and beta
        Ainv_ii = np.diag(Ainv)
        params  = np.concatenate([self.r0]+[p.ravel() for p in self.params])
        self.alpha = self.n_params/(np.sum((params-self.prior)**2) + np.sum(Ainv_ii) + 2.*self.a)
        # self.Alpha = self.alpha*np.ones(self.n_params)
        self.Alpha = 1./((params-self.prior)**2 + Ainv_ii + 2.*self.a)

        # update output precision
        self.Beta = self.N*np.linalg.inv(yCOV + 2.*self.b*np.eye(self.n_s))
        self.Beta = (self.Beta + self.Beta.T)/2.
        # make sure precision is positive definite
        self.Beta = self.make_pos_def(self.Beta, jnp.ones(self.n_s))
        self.BetaInv = np.linalg.inv(self.Beta)

        if self.verbose:
            print("Total samples: {:.0f}, Updated regularization: {:.2e}".format(self.N, self.alpha))

    def objective(self, params):
        # initial resource concentration
        r0 = np.array(params[:self.n_r], dtype=np.float32)
        # compute negative log posterior (NLP)
        self.NLP = np.sum(self.Alpha*(params-self.prior)**2) / 2.
        # compute residuals
        self.RES = 0.

        # reshape params and convert to torch tensors
        params = self.reshape(params[self.n_r:])

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured, inputs in self.dataset:

            # for each output
            output = np.nan_to_num(self.runODE(t_eval, Y_measured, r0, params, inputs))

            # Determine error
            Y_error = output[1:, :self.n_s] - Y_measured[1:]

            # Determine SSE and gradient of SSE
            self.NLP += np.einsum('tk,kl,tl->', Y_error, self.Beta, Y_error)/2.
            self.RES += np.sum(Y_error)/self.N

        # return NLP
        return self.NLP

    def jacobian_adj(self, params):
        # initial resource concentration
        r0 = np.array(params[:self.n_r], dtype=np.float32)

        # compute gradient of negative log posterior
        grad_NLP = self.Alpha*(params-self.prior)

        # reshape params and convert to torch tensors
        params = self.reshape(params[self.n_r:])

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured, inputs in self.dataset:

            # forward pass
            output = self.runODE(t_eval, Y_measured, r0, params, inputs)

            # adjoint at measured time points
            at = self.adjoint(output, Y_measured, self.Beta)

            # gradient of NLP
            for t, out, a in zip(t_eval[1:], output[1:], at[1:]):
                grad_NLP += self.runODEA(t, out, a, params, inputs)

        # return gradient of NLP
        return grad_NLP

    def jacobian_fwd(self, params):
        # initial resource concentration
        r0 = np.array(params[:self.n_r], dtype=np.float32)

        # compute gradient of negative log posterior
        grad_NLP = self.Alpha*(params-self.prior)

        # reshape params and convert to torch tensors
        params = self.reshape(params[self.n_r:])

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured, inputs in self.dataset:

            # integrate forward sensitivity equations
            xYZ = self.runODEZ(t_eval, Y_measured, r0, params, inputs)
            output = np.nan_to_num(xYZ[0])
            Y = xYZ[1]
            Z = xYZ[2:]

            # collect gradients and reshape
            Z = np.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], -1) for Z_i in Z], -1)

            # stack gradient matrices
            G = np.nan_to_num(np.concatenate((Y, Z), axis=-1))

            # determine error
            Y_error = output[1:, :self.n_s] - Y_measured[1:]

            # sum over time and outputs to get gradient w.r.t params
            grad_NLP += self.eval_grad_NLP(Y_error, self.Beta, G[1:, :self.n_s, :])

            # # compare to finite differences
            # # d NLP(theta) d_theta_i = [NLP(theta_i + eps) - NLP(theta_i)] / eps
            # grad_NLP = self.eval_grad_NLP(Y_error, self.Beta, G[1:, :self.n_s, :])
            # eps = 1e-3
            # fd_grad_r0 = np.zeros(self.n_r)
            # for i in range(self.n_r):
            #     # error plus eps
            #     fd_r0 = np.copy(r0)
            #     fd_r0[i] += eps
            #     fd_out = self.runODE(t_eval, Y_measured, fd_r0, params)
            #     fd_err = self.log_err(fd_out[1:,:self.n_s], Y_measured[1:])
            #     fd_nll = np.einsum("ti,ij,tj->", fd_err, self.Beta, fd_err)/2.
            #
            #     # error
            #     out = self.runODE(t_eval, Y_measured, r0, params, inputs)
            #     err = self.log_err(out[1:,:self.n_s], Y_measured[1:])
            #     nll = np.einsum("ti,ij,tj->", err, self.Beta, err)/2.
            #
            #     # approximate gradient
            #     fd_grad_r0[i] = (fd_nll - nll)/eps
            #
            # # plot
            # plt.scatter(fd_grad_r0, grad_NLP[:self.n_r])
            # plt.show()
            #
            # eps = 1e-3
            # fd_grad_params = np.zeros_like(grad_NLP[self.n_r:])
            # # tricky because params are weirdly shaped
            # for i, _ in enumerate(fd_grad_params):
            #
            #     # error plus eps
            #     fd_params = np.concatenate([np.copy(p).ravel() for p in params])
            #     fd_params[i] += eps
            #     fd_params = self.reshape(fd_params)
            #     fd_out = self.runODE(t_eval, Y_measured, r0, fd_params)
            #     fd_err = self.log_err(fd_out[1:,:self.n_s], Y_measured[1:])
            #     fd_nll = np.einsum("ti,ij,tj->", fd_err, self.Beta, fd_err)/2.
            #
            #     # error
            #     out = self.runODE(t_eval, Y_measured, r0, params, inputs)
            #     err = self.log_err(out[1:,:self.n_s], Y_measured[1:])
            #     nll = np.einsum("ti,ij,tj->", err, self.Beta, err)/2.
            #
            #     # approximate gradient
            #     fd_grad_params[i] = (fd_nll - nll)/eps
            #
            # # plot
            # plt.scatter(fd_grad_params, grad_NLP[self.n_r:])
            # plt.show()

        # return gradient of NLP
        return grad_NLP

    def hessian(self, params):
        # initial resource concentration
        r0 = np.array(params[:self.n_r], dtype=np.float32)

        # reshape params and convert to torch tensors
        params = self.reshape(params[self.n_r:])

        # compute Hessian of NLP
        self.A = np.diag(self.Alpha)

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured, inputs in self.dataset:

            # integrate forward sensitivity equations
            xYZ = self.runODEZ(t_eval, Y_measured, r0, params, inputs)
            output = np.nan_to_num(xYZ[0])
            Y = xYZ[1]
            Z = xYZ[2:]

            # collect gradients and reshape
            Z = np.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], -1) for Z_i in Z], -1)

            # stack gradient matrices
            G = np.nan_to_num(np.concatenate((Y, Z), axis=-1))

            # compute Hessian
            self.A += self.A_next(G[1:, :self.n_s, :], self.Beta)

        # make sure precision is symmetric
        self.A = (self.A + self.A.T)/2.

        # make sure precision is positive definite
        self.A = self.make_pos_def(self.A, self.Alpha)

        # return Hessian
        return self.A

    def update_precision(self):

        # update parameter covariance matrix given current parameter estimate
        self.A = np.diag(self.Alpha)

        # loop over each sample in dataset
        for treatment, t_eval, Y_measured, inputs in self.dataset:

            # integrate forward sensitivity equations
            xYZ = self.runODEZ(t_eval, Y_measured, self.r0, self.params, inputs)
            output = np.nan_to_num(xYZ[0])
            Y = xYZ[1]
            Z = xYZ[2:]

            # collect gradients and reshape
            Z = np.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], -1) for Z_i in Z], -1)

            # stack gradient matrices
            G = np.concatenate((Y, Z), axis=-1)

            # compute Hessian of NLL
            self.A += self.A_next(G[1:, :self.n_s, :], self.Beta)

        # Laplace approximation of posterior precision
        self.A = (self.A + self.A.T)/2.

        # make sure precision is positive definite
        self.A = self.make_pos_def(self.A, self.Alpha)

    def update_covariance(self):
        ### Approximate / fast method ###
        self.Ainv = self.compute_Ainv(self.A)

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

    def predict_point(self, x_test, t_eval, inputs=[]):

        # convert to torch tensors
        t_eval = np.array(t_eval, dtype=np.float32)
        x_test = np.atleast_2d(np.array(x_test, dtype=np.float32))

        # make predictions given initial conditions and evaluation times
        output = np.nan_to_num(self.runODE(t_eval, x_test, self.r0, self.params, inputs))

        return output

    # Define function to make predictions on test data
    def predict(self, x_test, t_eval, inputs=[], n_std=1.):

        # integrate forward sensitivity equations
        xYZ = self.runODEZ(t_eval, np.atleast_2d(x_test), self.r0, self.params, inputs)
        output = np.nan_to_num(np.array(xYZ[0]))
        Y = xYZ[1]
        Z = xYZ[2:]

        # collect gradients and reshape
        Z = np.concatenate([Z_i.reshape(Z_i.shape[0], Z_i.shape[1], -1) for Z_i in Z], -1)

        # stack gradient matrices
        G = np.concatenate((Y, Z), axis=-1)

        # calculate covariance of each output (dimension = [steps, outputs])
        BetaInv = np.zeros([self.n_s+self.n_r, self.n_s+self.n_r])
        BetaInv[:self.n_s, :self.n_s] = self.BetaInv
        covariance = BetaInv + self.GAinvG(G, self.Ainv)

        # predicted stdv of log y
        get_diag = vmap(jnp.diag, (0,))
        stdv = n_std*np.sqrt(get_diag(covariance))

        return output[:, :self.n_s], stdv[:, :self.n_s], np.exp(output[:, self.n_s:])
