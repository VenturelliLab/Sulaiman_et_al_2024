function y = CostFcn(theta,data,time,model,lambda)
% data: a Nx1 vector of OD
% time: a Nx1 vector of time
% model: a function handle to simulate dynamics
% lambda: regularization parameter (here uses L2 regularization)

[t_sim,x] = ode15s(@(t,x) model(t,x,theta),time,x0);

x = interp1(t_sim,x,time);

error = norm(x-data,2);

y = error + lambda*norm(theta,2);