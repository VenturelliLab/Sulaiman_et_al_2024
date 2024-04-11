function dx = LogisticGrowth(t,x,para)

r = para(1);
a = para(2);

dx = 0;

dx(1) = x(1)*(r - a*x(1));