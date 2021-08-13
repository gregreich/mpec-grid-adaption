function [c,ceq]=constraints(a,x,theta,beta,v,nGL)


% Extract variables
RC = theta(1,1);
theta11 = theta(2,1);
theta2 = theta(3,1);

N = size(x,1);
ai = reshape(a,N-1,2);
xk = x;

% Gauss-Laguerre nodes and weights
[zi,wi]=GaussLaguerre(nGL);


% EV constraints
s0 = @(dx,xk,ai,x) exp(v(dx+xk, theta11) + beta * Spline_Eval(ai,x,dx+xk));
s1 = @(ai,x) exp(-RC + beta * Spline_Eval(ai,x,1));

f = @(dx,xk,ai,x) theta2 * log(s0(dx,xk,ai,x) + s1(ai,x));

% Residual function
R = Spline_Eval(ai,x,xk) - 1/theta2*f(repmat(zi'/theta2,N,1),repmat(xk,1,nGL),ai,x)*wi';


% Constraints for function approximation
d=0:1;
poly = repmat(x,1,2).^(repmat(d,N,1));
cond_approx = sum(ai(1:end-1,:).*poly(2:end-1,:),2) - sum(ai(2:end,:).*poly(2:end-1,:),2);


c=[]; 
ceq= [R; cond_approx];