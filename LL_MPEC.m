function [LL,num,den]=LL_MPEC(VARS,xt,dt,v,beta,x)

% Extract variables
RC = VARS(1,1);
theta11 = VARS(2,1);

a = VARS(3:end,1);
N = size(x,1);
ai = reshape(a,N-1,2);

xt_plus = xt(2:end,:);
xt_minus = xt(1:end-1,:);
dt_plus = dt(2:end,:);

% Compute EV

s0 =  exp(v(xt_plus, theta11) + beta * Spline_Eval(ai,x,xt_plus));
s1 =  exp(-RC + beta * Spline_Eval(ai,x,1));

num = dt_plus.*s1 + (1-dt_plus).*s0;
den = s1 + s0;

EV_sum = sum(sum(log(num./den)));

LL = EV_sum;


