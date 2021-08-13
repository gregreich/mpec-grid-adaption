function [LL,num,den]=LL_MPEC_EndGrid(VARS,xt,dt,v,beta,N,xmin,xmax)


% Extract variables
x = [xmin; VARS(1:N-2,1); xmax];

RC = VARS(N-1,1);
theta11 = VARS(N,1);

a = VARS(N+1:2*(N-1)+N,1);

ai = reshape(a,N-1,2);

xt_plus = xt(2:end,:);
dt_plus = dt(2:end,:);

% Compute EV 

s0 =  exp(v(xt_plus, theta11) + beta * Spline_Eval(ai,x,xt_plus));
s1 =  exp(-RC + beta * Spline_Eval(ai,x,1));

num = dt_plus.*s1 + (1-dt_plus).*s0;
den = s1 + s0;

EV_sum = sum(sum(log(num./den)));

LL = EV_sum  ;


