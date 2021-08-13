function [VARS, LL, x,Exitflag]=MPEC_Fixed_Grid(N,xmin,xmax,theta0,xt,dt,v,beta,nGL,theta2_Est,options,Solver)


x = linspace(xmin,xmax,N)';

VARS0 = [theta0;0*ones((N-1)*2,1)];

% Constraints
lb = zeros(size(VARS0,1),1);
ub = zeros(size(VARS0,1),1);

% RC
lb(1,1)=0;
ub(1,1)=inf;

% theta11
lb(2,1)=0;
ub(2,1)=inf;


% EV
lb(3:end,1)=-Inf;
ub(3:end,1)=Inf;


% Run MPEC optimization
if Solver == 1
    [VARS,LL,Exitflag] = fmincon(@(VARS) -LL_MPEC(VARS,xt,dt,v,beta,x),VARS0,[],[],[],[],lb,ub,@(VARS) constraints_MPEC(VARS,x,beta,v,nGL,theta2_Est),options);
elseif Solver == 2
    [VARS,LL,Exitflag] = knitromatlab(@(VARS) -LL_MPEC(VARS,xt,dt,v,beta,x),VARS0,[],[],[],[],lb,ub,@(VARS) constraints_MPEC(VARS,x,beta,v,nGL,theta2_Est));
end