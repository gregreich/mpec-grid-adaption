%% Adaptive Grids for the Estimation of Dynamic Programming Models
% by Andreas Lanz, Gregor Reich and Ole Wilms
% Codes for monte carlo study for bus replacement model (Section 3.2.3)

clear all
close all

% Model parameters
beta = 0.99;
RC = 11.7257; 
theta11 = 2.4569;
theta2 = 1.5;
% theta2 = 0.075;
theta = [RC; theta11; theta2];

% Parameters for monte carlo simulations
NP = 100; % Number of Runs
NI = 3; % Number of Initial Guesses for each Run
theta0 = [2 10 17; 1 3 5];
if size(theta0,2) ~= NI
    error('theta0_All must have NI columns')
end



% Parameters for solution method
N_Acc = 400;    % Number of nodes for accurate solution        
N_Fixed = 17;    % Number of nodes for approximation with fixed grid and less nodes        
N = 5;         % Number of nodes for endogenous grid

N_True = 400; % number of nodes for solution of EV problem using the 'true' model parameters
   

% Cost function
c = @(x,theta11) 0.001*theta11 .*x;
% c = @(x,theta11) 0.000001*theta11.*x.^3 ;

v = @(x,theta11) -c(x,theta11);

% Maximum allowed difference in inequality constraints 
% (difference between x_i and x_i+1)
diff = 0.1;

% Error tolerance in the equi-sized sub-integrals
% (a value of 2 means, that the largest integral can't be larger then 2 times the smallest integral)
E_Tol = 0;

Method = 2; % Method 1: integral criterion; method 2: max abs
            % For Method = 2, the script first computes a solution using the integral
            % criterion (Method 1) and then uses this solution as an initial guess to
            % find the solution for Method = 2
            
J = 10; % Number of Gauss-Legendre quadrature nodes for objective function
nGL = 10; % Number of Gauss-Laguerre quadrature nodes for EV

[zi,wi] = lgwt(J,-1,1); % Gauss-Legendre nodes and weights           

Solver = 1; % 1: fmincon, 2: knitro 
options = optimoptions('fmincon','Algorithm','sqp','Display','off','MaxFunEvals',100000,'MaxIter',50);

% Approximation interval
xmin = 1;
% xmax is choosen as a multiple of the maximum value of x reached in the simulations


%% Compute the 'True' EV value given the parameters for the simulations
xmax_True = 800;

x_True = linspace(xmin,xmax_True,N_True)';
a0_True = 0*ones((N_True-1)*2,1);

if Solver == 1
    [a_True] = fmincon(@(a) 1,a0_True,[],[],[],[],[],[],@(a) constraints(a,x_True,theta,beta,v,nGL),options);
elseif Solver == 2  
    [a_True] = knitromatlab(@(a) 1,a0_True,[],[],[],[],[],[],@(a) constraints(a,x_True,theta,beta,v,nGL));
else
   error('Solver muste be 1 or 2')
end

ai_True = reshape(a_True,N_True-1,2);
EV_True = @(xi) Spline_Eval(ai_True,x_True,xi);


%% Simulate Data

Results_EndGrid = zeros(3,NI,NP);
Results_Acc = zeros(3,NI,NP);
Results_Fixed = zeros(3,NI,NP);
Results_Fixed_2 = zeros(3,NI,NP);

Converged_Paths_Endgrid = zeros(NI,NP);
Converged_Paths_Acc = zeros(NI,NP);
Converged_Paths_Fixed = zeros(NI,NP);
Converged_Paths_Fixed_2 = zeros(NI,NP);
xmax_all = zeros(NP,1);


parfor run = 1:NP

    nT = 150; % number of time periods
    nBus = 500; % number of busses

    ExpPDF = @(x,theta) theta*exp(-theta*x);
    ExpCDF = @(x,theta) 1-exp(-theta*x);
    InvExpCDF = @(p,theta) -log(1-p)/theta;

    P0 = @(x) 1./ (1 + exp( c(x,theta11) - beta.*EV_True(x) - RC + beta*EV_True(1))); 

    % Simulate data
        Rx  = unifrnd(0, 1, nT, nBus);
        Rd  = unifrnd(0, 1, nT, nBus);

        xt = zeros(nT, nBus);
        xt(1,:) = 1;
        dx = zeros(nT, nBus);
        dt = zeros(nT, nBus);

        for t = 1:nT
            dt(t,:) = (Rd(t,:) >= P0(xt(t,:)));
            for i = 1:nBus
                dx(t,i) = InvExpCDF(Rx(t,i),theta2);
                if t < nT
                    if dt(t,i) == 1
                       xt(t+1,i) = 1 + dx(t,i);
                    else 
                       xt(t+1,i) = min(xt(t,i) + dx(t,i),xmax_True);
                       xt(t+1,i) = xt(t,i) + dx(t,i);
                    end
                end
            end
        end


    xmax = 1.5*max(max(xt));
    xmax_all(run,1) = xmax;

    % Fix theta2 (could be estimated independently)
    theta2_Est = theta2;


    for ni = 1:NI


        %% MPEC estimation with fixed grid and N_Acc nodes (high accuracy)

        tic
        % Run MPEC optimization
        [VARS_Acc, LL_Acc, x_Acc,Exitflag_Acc]=MPEC_Fixed_Grid(N_Acc,xmin,xmax,theta0(:,ni),xt,dt,v,beta,nGL,theta2_Est,options,Solver);

        CompTime_Acc = toc;

        % Extract variables
        RC_Acc = VARS_Acc(1,1);
        theta11_Acc = VARS_Acc(2,1);
        a_Acc = VARS_Acc(3:end,1);
        ai_Acc = reshape(a_Acc,N_Acc-1,2);
        EV_Acc = @(xi) Spline_Eval(ai_Acc,x_Acc,xi);
        Converged_Paths_Acc(ni,run) = Exitflag_Acc;


        %% MPEC estimation with fixed grid and N_Fixed nodes
        tic

        % Run MPEC Optimization
        [VARS_Fixed, LL_Fixed, x_Fixed,Exitflag_Fixed]=MPEC_Fixed_Grid(N_Fixed,xmin,xmax,theta0(:,ni),xt,dt,v,beta,nGL,theta2_Est,options,Solver);

        CompTime_Fixed = toc;

        % Extract Variables
        RC_Fixed = VARS_Fixed(1,1);
        theta11_Fixed = VARS_Fixed(2,1);
        a_Fixed = VARS_Fixed(3:end,1);
        ai_Fixed = reshape(a_Fixed,N_Fixed-1,2);
        EV_Fixed = @(xi) Spline_Eval(ai_Fixed,x_Fixed,xi);
        Converged_Paths_Fixed(ni,run) = Exitflag_Fixed;


        %% MPEC estimation with as many nodes as used for the flexible grid
        tic

        % Run MPEC optimization
        [VARS_Fixed_2, LL_Fixed_2, x_Fixed_2,Exitflag_Fixed_2]=MPEC_Fixed_Grid(N,xmin,xmax,theta0(:,ni),xt,dt,v,beta,nGL,theta2_Est,options,Solver);

        CompTime_Fixed_2 = toc;

        % Extract Variables
        RC_Fixed_2 = VARS_Fixed_2(1,1);
        theta11_Fixed_2 = VARS_Fixed_2(2,1);
        a_Fixed_2 = VARS_Fixed_2(3:end,1);
        ai_Fixed_2 = reshape(a_Fixed_2,N-1,2);
        EV_Fixed_2 = @(xi) Spline_Eval(ai_Fixed_2,x_Fixed_2,xi_2);
        Converged_Paths_Fixed_2(ni,run) = Exitflag_Fixed_2;


        %% MPEC estimation with flexible grid

        tic
        % To compute the fleixble grid solution, we first need to compute a
        % feasible initial guess. For this we proceed as follows:
        % 1. Solve the MPEC Problem with a fixed uniform grid with N nodes
        % 2. Use the estimated Parameters RC and theta1 to find the flexible grid
        % that minimizes the L2/L_infty error. 
        % 3. Use this solutions as the initial guess for the MPEC problem with flexible grid

        % Step 1: solve MPEC optimization with fixed grid
        [VARS_Initial, LL_Initial, x_Initial]=MPEC_Fixed_Grid(N,xmin,xmax,theta0(:,ni),xt,dt,v,beta,nGL,theta2_Est,options,Solver);

        % Step 2: use estimated parameters to compute optimal flexible grid

        % Method 1: integral criterion; method 2: max abs
        % For Method = 2, the script first computes a solution using the integral
        % criterion (Method 1) and then uses this solution as an initial guess to
        % find the solution for Method = 2
        if Method == 1
            [y0_Endgrid] = compEV_givenTheta([VARS_Initial(1:2,1);theta2_Est],beta,v,nGL,N,x_Initial,VARS_Initial(3:end,1),xmin,xmax,J,zi,wi,diff,E_Tol,options,Method,Solver);
        elseif Method == 2
            [y0_Endgrid] = compEV_givenTheta([VARS_Initial(1:2,1);theta2_Est],beta,v,nGL,N,x_Initial,VARS_Initial(3:end,1),xmin,xmax,J,zi,wi,diff,E_Tol,options,1,Solver);
            [y0_Endgrid] = compEV_givenTheta([VARS_Initial(1:2,1);theta2_Est],beta,v,nGL,N,[xmin; y0_Endgrid(1:N-2,1) ;xmax],y0_Endgrid(N-1:2*(N-1)+N-2,1),xmin,xmax,J,zi,wi,diff,E_Tol,options,Method,Solver);
        end

        % Extract variables
        VARS0 = [y0_Endgrid(1:N-2,1); VARS_Initial(1:2,1);  y0_Endgrid(N-1:end,1)];
        % The first N-2 parameters are the endogenuous grid points
        % (without xmin and xmax), the next two are RC and theta1, then we have
        % (N-1)*2 coefficients of the piecewise-linear approximation and finally N-1 slack
        % variables for the Integrals/MaxAbs values in the N-1 Subintervals


        % Linear constraints:
        lb = zeros(size(VARS0,1),1);
        ub = zeros(size(VARS0,1),1);

        % RC
        lb(N-1,1)=0;
        ub(N-1,1)=Inf;

        % theta11
        lb(N,1)=0;
        ub(N,1)=Inf;

        % grid points
        lb(1:N-2,1)=1;
        ub(1:N-2,1)=xmax;

        % Coefficients for piecewise-linear approximation
        lb(N+1:2*(N-1)+N,1)=-Inf;
        ub(N+1:2*(N-1)+N,1)=Inf;

        % Slack variables for error criterion
        lb(2*(N-1)+N+1:end,1)=0;
        ub(2*(N-1)+N+1:end,1)=Inf;

        % Inequality constraints
        N_tot = size(VARS0,1); % Total Number of Variables

        % There are N-1 inequality constraints for x_i<x_i+1
        % and 2*(N-1)*(N-2) constraints on the slack variables (1-eps)*zi-zj < 0
        % and (-1-eps)*zi + zj < 0 for all i =\ j 

        A = zeros((N-1)+2*(N-1)*(N-2),N_tot);
        b = zeros((N-1)+2*(N-1)*(N-2),1);

        % Imposing that xi(1) > xmin;
        A(1,1) = -1;
        b(1,1) = -xmin;

        % Imposing that xi(end) < xmax;
        A(2,N-2) = 1;
        b(2,1) = xmax;

        % Imposing that xi(i+1) > xi(i)
        for i = 1:N-3
            A(i+2,i) = 1;
            A(i+2,i+1) = -1;
        end

        b(1:N-1,1) = b(1:N-1,1)-diff;


        % Impose restrictions on slack variables (1-eps)*zi-zj < 0
        % and (-1-eps)*zi + zj < 0 for all i =\ j
        A_slack1 = zeros((N-1)*(N-2),N-1);
        A_slack2 = zeros((N-1)*(N-2),N-1);
        help = eye(N-2);
        help2 = ones(N-2,1)*(1-E_Tol);
        help3 = ones(N-2,1)*(-1-E_Tol);

        % The following lines construct the A matrizes only for the two conditions
        % and slack variables
        A_slack1(1:N-2,:) = [help2 -help];
        A_slack2(1:N-2,:) = [help3 help];

        for i = 2:N-1
                A_slack1((i-1)*(N-2)+1:i*(N-2),:) = [-help(:,1:i-1) , help2 -help(:,i:end)];
                A_slack2((i-1)*(N-2)+1:i*(N-2),:) = [help(:,1:i-1) , help3 help(:,i:end)];
        end

        % Insert the A matrizes for the slack variables into the full A matrix 
        A(N:N+size(A_slack1,1)-1,end-size(A_slack1,2)+1:end) = A_slack1;
        A(end-size(A_slack1,1)+1:end,end-size(A_slack1,2)+1:end) = A_slack2;


        % Step 3: Run MPEC optimization for flexible grid
        if Solver == 1
            [VARS_EndGrid,LL_EndGrid,Exitflag_EndGrid] = fmincon(@(VARS) -LL_MPEC_EndGrid(VARS,xt,dt,v,beta,N,xmin,xmax),VARS0,A,b,[],[],lb,ub,@(VARS) constraints_MPEC_EndGrid(VARS,beta,v,nGL,theta2_Est,N,xmin,xmax,J,zi,wi,Method),options);

            if Exitflag_EndGrid < 1
                [VARS_EndGrid,LL_EndGrid,Exitflag_EndGrid] = fmincon(@(VARS) -LL_MPEC_EndGrid(VARS,xt,dt,v,beta,N,xmin,xmax),VARS_EndGrid,A,b,[],[],lb,ub,@(VARS) constraints_MPEC_EndGrid(VARS,beta,v,nGL,theta2_Est,N,xmin,xmax,J,zi,wi,Method),options); 
            end

        elseif Solver ==2
            [VARS_EndGrid,LL_EndGrid,Exitflag_EndGrid] = knitromatlab(@(VARS) -LL_MPEC_EndGrid(VARS,xt,dt,v,beta,N,xmin,xmax),VARS0,A,b,[],[],lb,ub,@(VARS) constraints_MPEC_EndGrid(VARS,beta,v,nGL,theta2_Est,N,xmin,xmax,J,zi,wi,Method));
        end

        CompTime_EndoGrid = toc;

        Converged_Paths_Endgrid(ni,run) = Exitflag_EndGrid;

        % Extract variable
        x_EndGrid = [xmin; VARS_EndGrid(1:N-2,1); xmax];
        RC_EndGrid = VARS_EndGrid(N-1,1);
        theta11_EndGrid = VARS_EndGrid(N,1);

        a_EndGrid = VARS_EndGrid(N+1:2*(N-1)+N,1);
        ai_EndGrid = reshape(a_EndGrid,N-1,2);
        EV_EndGrid = @(xi) Spline_Eval(ai_EndGrid,x_EndGrid,xi);


        %% Collect results

        Results_Run_Acc = [VARS_Acc(1:2,1); CompTime_Acc];
        Results_Acc(:,ni,run) = Results_Run_Acc;

        Results_Run_Fixed = [VARS_Fixed(1:2,1);  CompTime_Fixed];
        Results_Fixed(:,ni,run) = Results_Run_Fixed;

        Results_Run_Fixed_2 = [VARS_Fixed_2(1:2,1);  CompTime_Fixed_2];
        Results_Fixed_2(:,ni,run) = Results_Run_Fixed_2;

        Results_Run_EndGrid = [VARS_EndGrid(N-1:N,1); CompTime_EndoGrid];
        Results_EndGrid(:,ni,run) = Results_Run_EndGrid;

    end

end


%% Clean up results

% Only use runs that converged for each case
Converged_Paths_Endgrid_Index = find(Converged_Paths_Endgrid ==1 | Converged_Paths_Endgrid ==2);
Converged_Paths_Acc_Index = find(Converged_Paths_Acc ==1 | Converged_Paths_Acc ==2);
Converged_Paths_Fixed_Index = find(Converged_Paths_Fixed ==1 | Converged_Paths_Fixed ==2);
Converged_Paths_Fixed_2_Index = find(Converged_Paths_Fixed_2 ==1 | Converged_Paths_Fixed_2 ==2);
Not_Converged_Paths_Acc_Index = find(Converged_Paths_Acc ~=1);

% Delete all paths where the benchmark didn't converge 
for i = 1:size(Not_Converged_Paths_Acc_Index,1)
    Converged_Paths_Endgrid_Index(Converged_Paths_Endgrid_Index==Not_Converged_Paths_Acc_Index(i,1)) = [];
    Converged_Paths_Fixed_Index(Converged_Paths_Fixed_Index==Not_Converged_Paths_Acc_Index(i,1)) = [];
    Converged_Paths_Fixed_2_Index(Converged_Paths_Fixed_2_Index==Not_Converged_Paths_Acc_Index(i,1)) = [];
end

% Reshape results
Results_EndGrid_Reshape = reshape(Results_EndGrid,3,NI*NP);
Results_Acc_Reshape = reshape(Results_Acc,3,NI*NP);
Results_Fixed_Reshape = reshape(Results_Fixed,3,NI*NP);
Results_Fixed_2_Reshape = reshape(Results_Fixed_2,3,NI*NP);


% Compute means and standard deviations of the estimates
MEAN_Results_EndGrid = mean(Results_EndGrid_Reshape(:,Converged_Paths_Endgrid_Index),2);
STD_Results_EndGrid = std(Results_EndGrid_Reshape(:,Converged_Paths_Endgrid_Index),0,2);

MEAN_Results_Acc = mean(Results_Acc_Reshape(:,Converged_Paths_Acc_Index),2);
STD_Results_Acc = std(Results_Acc_Reshape(:,Converged_Paths_Acc_Index),0,2);

MEAN_Results_Fixed = mean(Results_Fixed_Reshape(:,Converged_Paths_Fixed_Index),2);
STD_Results_Fixed = std(Results_Fixed_Reshape(:,Converged_Paths_Fixed_Index),0,2);

MEAN_Results_Fixed_2 = mean(Results_Fixed_2_Reshape(:,Converged_Paths_Fixed_2_Index),2);
STD_Results_Fixed_2 = std(Results_Fixed_2_Reshape(:,Converged_Paths_Fixed_2_Index),0,2);


% Compute root mean squared errors
RMSE_EndGrid = (mean((((Results_EndGrid_Reshape(1,intersect(Converged_Paths_Endgrid_Index,Converged_Paths_Acc_Index))-Results_Acc_Reshape(1,intersect(Converged_Paths_Endgrid_Index,Converged_Paths_Acc_Index)))./Results_Acc_Reshape(1,intersect(Converged_Paths_Endgrid_Index,Converged_Paths_Acc_Index)))).^2) + mean((((Results_EndGrid_Reshape(2,intersect(Converged_Paths_Endgrid_Index,Converged_Paths_Acc_Index))-Results_Acc_Reshape(2,intersect(Converged_Paths_Endgrid_Index,Converged_Paths_Acc_Index)))./Results_Acc_Reshape(2,intersect(Converged_Paths_Endgrid_Index,Converged_Paths_Acc_Index)))).^2))^0.5;
RMSE_Fixed = (mean((((Results_Fixed_Reshape(1,intersect(Converged_Paths_Fixed_Index,Converged_Paths_Acc_Index))-Results_Acc_Reshape(1,intersect(Converged_Paths_Fixed_Index,Converged_Paths_Acc_Index)))./Results_Acc_Reshape(1,intersect(Converged_Paths_Fixed_Index,Converged_Paths_Acc_Index)))).^2) + mean((((Results_Fixed_Reshape(2,intersect(Converged_Paths_Fixed_Index,Converged_Paths_Acc_Index))-Results_Acc_Reshape(2,intersect(Converged_Paths_Fixed_Index,Converged_Paths_Acc_Index)))./Results_Acc_Reshape(2,intersect(Converged_Paths_Fixed_Index,Converged_Paths_Acc_Index)))).^2))^0.5;
RMSE_Fixed_2 = (mean((((Results_Fixed_2_Reshape(1,intersect(Converged_Paths_Fixed_2_Index,Converged_Paths_Acc_Index))-Results_Acc_Reshape(1,intersect(Converged_Paths_Fixed_2_Index,Converged_Paths_Acc_Index)))./Results_Acc_Reshape(1,intersect(Converged_Paths_Fixed_2_Index,Converged_Paths_Acc_Index)))).^2) + mean((((Results_Fixed_2_Reshape(2,intersect(Converged_Paths_Fixed_2_Index,Converged_Paths_Acc_Index))-Results_Acc_Reshape(2,intersect(Converged_Paths_Fixed_2_Index,Converged_Paths_Acc_Index)))./Results_Acc_Reshape(2,intersect(Converged_Paths_Fixed_2_Index,Converged_Paths_Acc_Index)))).^2))^0.5;

% Compute number of converged paths
Number_Converged_Paths_Endgrid = sum(sum(Converged_Paths_Endgrid ==1 | Converged_Paths_Endgrid ==2));
Number_Converged_Paths_Acc = sum(sum(Converged_Paths_Acc ==1 | Converged_Paths_Acc ==2));
Number_Converged_Paths_Fixed = sum(sum(Converged_Paths_Fixed ==1 | Converged_Paths_Fixed ==2));
Number_Converged_Paths_Fixed_2 = sum(sum(Converged_Paths_Fixed_2 ==1 | Converged_Paths_Fixed_2 ==2));


% Collect results
Results_table = [MEAN_Results_Acc(1) MEAN_Results_Fixed_2(1) MEAN_Results_Fixed(1)  MEAN_Results_EndGrid(1);...
    STD_Results_Acc(1) STD_Results_Fixed_2(1) STD_Results_Fixed(1)  STD_Results_EndGrid(1);...
    MEAN_Results_Acc(2) MEAN_Results_Fixed_2(2) MEAN_Results_Fixed(2)  MEAN_Results_EndGrid(2);...
    STD_Results_Acc(2) STD_Results_Fixed_2(2) STD_Results_Fixed(2)  STD_Results_EndGrid(2);...
    0 RMSE_Fixed_2 RMSE_Fixed  RMSE_EndGrid;...
    MEAN_Results_Acc(3) MEAN_Results_Fixed_2(3) MEAN_Results_Fixed(3)  MEAN_Results_EndGrid(3);...
    0 MEAN_Results_Fixed_2(3)/MEAN_Results_EndGrid(3) MEAN_Results_Fixed(3)/MEAN_Results_EndGrid(3)  MEAN_Results_EndGrid(3)/MEAN_Results_EndGrid(3);...
    N_Acc N N_Fixed N;...
    Number_Converged_Paths_Acc/(NP*NI) Number_Converged_Paths_Fixed_2/(NP*NI) Number_Converged_Paths_Fixed/(NP*NI) Number_Converged_Paths_Endgrid/(NP*NI)];

% Display results

disp('Table 3/5')
disp(Results_table)

