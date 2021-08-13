function [y] =compEV_givenTheta(theta,beta,v,nGL,N,x0,a0,xmin,xmax,J,zi,wi,diff,E_Tol,options,Method,Solver)

% Compute the EV coefficients a and optimal nodes x given a fixed value for theta

% Collect variables
x0 = x0(2:end-1,1);
y0 = [x0;a0];

% The N-1 additional variables are the slack variables for integral (max abs) values in the subintervals
y0 = [y0;0.1*ones(N-1,1)];
N_tot = size(y0,1); % Total Number of Variables

% There are N-1 inequality constraints_EndoGrid for x_i<x_i+1
% and 2*(N-1)*(N-2) constraints_EndoGrid on the slack variables (1-eps)*zi-zj < 0
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

% Impose Restrictions on slack variables (1-eps)*zi-zj < 0
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

% Insert the A matrizes for the slack variables into the A matrix of the
% optimization

A(N:N+size(A_slack1,1)-1,end-size(A_slack1,2)+1:end) = A_slack1;
A(end-size(A_slack1,1)+1:end,end-size(A_slack1,2)+1:end) = A_slack2;

if Solver == 1
    [y] = fmincon(@(y) 1,y0,A,b,[],[],[],[],@(y) constraints_EndoGrid(y,theta,beta,v,nGL,N,xmin,xmax,J,zi,wi,Method),options);
elseif Solver ==2
    [y] = knitromatlab(@(y) 1,y0,A,b,[],[],[],[],@(y) constraints_EndoGrid(y,theta,beta,v,nGL,N,xmin,xmax,J,zi,wi,Method));
end
