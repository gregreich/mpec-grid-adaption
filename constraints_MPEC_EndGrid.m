function [c,ceq,g]=constraints_MPEC_EndGrid(VARS,beta,v,nGL,theta2,N,xmin,xmax,J,zi,wi,Method)


% Extract variables
x = [xmin; VARS(1:N-2,1); xmax];

RC = VARS(N-1,1);
theta11 = VARS(N,1);

a = VARS(N+1:2*(N-1)+N,1);
z = VARS(2*(N-1)+N+1:end,1);

ai = reshape(a,N-1,2);
xk = x;

% Gauss-Laguerre nodes and weights
[zi_Laguerre,wi_Laguerre]=GaussLaguerre(nGL);


%% EV constraints
s0 = @(dx,xk,ai,x) exp(v(dx+xk, theta11) + beta * Spline_Eval(ai,x,dx+xk));
s1 = @(ai,x) exp(-RC + beta * Spline_Eval(ai,x,1));

f = @(dx,xk,ai,x) theta2 * log(s0(dx,xk,ai,x) + s1(ai,x));

% Residual function
R = @(xi) Spline_Eval(ai,x,xi) - 1/theta2*f(repmat(zi_Laguerre'/theta2,size(xi,1),1),repmat(xi,1,nGL),ai,xk)*wi_Laguerre';


%% Constraints for function approximation
d=0:1;
poly = repmat(x,1,2).^(repmat(d,N,1));

cond_approx = sum(ai(1:end-1,:).*poly(2:end-1,:),2) - sum(ai(2:end,:).*poly(2:end-1,:),2);


%% Equi-oszilation constraints

N = N-1;
if Method == 1

    % Integral Criterion:

    % Integral bouds
    lb = x(1:end-1,1);
    ub = x(2:end,1);

    % Integration nodes (JxN-1) (Gauss-Legendre Quadrature)
    % Each column consists of the J quadrature nodes of the corresponding (N-1)
    % intervals
    xij = (repmat(zi,1,N)+1).*(repmat(ub',J,1) - repmat(lb',J,1))./2 + repmat(lb',J,1);
    g = (wi'* abs(  reshape(R(xij(:)),J,N ) ) )' .* ((ub-lb)./2);


elseif Method == 2

    % Max abs:

    lb = sort(x(1:end-1,1));
    ub = sort(x(2:end,1));
    
    % Grid search for max abs
    
    N_Eval = 10; % number of grid points
    i_Eval = [1:N_Eval];
    k = size(ub,1);
    
    x_eval = repmat(lb,1,N_Eval) + (repmat(ub,1,N_Eval)-repmat(lb,1,N_Eval)).*(repmat(i_Eval,k,1)-1)./(N_Eval-1);
    
    g_help = abs( reshape(R(x_eval(:)),k,N_Eval) );
    g = max(g_help,[],2);

end


c=[]; 
ceq= [R(xk); cond_approx; z-g];
