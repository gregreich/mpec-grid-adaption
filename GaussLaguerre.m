function [x,w]=GaussLaguerre(n)
% Nodes and weights for Gauss-Laguerre quadrature 
%
% input:   n - number of gridpoints
%    
% output:  x - the (Laguerre) grid (n x 1)
%          w - the corresponding weights (1 x n)
% 
% From: Edward Neuman
% http://www.math.siu.edu/matlab/tutorial5.pdf
%
d=-(1:n-1);
f=1:2:2*n-1;
J=diag(d,-1)+diag(f)+diag(d,1);
[u,v]=eig(J);
[x,j]=sort(diag(v));
w=(u(1,:).^2);
w=w(j);