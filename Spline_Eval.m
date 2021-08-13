function [g]=Spline_Eval(ai,xi,x)

% ai: Spline Coefficients (N-1 x J matrix)
% xi: interpolatio nodes (N x 1 vector)
% x: evaluation node (k x j matrix)


J = size(ai,2);

k = size(x,1);
j = size(x,2);
N = size(xi,1);

x = x(:);

% Search for corresponding interval

[~, index] = histc(x,[sort(xi); inf]);

% Account for extrapolation
index(index==0) = 1;
index(index>N-1) = N-1;


d=0:J-1;
eval = sum(ai(max(index,1),:).*(repmat(x,1,J).^repmat(d,k*j,1)),2);


g = reshape(eval,k,j);














