function V = chovec(A)
% function V = chovec(A)
% This extracts the Cholesky coefficients from a lower triangular Cholesky
% matrix. They are vectorised column by column. First the first column,
% then the second column, etc.
%
N=size(A,1);   %number of rows
V=[];
for jj=1:N-1
    V=[V; A(jj+1:N,jj)];
end
