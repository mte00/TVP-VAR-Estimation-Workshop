function V = stackA(A)
% function V = stackA(A)
% This extracts the coefficients below the diagonal from a lower triangular
% matrix. They are vectorised row by row, first the second row, then the
% third, etc.
%
N=size(A,1);
V=[];
for jj=2:N
    V=[V' A(jj,1:jj-1)]';
end
