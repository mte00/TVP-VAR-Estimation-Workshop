function QA = gibbs2Q(TQ,DF,N,L)
% function QA = GIBBS2Q(TQ,DF,N,L);
% This file executes the second stage of the Gibbs sampler, generating random draws for Q using the
% inverse-Wishart density. The scale matrix is T times the usual covariance estimator, and
% E(QA) is approximately (1/T)QR
%
% DF = degrees of freedom
% inv(Q) is the scale matrix for Q
% TQ is N(1+NL) x N(1+NL)
%
PS = real(sqrtm(inv(TQ)));   %taking the square root of the inverse of the scale matrix
u = randn(N*(1+N*L),DF);     %drawing random numbers from a standard normal
QA = inv(PS*u*u'*PS');       %to obtain a draw from the inverted Wishart:
                             % (Q^-.5*u*u'*Q^-.5)^(-1)
% the Wishart density is a multivariate counterpart to the Chi-squared distribution:
% u ~ N(0,1) --> u^2 ~ chi-square(1)
