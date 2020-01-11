function u=bootstrap(U)
%
% Luca Benati
% Bank of England
% Monetary Assessment and Strategy Division
% February 2004
%
% function u=bootstrap(U)
% This program bootstraps the residuals matrix U (a NxK matrix), giving you a Kx1 vector of bootstrapped residuals u.
%                                             Input of the program is:
% U = a NxK matrix of regression residuals. Each column contains the residuals from one equation
%                                             Output of the program is:
% u = a Kx1 vector of bootstrapped residuals
%
[N,K]=size(U);
k=fix(rand*(N-0.01))+1;
u=U(k,:);