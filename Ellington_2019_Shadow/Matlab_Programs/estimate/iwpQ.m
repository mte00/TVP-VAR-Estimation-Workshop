function [TQ,DF] = iwpQ(theta,T,TQ0,df)
%function [TQ,DF] = IWPQ(theta,N,T,L,TQ0,df);
%
% This file computes posterior estimates of TQ for an informative prior, in which Q is inverse
% wishart with degrees of freedom df and scale matrix TQ0.  The posterior density is also inverse 
% wishart, with scale matrix TQ and degrees of freedom DF
%
% N = number of equations
% T = number of time periods
% L = number of lags
%
% Conditional on a specific time path of theta, the innovations to the random walk parameters are observable:
v(:,2:T) = theta(:,2:T) - theta(:,1:T-1);     %measurement innovations
% The other conditioning variables are redundant in constructing the
% conditional distribution of Q because the random walk increment v is
% uncorrelated with the other error terms
% Posterior estimate of Q: scale matrix
TQ = TQ0 + v*v';   % v*v' is the sum of squared residuals
% Posterior degress of freedom:
DF = df + T;