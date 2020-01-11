function [h,R] = svmh(hlead,hlag,alpha,delta,sv,yt,hlast)
% h = svmh(hlead,hlag,alpha,delta,sv,y,hlast);
%
% This file returns a draw from the posterior conditional density for the stochastic volatility parameter at time t.
% This is conditional on adjacent realizations, hlead and hlag, as well as the data and parameters of the svol
% process. hlast is the previous draw in the chain, and is used in the acceptance step. R is a dummy variable that
% takes a value of 1 if the trial is rejected, 0 if accepted. Following JPR (1994), we use a MH step, but with a 
% simpler log-normal proposal density. (Their proposal is coded in jpr.m.) 
%
% Mean and variance for log(h) (proposal density) (See equation (81) in
% Appendix to 'Drift and Volatilities ...')
% alpha=0 and delta=1 since we assume geometric random walk without drift!
%
% conditional mean of ln(h)
mu = alpha*(1-delta) + delta*(log(hlead)+log(hlag))/(1+delta^2);
% conditional variance of ln(h)
ss = (sv^2)/(1+delta^2);
%
% Candidate draw from lognormal:
htrial = exp(mu + (ss^.5)*randn(1,1));
%
% Acceptance probability: yt are the orthogonalized VAR innovations
lp1 = -0.5*log(htrial) - (yt^2)/(2*htrial);
lp0 = -0.5*log(hlast) - (yt^2)/(2*hlast);
accept = min(1,exp(lp1 - lp0));  %ratio of conditional likelihood

u = rand(1);
if u <= accept,
   h = htrial;
   R = 0;   % R is a dummy varible to check the rejection rate of algorithm
else
   h = hlast;
   R = 1;
end

% NOTE: This algorithm is applied on a date-by-date basis to each element
% of u(t) i.e. the volatility states are drawn one at a time. This
% drastically increases the autocorrelations of the draws and decreases
% efficiency (as pointed out by Carter and Kohn (1994)). See Primiceri
% (2005) for an alternative approach.
