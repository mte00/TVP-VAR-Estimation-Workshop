function [H,SV]=stochvol(y,NG,NB,D,presample,alpha)
%
% Luca Benati
% Bank of England
% Monetary Assessment and Strategy Division
% November 2004
%
% function [H,SV]=stochvol(y,NG,NB,D,presample,alpha)
% This program estimates a stochastic volatility model for a time series of innovations, based on the algorithm
% of Jacquier, Polson, and Rossi. The model is:
%                                     ln( sigma(t) ) = ln( sigma(t-1) ) + u(t)
% where sigma(t) is the standard deviation of the innovation at time t.
%
%                                               Input of the program is:
% y  = a Nx1 vector
% NG = Number of total iterations
% NB = Number of burn-in iterations
% D  = Sampling interval from the Markov chain
% presample = fraction of the overall sample that is used as pre-sample
% alpha = One minus the coverage of the confidence intervals -- eg 0.1
%                                               Output of the program is:
% H  = Stochastic volatilities
% SV  = Standard deviation of volatilities innovation
%
path(path,'n:\users\lb1\matprog\stats')
path(path,'n:\users\lb1\matprog\')
path(path,'N:\Users\lb1\gibbs\cogmorsar')
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
T = length(y);
% Presample:
y0=y(1:fix(T*presample));
y=y(fix(T*presample)+1:T);
T = length(y);
%
% Priors for SVOL parameters:
%
% Priors for sv (inverse gamma) (standard dev for volatility innovation)
sv0 = .01;
v0 = 1;
d0 = sv0^2;
eh = zeros(T,1); % volatility innovations
% Priors for log h0 (normal); ballpark numbers
ss0 = 10;
mu0 = log(std(y0)^2);
%
% Initialize gibbs arrays:
SV = zeros(NG+NB+1,1); % Standard error for volatility innovation
H = eps*ones(T+1,NG+NB+1); % Stochastic volatilities 
%
% Initial for log(h(t))
lh(1,1) = mu0;
lh(2:T+1,:) = log(y.^2);
H0 = exp(lh);
%
H(:,1)=H0;
SV(1,1)=mu0;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                                   Here the MCMC algorithm:
iter=1;
while iter <= NG+NB
    % R conditional on states and data (svol programs)
    lh = log(H(:,iter));
    % sv|b,lh,ch,y
    eh = diff(lh);
    v = ig2(v0,d0,eh);
    SV(iter+1,1) = v^.5;
    %
    % lh|ch,sv,b1,y
    H(1,iter+1) = svmh0(H(2,iter),0,1,SV(iter+1,1),mu0,ss0);
    for t = 2:T
        H(t,iter+1) = svmh(H(t+1,iter),H(t-1,iter+1),0,1,SV(iter+1,1),y(t-1),H(t,iter)); 
    end
    H(T+1,iter+1) = svmhT(H(T,iter+1),0,1,SV(iter+1,1),y(T),H(T+1,iter)); 
    iter=iter+1;
end
%
H=H(:,NB+2:NG+NB+1);
SV=SV(NB+2:NG+NB+1,:);
%
H=H(:,D:D:NG);
SV=SV(D:D:NG,:);
%
SV=sort(SV);
H=sort(H,2);
%

N=size(SV,1);
%
N1=fix(N*alpha/2);
N2=fix(N*(1-alpha/2));
%
SV=[SV(N1) median(SV) SV(N2)]';
H=[H(:,N1) median(H,2) H(:,N2)];