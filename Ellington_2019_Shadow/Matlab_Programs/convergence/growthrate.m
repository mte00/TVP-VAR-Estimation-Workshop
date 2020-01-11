function Y=growthrate(X,L)
%
%		Luca Benati
%		Bank of England
%		Monetary Assessment and Strategy Division
%		December 2002
%
% function Y=growthrate(X,L)
% This program computes the L-period growth rates for a previously defined matrix of series X. If nargin==1, or L=1,
% you get the simple period-to-period growth rates, otherwise you get the L-period growth rates.
%
[T,N]=size(X);
%
if nargin==1
    L=1;
else
    L=L;
end
for kk=1:N
    for tt=L+1:T
        Y(tt,kk)=(X(tt,kk)-X(tt-L,kk))/X(tt-L,kk);
    end
end
Y=Y(L+1:T,:);