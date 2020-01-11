function [yN,yC,yI]=cffilter(y,XX,pl,pu)
%
%		Luca Benati
%		Bank of England
%		Monetary Assessment and Strategy Division
%		Last modified: June 2004
%
% function [yN,yC,yI]=cffilter(y,XX,pl,pu)
% This program performs band-pass filtering of a series of interest via the algorithm designed by:
% L. J. Christiano and T. Fitzgerald (2003), 'The Band-Pass Filter', International Economic Review, 44(2), 435-465
%
%                                              Input of the program is:
% y          = the series of interest
% XX         = has to be set to 'Q' if the data are quarterly, to 'M' if the data are monthly, and to 'A' if the data are annual
% pl,pu      = the lower and upper bounds of the frequency band of interest, expressed in months, quarters, or years
%              (these inputs are optional, if you don't put them, they are set equal to the standard values of 1.5
%              and 8 years.
%
%                                              Output of the program is:
%
% [yN,yC,yI] = the band-pass filtered trend, cycle, and irregular components
%
[T,K]=size(y);
%
if nargin<4
    if XX=='Q'
        pl=6;
        pu=32;
    elseif XX=='M'
        pl=18;
        pu=96;
    elseif XX=='A'
        pl=1.5;
        pu=8;
    else
        disp('You have to specify the frequency of the data:')
        disp('the program is being terminated')
        yN=NaN*size(y);
        yC=NaN*size(y);
        yI=NaN*size(y);
        return
    end
end
%
A=(2*pi)/pu;
B=(2*pi)/pl;
%
for kk=1:K
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (1) Here the business-cycle component:
    B0=(B-A)/pi;
    % Here we compute the Bj's.
    for j=1:T
        Bj(j,1)=(sin(B*j)-sin(A*j))/(j*pi);
    end
    for t=1:1
        B_last=-0.5*B0-sum(Bj(1:T-2));
        % Here we compute the vector of the filter weights.
        W=[0.5*B0 (Bj(1:T-2))' B_last]';
        % Here we compute the band-pass filtered value.
        yC(t,kk)=W'*y(:,kk);
    end
    for t=2:T-1
        % Here we compute the first and last term of the vector of weights.
        B_last=-(0.5*B0+sum(Bj(1:T-t-1)));
        B_first=-B_last-B0-sum(Bj(1:t-2))-sum(Bj(1:T-t-1));
        % Here we compute the vector of the filter weights.
        W=[B_first (flipud(Bj(1:t-2)))' B0 Bj(1:T-t-1)' B_last]';
        % Here we compute the band-pass filtered value.
        yC(t,kk)=W'*y(:,kk);
    end
    for t=T:T
        B_first=-0.5*B0-sum(Bj(1:T-2));
        % Here we compute the vector of the filter weights.
        W=[B_first (flipud(Bj(1:t-2)))' 0.5*B0]';
        % Here we compute the band-pass filtered value.
        yC(T,kk)=W'*y(:,kk);
    end
end
%
if nargout==1
    yN=yC;
    yI=yC;
    return
end
%
for kk=1:K
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (2) Here the irregular component:
    B0=(pi-B)/pi;
    % Here we compute the Bj's.
    for j=1:T
        Bj(j,1)=(sin(pi*j)-sin(B*j))/(j*pi);
    end
    for t=1:1
        B_last=-0.5*B0-sum(Bj(1:T-2));
        % Here we compute the vector of the filter weights.
        W=[0.5*B0 (Bj(1:T-2))' B_last]';
        % Here we compute the band-pass filtered value.
        yI(t,kk)=W'*y(:,kk);
    end
    for t=2:T-1
        % Here we compute the first and last term of the vector of weights.
        B_last=-(0.5*B0+sum(Bj(1:T-t-1)));
        B_first=-B_last-B0-sum(Bj(1:t-2))-sum(Bj(1:T-t-1));
        % Here we compute the vector of the filter weights.
        W=[B_first (flipud(Bj(1:t-2)))' B0 Bj(1:T-t-1)' B_last]';
        % Here we compute the band-pass filtered value.
        yI(t,kk)=W'*y(:,kk);
    end
    for t=T:T
        B_first=-0.5*B0-sum(Bj(1:T-2));
        % Here we compute the vector of the filter weights.
        W=[B_first (flipud(Bj(1:t-2)))' 0.5*B0]';
        % Here we compute the band-pass filtered value.
        yI(T,kk)=W'*y(:,kk);
    end
    % (3) Here the trend component:
    yN(:,kk)=y(:,kk)-yC(:,kk)-yI(:,kk);
end







