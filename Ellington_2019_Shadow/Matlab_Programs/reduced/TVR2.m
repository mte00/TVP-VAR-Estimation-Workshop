function [Rsquared_t1, Rsquared_t4, Rsquared_t8, detOM] = TVR2(DFILE2,N,L)
% This function computes time-varying R2 statistics as in Cogley, Primiceri
% and Sargent (2010) AEJ.

% Step 1: Compute the VAR reduced form covariance matrix 
% \Omega_{t|T} = A^{-1}_{t|T}H_{t|T}A^{-1}_{t|T}
% and retrieve the \Theta_{t|T} matrices convert to companion form
% \tilde{\Theta_{t|T}}
% Step 2: Compute Time-varying R squared statistic over the draws for each
% point in time.
%
% R^{2}_{x,t} = 1- 
% (e_{x}[\sum^{j-1}_{h=0}(\tilde{\Theta_{t|T}})\Omega_{t|T}(\tilde{\Theta_{t|T}}')]e_{x}')/
% (e_{x}[\sum^{\infty}_{h=0}(\tilde{\Theta_{t|T}})\Omega_{t|T}(\tilde{\Theta_{t|T}})]e_{x}')

load(DFILE2)
NN = size(SD,3)-1;
T = size(SD,2);
% Compute \Omega_{t|T}
Omega = zeros(N,N,T,NN);
detOM = zeros(NN,T);
for i = 1:NN
    for jjj = 1:T
        Omega(:,:,jjj,i) = (chofac(N,AA(:,jjj,i))\diag(OM(jjj,:,i)))*inv(chofac(N,AA(:,jjj,i)))';
        detOM(i,jjj) = log(det(Omega(:,:,jjj,i)));
    end
end

sely = [1 0 0 0 0 0 0 0];
selp = [0 1 0 0 0 0 0 0];
seli = [0 0 1 0 0 0 0 0];
selm = [0 0 0 1 0 0 0 0];

% Sort THETA MAT to conformable form then put as companion form
BETA = zeros(N,2*N,T,NN);
for i = 1:NN
    for jjj = 1:T
        temp = reshape(SD(:,jjj,i)',1+N*L,N)';
        temp = temp(:,2:end);
        BETA(:,:,jjj,i) = temp;
        temp = [];
    end
end


% Instantaneous variance
S = [eye(N*(L-1)), zeros(N*(L-1),N)];
Rsquared_t1 = zeros(T, NN, N);
Rsquared_t4 = Rsquared_t1;
Rsquared_t8 = Rsquared_t1;
R = zeros(N*L,N*L);
for i = 1:NN
    %BB = [SD(2:1+N*L,1,i)'; SD(2+(1+N*L):2*(1+N*L),1,i)'; SD(2+2*(1+N*L):3*(1+N*L),1,i)';...
    %    SD(2+3*(1+N*L):4*(1+N*L),1,i); S];
    BB = [BETA(:,:,1,i); S];
    R(1:N,1:N) = Omega(:,:,1,i);
    VV = doublej(BB,R);
    Rsquared_t1(1,i,1) = (sely*BB*VV*BB'*sely')/(sely*VV*sely');
    Rsquared_t1(1,i,2) = (selp*BB*VV*BB'*selp')/(selp*VV*selp');
    Rsquared_t1(1,i,3) = (seli*BB*VV*BB'*seli')/(seli*VV*seli');
    Rsquared_t1(1,i,4) = (selm*BB*VV*BB'*selm')/(selm*VV*selm');
    Rsquared_t4(1,i,1) = (sely*(BB^4)*VV*(BB^4)'*sely')/(sely*VV*sely');
    Rsquared_t4(1,i,2) = (selp*(BB^4)*VV*(BB^4)'*selp')/(selp*VV*selp');
    Rsquared_t4(1,i,3) = (seli*(BB^4)*VV*(BB^4)'*seli')/(seli*VV*seli');
    Rsquared_t4(1,i,4) = (selm*(BB^4)*VV*(BB^4)'*selm')/(selm*VV*selm');
    Rsquared_t8(1,i,1) = (sely*(BB^8)*VV*(BB^8)'*sely')/(sely*VV*sely');
    Rsquared_t8(1,i,2) = (selp*(BB^8)*VV*(BB^8)'*selp')/(selp*VV*selp');
    Rsquared_t8(1,i,3) = (seli*(BB^8)*VV*(BB^8)'*seli')/(seli*VV*seli');
    Rsquared_t8(1,i,4) = (selm*(BB^8)*VV*(BB^8)'*selm')/(selm*VV*selm');
    for t = 2:T
        %BB = [SD(2:1+N*L,t,i)'; SD(2+(1+N*L):2*(1+N*L),t,i)'; SD(2+2*(1+N*L):3*(1+N*L),t,i)';...
        %SD(2+3*(1+N*L):4*(1+N*L),t,i); S];
        BB = [BETA(:,:,t,i); S];
        R(1:N,1:N) = Omega(:,:,t,i);
        VV = doublej(BB,R);
        Rsquared_t1(t,i,1) = (sely*BB*VV*BB'*sely')/(sely*VV*sely');
        Rsquared_t1(t,i,2) = (selp*BB*VV*BB'*selp')/(selp*VV*selp');
        Rsquared_t1(t,i,3) = (seli*BB*VV*BB'*seli')/(seli*VV*seli');
        Rsquared_t1(t,i,4) = (selm*BB*VV*BB'*selm')/(selm*VV*selm');
        Rsquared_t4(t,i,1) = (sely*(BB^4)*VV*(BB^4)'*sely')/(sely*VV*sely');
        Rsquared_t4(t,i,2) = (selp*(BB^4)*VV*(BB^4)'*selp')/(selp*VV*selp');
        Rsquared_t4(t,i,3) = (seli*(BB^4)*VV*(BB^4)'*seli')/(seli*VV*seli');
        Rsquared_t4(t,i,4) = (selm*(BB^4)*VV*(BB^4)'*selm')/(selm*VV*selm');
        Rsquared_t8(t,i,1) = (sely*(BB^8)*VV*(BB^8)'*sely')/(sely*VV*sely');
        Rsquared_t8(t,i,2) = (selp*(BB^8)*VV*(BB^8)'*selp')/(selp*VV*selp');
        Rsquared_t8(t,i,3) = (seli*(BB^8)*VV*(BB^8)'*seli')/(seli*VV*seli');
        Rsquared_t8(t,i,4) = (selm*(BB^8)*VV*(BB^8)'*selm')/(selm*VV*selm');
    end
end
clear SD AA OM 