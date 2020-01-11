function [S0,P0,P1]=kfP(YS,XS,Q,AA,H,SI,PI,T,N,L)
% function [S0,P0,P1]=kfP(YS,XS,Q,AA,H,SI,PI,T,N,L);
% This file performs the forward Kalman filter recursions for the random coefficients VAR.  R is time varying,
% depending on the stochastic volatilities: R = inv(A)*H(t)*inv(A)'. SI, PI are the initial values for the
% recursions, S(1|0) and P(1|0) to initialize the Kalman filter
%
S0 = zeros(N*(1+N*L),T);           % current estimate of the state, S(t|t)
P0 = zeros(N*(1+N*L),N*(1+N*L),T); % current estimate of the covariance matrix, P(t|t)
P1 = zeros(N*(1+N*L),N*(1+N*L),T); % one-step ahead covariance matrix, P(t|t-1)
%
% Date 1
P1(:,:,1) = PI;     % P(1|0)
A=chofac(N,AA(:,2));
R = (A\diag(H(2,:)))*inv(A)'; % R is shifted one period rel to observables
K = (P1(:,:,1)*XS(:,:,1))*inv(XS(:,:,1)'*P1(:,:,1)*XS(:,:,1) + R); % K(1)
P0(:,:,1) = P1(:,:,1) - K*(XS(:,:,1)'*P1(:,:,1)); % P(1|1)
S0(:,1) = SI + K*(YS(:,1) - XS(:,:,1)'*SI );     % S(1|1)
% Iterating through the rest of the sample until T
for i = 2:T
    P1(:,:,i) = P0(:,:,i-1) + Q; % P(t|t-1)
    A=chofac(N,AA(:,i+1));
    R = (A\diag(H(i+1,:)))*inv(A)'; % R is shifted one period rel to observables
    K = (P1(:,:,i)*XS(:,:,i))*inv(XS(:,:,i)'*P1(:,:,i)*XS(:,:,i) + R); % K(t)
    P0(:,:,i) = P1(:,:,i) - K*(XS(:,:,i)'*P1(:,:,i));           % P(t|t)
    S0(:,i) = S0(:,i-1) + K*(YS(:,i) - XS(:,:,i)'*S0(:,i-1) ); % S(t|t)
end