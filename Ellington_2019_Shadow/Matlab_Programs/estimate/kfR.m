function [S0,P0,P1] = kfR(Y,X,Q,B,H,SI,PI,T,N,L)
% function [S0,P0,P1] = KFR(Y,X,Q,B,H,SI,PI,T,N,L);
% This file performs the forward kalman filter recursions for the random coefficients VAR.  R is time varying,
% depending on the stochastic volatilities: R = inv(B)*H(t)*B
% C is zero
%
% SI,PI are the initial values for the recursions, S(1|0) and P(1|0)
%
if size(Y,1)>1
    S0 = zeros(N*(1+N*L),T); % current estimate of the state, S(t|t)
    P0 = zeros(N*(1+N*L),N*(1+N*L),T); % current estimate of the covariance matrix, P(t|t)
    P1 = zeros(N*(1+N*L),N*(1+N*L),T); % one-step ahead covariance matrix, P(t|t-1)
    %
    BI = inv(B);
    % date 1
    P1(:,:,1) = PI; % P(1|0)
    R = BI*diag(H(2,:))*BI'; % H is shifted one period rel to observables
    K = (P1(:,:,1)*X(:,:,1))*inv( X(:,:,1)'*P1(:,:,1)*X(:,:,1) + R); % K(1)
    P0(:,:,1) = P1(:,:,1) - K*(X(:,:,1)'*P1(:,:,1)); % P(1|1)
    S0(:,1) = SI + K*( Y(:,1) - X(:,:,1)'*SI ); % S(1|1)
    % Iterating through the rest of the sample
    for i = 2:T
        if size(Q,3)==1
            P1(:,:,i) = P0(:,:,i-1) + Q;
        else
            P1(:,:,i) = P0(:,:,i-1) + Q(:,:,i);
        end
        R = BI*diag(H(i+1,:))*BI';
        K = (P1(:,:,i)*X(:,:,i))*inv( X(:,:,i)'*P1(:,:,i)*X(:,:,i) + R); % K(t)
        P0(:,:,i) = P1(:,:,i) - K*(X(:,:,i)'*P1(:,:,i)); % P(t|t)
        S0(:,i) = S0(:,i-1) + K*( Y(:,i) - X(:,:,i)'*S0(:,i-1) ); % S(t|t)
    end
else
    S0 = zeros(N*(1+N*L),T); % current estimate of the state, S(t|t)
    P0 = zeros(N*(1+N*L),N*(1+N*L),T); % current estimate of the covariance matrix, P(t|t)
    P1 = zeros(N*(1+N*L),N*(1+N*L),T); % one-step ahead covariance matrix, P(t|t-1)
    % date 1
    P1(:,:,1) = PI; % P(1|0)
    R = (H(2,:)); % H is shifted one period rel to observables
    K = (P1(:,:,1)*X(:,:,1))*inv( X(:,:,1)'*P1(:,:,1)*X(:,:,1) + R); % K(1)
    P0(:,:,1) = P1(:,:,1) - K*(X(:,:,1)'*P1(:,:,1)); % P(1|1)
    S0(:,1) = SI + K*( Y(:,1) - X(:,:,1)'*SI ); % S(1|1)
    % Iterating through the rest of the sample
    for i = 2:T
        if size(Q,3)==1
            P1(:,:,i) = P0(:,:,i-1) + Q;
        else
            P1(:,:,i) = P0(:,:,i-1) + Q(:,:,i);
        end
        R = H(i+1,:);
        K = (P1(:,:,i)*X(:,:,i))*inv( X(:,:,i)'*P1(:,:,i)*X(:,:,i) + R); % K(t)
        P0(:,:,i) = P1(:,:,i) - K*(X(:,:,i)'*P1(:,:,i)); % P(t|t)
        S0(:,i) = S0(:,i-1) + K*( Y(:,i) - X(:,:,i)'*S0(:,i-1) ); % S(t|t)
    end
end