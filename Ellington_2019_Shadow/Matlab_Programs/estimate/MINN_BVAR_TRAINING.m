function [SI, PI, RI] = MINN_BVAR_TRAINING(Y,X,T,N,L)
%
% LOAD IN Y0' in my code
% DEFINE X0 to estimate this model in 2d form from X1. Therefore X0_1 =
% X1(1:T0,:); 
%Y = Y0'; X = X1(1:T0,:);
M = N;
p = L;
K = N*L+1;
constant=1;
%T = T0;
% Estimates a Bayesian VAR model on the training sample using a Minnesota
% prior to introduce shrinkage on the theta_{0} we use the posterior mean
% of the associated simulations to estimate the TVP VAR model. 

    A_prior = [zeros(1,M); 0.9*eye(M); zeros((p-1)*M,M)];  %<---- prior mean of ALPHA (parameter matrix) 
    a_prior = A_prior(:);               %<---- prior mean of alpha (parameter vector)
    
    % Minnesota Variance on VAR regression coefficients
    % First define the hyperparameters 'a_bar_i'
    a_bar_1 = 0.5;
    a_bar_2 = 0.5;
    a_bar_3 = 10^2;
    
    % Now get residual variances of univariate p_MIN-lag autoregressions. Here
    % we just run the AR(p) model on each equation, ignoring the constant
    % and exogenous variables (if they have been specified for the original
    % VAR model)
    p_MIN = 6;
    sigma_sq = zeros(M,1); % vector to store residual variances
    for i = 1:M
        % Create lags of dependent variables   
        Ylag_i = mlag2(Y(:,i),p);
        Ylag_i = Ylag_i(p_MIN+1:T,:);
        X_i = [ones(T-p_MIN,1) Ylag_i];
        Y_i = Y(p_MIN+1:T,i);
        % OLS estimates of i-th equation
        alpha_i = inv(X_i'*X_i)*(X_i'*Y_i);
        sigma_sq(i,1) = (1./(T-p_MIN))*(Y_i - X_i*alpha_i)'*(Y_i - X_i*alpha_i);
    end
    % Now define prior hyperparameters.
    % Create an array of dimensions K x M, which will contain the K diagonal
    % elements of the covariance matrix, in each of the M equations.
    V_i = zeros(K,M);
    
    
    % index in each equation which are the own lags
    ind = zeros(M,p);
    for i=1:M
        ind(i,:) = constant+i:M:K;
    end
    for i = 1:M  % for each i-th equation
        for j = 1:K   % for each j-th RHS variable
            if constant==1
                if j==1 % if there is constant, use this code
                    V_i(j,i) = a_bar_3*sigma_sq(i,1); % variance on constant                
                elseif find(j==ind(i,:))>0
                    V_i(j,i) = a_bar_1./(ceil((j-1)/M)^2); % variance on own lags           
                    % Note: the "ceil((j-1)/M)" command finds the associated lag 
                    % number for each parameter
                else
                    for kj=1:M
                        if find(j==ind(kj,:))>0
                            ll = kj;                   
                        end
                    end                 % variance on other lags  
                    V_i(j,i) = (a_bar_2*sigma_sq(i,1))./((ceil((j-1)/M)^2)*sigma_sq(ll,1));           
                end
            else   % if no constant is defined, then use this code
                if find(j==ind(i,:))>0
                    V_i(j,i) = a_bar_1./(ceil(j/M)^2); % variance on own lags
                else
                    for kj=1:M
                        if find(j==ind(kj,:))>0
                            ll = kj;
                        end                        
                    end                 % variance on other lags  
                    V_i(j,i) = (a_bar_2*sigma_sq(i,1))./((ceil(j/M)^2)*sigma_sq(ll,1));            
                end
            end
        end
    end
    % Now V is a diagonal matrix with diagonal elements the V_i
    V_prior = diag(V_i(:));  % this is the prior variance of the vector alpha
    
    %NOTE: No prior for SIGMA. SIGMA is simply a diagonal matrix with each
    %diagonal element equal to sigma_sq(i). See Kadiyala and Karlsson (1997)
%    SIGMA = diag(sigma_sq);
SIGMA = eye(M);
S = eye(M);
%    RI = SIGMA;
% set storage matrices
reps = 10000;
v_post = zeros(M*K,M*K,reps);
a_po = zeros(M*K,reps);
s_po = zeros(M,M,reps);
% sig is known and diagonal
for jj = 1:reps
    % draw theta
        for i = 1:M
            V_post = inv( inv(V_prior((i-1)*K+1:i*K,(i-1)*K+1:i*K)) + inv(SIGMA(i,i))*X'*X );
            a_post = V_post*(inv(V_prior((i-1)*K+1:i*K,(i-1)*K+1:i*K))*a_prior((i-1)*K+1:i*K,1) + inv(SIGMA(i,i))*X'*Y(:,i));
            alpha((i-1)*K+1:i*K,1) = a_post + chol(V_post)'*randn(K,1); % Draw alpha
        end
    ALPHA = reshape(alpha,K,M); % Create draw in terms of ALPHA
    % draw sigma
    e = Y-X*ALPHA;
    scale = e'*e+S;
    SIGMA = iwpq_2(T+M+1,inv(scale));
    ALPHA= ALPHA(:);
    V_po = inv( inv(V_prior) + kron(inv(SIGMA),X'*X));
    v_post(:,:,jj) = V_po;
    a_po(:,jj) = ALPHA;
    s_po(:,:,jj) = SIGMA;
end

PI = mean(v_post,3);
SI = mean(a_po,2);
RI = mean(s_po,3);