function [SI, PI, a, RI] = Minn_NWprior(Y,T,N,L,shrinkage)
% These are priors adapted from Petrova 2018. For RV data, I create a
% persistent prior mean to deal with near unit root behaviour. This can
% obviously change depending on application. For example, using daily
% return data will result in a prior mean of zero. 

% Michael Ellington

K=N*L+1; % dimension for Kronecker structure

SI=[zeros(1,N); 0.5*eye(N); zeros((L-1)*N,N)]; % Prior mean parameter matrix
% SI=SI(:); % parameter vector uncomment if required
% SI=zeros(K,N);
PI=zeros(K,1); % kronecker structure
sigma_sq=zeros(N,1); % matrix to store residual variance

for i=1:N
    % Create lags of dependent variable
    Y_i=mlag2(Y(:,i),L);
    Y_i=Y_i(L+1:T,:);
    X_i=[ones(T-L,1) Y_i];
    y_i=Y(L+1:T,i);
    % OLS estimates of i-th equation
    alpha_i=(X_i'*X_i)\(X_i'*y_i);
    sigma_sq(i,1)=(1./(T-L+1))*(y_i-X_i*alpha_i)'*(y_i-X_i*alpha_i);
end

s=sigma_sq.^(-1);
for ii=1:L
   PI(2+N*(ii-1):1+N*ii)=(shrinkage^2)*s/(ii^2); 
end
 PI(1)=10^2; % prior variance for constant is loose
 PI=diag(PI);
 
 % now for Wishart priors following Petrova (2018)
 a=max(N+2,N+2*8-T);
 RI=(a-N-1)*sigma_sq;
 RI=diag(RI);
end