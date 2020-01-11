function [betas,vols,b0] = QBLL_univariate(RET,FACTORS)
% Follows Petrova thesis using Normal-Gamma prior and posterior

% Michael Ellington 07/01/2020
nsim=500; T=size(RET,1); N=size(RET,2);

y=RET;
X=FACTORS; k=size(FACTORS,2);
X=[ones(T,1), X];
% USE full sample OLS estimates as priors...

b0=X\y;
e=y-X*b0;
%b0=zeros(k+1,N);
% because we are working with linear models with 1 dependent variable and
% each model has a different kappa0 positive definite matrix, we need to
% define Bcov individually. 
s2=e'*e/(T-k); % diagonal elements of this are the scale parameter of Gamma distribution.
% let's call them gamma0;

gamma0=1./diag(s2);
% now we need to define the kapp0 matrices for each of the N assets.
kappa0=zeros(k+1,k+1,N);
for kk=1:N
kappa0(:,:,kk)=s2(kk,kk)*inv(X'*X);
end
a0=max(k+2,k+2*8-T);
gamma0=(a0-k-1)*gamma0;
% calculate weights
weights=normker(T,sqrt(T));

% define storage for betas and lambdas

betas=zeros(1+k,N,3,T);
vols=zeros(T,N,3);
qq=[0.025 0.5 0.975];

parfor kk=1:N
    kk
    k0=kappa0(:,:,kk);
    g0=gamma0(kk,1);
    b01=b0(:,kk);
    yy=y(:,kk);
    BET=zeros(1+k,nsim,T);
    VOL=zeros(T,nsim);
   % now parfor loop to get betas
   for ii=1:T
       B1=zeros(k+1,nsim);
       V1=zeros(1,nsim);
       w=weights(ii,:);
       bayesprec=(k0+X'*diag(w)*X);
       bayessv=bayesprec^(-1);
       BB=bayessv*((X'*diag(w))*yy+k0*b01);
       bayesalpha=a0+sum(w)/2;
       g1=b01'*k0*b01; g2=yy'*diag(w)*yy; g3=BB'*bayesprec*BB;
       bayesgamma=g0+0.5*(g1+g2-g3)^3;
       for ll=1:nsim
           gam=1/gamrnd(bayesalpha,bayesgamma);
           V1(1,ll)=gam;
           nu=randn(k+1,1);
           B=(BB+chol(bayessv)'*nu*sqrt(gam))';
           B1(:,ll)=B;
       end
       BET(:,:,ii)=B1;
       VOL(ii,:)=V1;

   end
   betas(:,kk,:,:)=quantile(BET,qq,2);
   vols(:,kk,:)=quantile(VOL,qq,2);
end 




end
