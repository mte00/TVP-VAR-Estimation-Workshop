% ESTIMATE TVP VAR with time-varying covariance matrix using Quasi-Bayesian
% Local Likelihood estimation method following Petrova (2019) Journal of
% Econometrics and compute time-frequency connectedness measures.

% Uses parallel computing to speed up computation
% Especially in using large model such as N>100.

% Full estimation Procedure carried out in 1 step so we ignore estimated
% covariance matrices and time-varying parameters. Only interested in
% connectedness measures.

% Michael Ellington
% First WriTen: 24/02/2019
% Last Updated: 08/01/2020
clc; clear all; close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load in Data                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning('off','all');

seed=07012020;
rng(seed);
addpath('src')

% load data
data=xlsread('Example_5_data','Sheet1','B2:I1002');

data=log(sqrt(1+data));

% generate figure
figure(1)
plot(data(:,:),'LineWidth',1.5)
axis tight
ylabel('%')
legend('AAPL','AIG','AMZN','GOOGL','NFLX','NKE','UPS','Location','SouthOutside')
%matlab2tikz('EX5_1.tex')

[T,N]=size(data);
nsim=500;
shrinkage=0.05; % Overall shrinkage of Minnesota Prior. Change this to investigate influence
L=2; % Daily data corresponds to 2 day lag (5 lags is one week).
K=N*L+1;
% Get data in VAR setup
dat2=data;
for i=1:L
    temp=lag0(dat2,i);
    X(:,1+N*(i-1):i*N)=temp(1+L:T,:);
end
y=dat2(1+L:T,:); T=T-L; X=[ones(T,1),X]; % y[T x N], X[T x N*L+1]
clear temp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prior Specification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[SI,PI,a,RI]=Minn_NWprior(dat2,T,N,L,shrinkage);
% SI prior mean matrix. PI prior variance of SI (diagonal matrix (K x K)),
% and RI is prior for covariance matrix of VAR model.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate weights
weights=normker(T,sqrt(T));
% Follow Petrova (2018) work with precision
priorprec0=PI^(-1);
clear PI data dat2 ind
priorprec0=sparse(priorprec0); % create sparse matrix allowing more efficient allocation of memory.
RI=sparse(RI); % create sparse matrix allowing more efficient allocation of memory.
% B=(X'*diag(w)*X)^(-1)*(X'*diag(w)*y);
% bhat=B(:);
% bhat=((kron(eye(N),X'*diag(w)*X))^(-1))*(kron(eye(N),X'*diag(w))*y(:));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Posteriors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stab_ind=zeros(nsim,T,'single');
max_eig=zeros(nsim,T,'single');
% Generate storage matrices
HO=100+1;
TIC=zeros(T,nsim);
TTC=zeros(T,N,nsim); % Shocks transimmiTed to ith variable
TRC=zeros(T,N,nsim); % Shocks received from ith variable
NDC=zeros(T,N,nsim); % Net-directional Connectedness for each variable
PWC=zeros(N,N,nsim,T);

varname(1,:)='TIC'; varname(2,:)='TTC'; varname(3,:)='TRC';
varname(4,:)='NDC'; varname(5,:)='PWC';
DFILE='Connect_QBLL_7_stocks';
tic;
parfor kk=1:T
   kk/T % This reports the proportion of computationsim carried out.
   % when using parallel computing numbers will not be in order.
   % Storage for connectedness measures
   tic=zeros(1,nsim); ttc=zeros(N,nsim); trc=zeros(N,nsim); ndc=zeros(N,nsim); pw=zeros(N,N,nsim);
   % Estimation and connectedness computing
   w=weights(kk,:);
   bayesprec=(priorprec0+X'*diag(w)*X);
   bayessv=bayesprec^(-1);
   BB=bayessv*((X'*diag(w))*y+priorprec0*SI);
   bayesb=BB(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% equivalent but more computationally intensimive formula: 
% bayesb1=(kron(eye(N),bayesv^(-1)))*(kron(eye(N),X'*diag(w)*X)*bhat+kron(eye(N),priorprec0)*priormean(:));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  bayesalpha=a+sum(w);
  g1=SI'*priorprec0*SI;
  g2=y'*diag(w)*y;
  g3=BB'*bayesprec*BB;
  bayesgamma=RI+g1+g2-g3;
  bayesgamma=0.5*bayesgamma+0.5*bayesgamma'; %it is symmetric but just in case
  %make draws
  for ii=1:nsim
     mm=0;
     while mm<1
         SIGMA=iwishrnd(bayesgamma,bayesalpha); % Draw from IW distribution
         % Fi=mvnrnd(bayesb,kron(Sigma,bayesv))';
         nu=randn(N*L+1,N);
         Fi1=(BB+chol(bayessv)'*nu*(chol(SIGMA)))';         
         max_eig(ii,kk)=max(abs(eig([Fi1(:,2:end); eye(N), zeros(N,N)])));
         if max_eig(ii,kk)<.999 % check stability of draw
             stab_ind(ii,kk)=1;
             mm=1;
         end
     end
     [irf,wold]=get_GIRF(Fi1,SIGMA,1,L,HO-1);
     [tc, tr,tt,nd,pwc]=get_timeconnect(N,HO,irf); % time connectedness. Comment out if not required.
     tic(:,ii)=tc; ttc(:,ii)=tt; trc(:,ii)=tr; ndc(:,ii)=nd; pw(:,:,ii)=pwc;
  end
    TIC(kk,:)=tic;
    TTC(kk,:,:)=ttc;
    TRC(kk,:,:)=trc;
    NDC(kk,:,:)=ndc;
    PWC(:,:,:,kk)=pw;
end
toc
save(DFILE,varname(1,:),varname(2,:),varname(3,:),varname(4,:),varname(5,:));
%%

figure(1)
plot(median(TIC,2),'b-','LineWidth',1.5)
hold on,
plot(quantile(TIC,0.025,2),'b--')
plot(quantile(TIC,0.975,2),'b--')
title('Total Connectedness')
axis tight
legend off
%matlab2tikz('EX5_2.tex')

NNDC=quantile(NDC,[0.025, 0.5, 0.975],3);

figure(2)
subplot(1,7,1)
plot(NNDC(:,1,2),'k-','LineWidth',1.5)
hold on,
plot(NNDC(:,1,1),'k--')
plot(NNDC(:,1,3),'k--')
title('AAPL')
axis tight
subplot(1,7,2)
plot(NNDC(:,2,2),'k-','LineWidth',1.5)
hold on,
plot(NNDC(:,2,1),'k--')
plot(NNDC(:,2,3),'k--')
title('AIG')
axis tight
subplot(1,7,3)
plot(NNDC(:,3,2),'k-','LineWidth',1.5)
hold on,
plot(NNDC(:,3,1),'k--')
plot(NNDC(:,3,3),'k--')
title('AMZN')
axis tight
subplot(1,7,4)
plot(NNDC(:,4,2),'k-','LineWidth',1.5)
hold on,
plot(NNDC(:,4,1),'k--')
plot(NNDC(:,4,3),'k--')
title('GOOGL')
axis tight
subplot(1,7,5)
plot(NNDC(:,5,2),'k-','LineWidth',1.5)
hold on,
plot(NNDC(:,5,1),'k--')
plot(NNDC(:,5,3),'k--')
title('NFLX')
axis tight
subplot(1,7,6)
plot(NNDC(:,6,2),'k-','LineWidth',1.5)
hold on,
plot(NNDC(:,6,1),'k--')
plot(NNDC(:,6,3),'k--')
title('NKE')
axis tight
subplot(1,7,7)
plot(NNDC(:,7,2),'k-','LineWidth',1.5)
hold on,
plot(NNDC(:,7,1),'k--')
plot(NNDC(:,7,3),'k--')
title('UPS')
axis tight
%matlab2tikz('EX5_3.tex')






