function a=TVP_BURN_G2(DFILE1,y,NB,NG,L,TP,SC,Lambda)

% This function estimates a TVC-BVAR along the lines of Primiceri (2005) and of
% Cogley-Morozov-Sargent (2003)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%alpha=0.1;       % One minus the coverage of the confidence intervals
%t0 = clock;
%
[T,N] = size(y); 
% initial estimates through T0 (training sample length: TP)
T0 = 4*TP - L;
% adaptive estimation through T1
T1 = T-L;
% create lagged dependent variables: X1 = [y(t-1) y(t-2) ... y(t-L)]
[X,Y,X1]=lagdep(N,L,T,y);
% partition the data
Y0 = Y(:,1:T0);  
YS = Y(:,1+T0:T1);
X0 = X(:,:,1:T0);
XS = X(:,:,1+T0:T1);
%X01 = X1(1:T0,:);
XS1 = X1(1+T0:T1,:);
%time = time(1+T0:T1,:);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SET PRIORS
% Following Primiceri, we assume that the prior for the parameters of the VAR (the states) is normal with mean
% equal to the OLS estimates of the time-invariant VAR over the first 15 years of data, and variance equal to
% 4 times its estimated variance:
%
% [B,VARB,U,RI]=varp(Y0',L,'Y');
% SI=vec(B'); % Prior for the state
% PI=VARB;    % Prior for its covariance matrix as in Primiceri
% clear B U
%
%%[SI,PI,RI] = surreg(Y0,X0,T0);   % OLS point estimates
%%
[SI,PI,RI]=MINN_BVAR_TRAINING(Y0',X1(1:T0,:),T0,N,L);
%
% An inverse-Wishart prior for the matrix Q:
if (T0<length(SI))==1
    df = length(SI)+1;
elseif Lambda=='CogSa'
    df = length(SI)+1;
else df = T0;
end
%df = N*((N*L)+1)+1; % Prior degrees of freedom (length of coefficient vector+1)
%df = T0;            % length of training sample (always check that it exceeds length of coefficient vector+1)
DF = df + T1 - T0;   % Posterior degrees of freedom
%Q0 = 0.01*PI;       % Prior covariance matrix for state innovations, same as Gambetti (2005)
if Lambda=='Small'
    Q0 = 0.01^2*PI;   % Primiceri (allows for less time variation)
elseif Lambda=='Large'
    Q0 = 0.01*PI;     % allows for most time variation
elseif Lambda=='CogSa'
    Q0 = (3.5e-4)*PI; % Cogley and Sargent
end

TQ0 = df*Q0;         % Prior scaling matrix
TQ = TQ0;            % Initialize posterior scaling matrix
PI = 4*PI;             % Variance covariance matrix of the coefficients
%
% Priors for the stochastic volatility parameters, based on the factorisation RI = inv(A)*H*inv(A)' :
B=chol(RI)';         % Cholesky factor (lower triangular)
invA=zeros(N,N); 
for jj=1:N
    invA(:,jj)=B(:,jj)/B(jj,jj);
end
A=inv(invA);
H=(diag(diag(B))).^2;% standard deviations on the diagonal of B squared to get variances

% A normal prior for the diagonal elements--the elements of H(t). Here we follow Cogley-Morozov-Sargent in assuming
% a very diffuse prior (much more than Primiceri). The specification we use is different from the one in Primiceri,
% in that the hi(t) are assumed to evolve independently (implies sampling from univariate stochastic volatilities).
mu0 = log(diag(H));
ss0 = 10;           %in Primiceri (2005), ss0 = 1

% An inverse gamma prior for the innovations to the log-volatilities (again, we follow Cogley and Sargent):
sv0 = .01;
v0 = 1;                   % a single degree of freedom
d0 = sv0^2;               % scale parameter = 10 to the power of -4
eh = zeros(size(YS,2),N); % volatility innovations
%
% A normal prior for the off-diagonal elements of A(t), as in Primiceri:
muA0 = stackA(A);
ssA0 = diag(abs(muA0))*10;
clear B invA A H

% NOW collect diagonal elements of the standard deviations of Q0
B = chol(Q0)';
HQ = diag(diag(Q0)).^2;
muq0 = log(diag(HQ));
ssq0 = 10;

% IG prior for innovations of log vol of q
svq0 = .01;
vq0 = 10;
dq0 = svq0*svq0;


% An inverse Wishart prior for the covariance matrix of the innovations to
% the the alpha(t): the 2 blocks of S

% The matrix S1:
dfS1 = 2;                      % Prior degrees of freedom (minimum allowed)
DFS1 = dfS1 + T1 - T0;         % Posterior degrees of freedom
S1_0 = abs(muA0(1))*10*0.01^2; % Prior covariance matrix for innovations to the first block of S
TS1_0 = dfS1*S1_0;             % Prior scaling matrix
TS1 = TS1_0;                   % Initialize posterior scaling matrix

% The matrix S2:
dfS2 = 3;                              % Prior degrees of freedom 
DFS2 = dfS2 + T1 - T0;                 % Posterior degrees of freedom
S2_0 = diag(abs(muA0(2:3)))*10*0.01^2; % Prior covariance matrix for innovations to the second block of S
TS2_0 = dfS2*S2_0;                     % Prior scaling matrix
TS2 = TS2_0;                           % Initialize posterior scaling matrix

if N==4
    % The matrix S3:
    dfS3 = 4;                              % Prior degrees of freedom 
    DFS3 = dfS3 + T1 - T0;                 % Posterior degrees of freedom
    S3_0 = diag(abs(muA0(4:6)))*10*0.01^2; % Prior covariance matrix for innovations to the second block of S
    TS3_0 = dfS3*S3_0;                     % Prior scaling matrix
    TS3 = TS3_0;  
    
elseif N==5
    % The matrix S3:
    dfS3 = 4;                              % Prior degrees of freedom 
    DFS3 = dfS3 + T1 - T0;                 % Posterior degrees of freedom
    S3_0 = diag(abs(muA0(4:6)))*10*0.01^2; % Prior covariance matrix for innovations to the second block of S
    TS3_0 = dfS3*S3_0;                     % Prior scaling matrix
    TS3 = TS3_0; 
    % The matrix S4:
    dfS4 = 5;                              % Prior degrees of freedom
    DFS4 = dfS4 + T1 - T0;                 % Posterior degrees of freedom
    S4_0 = diag(abs(muA0(7:10)))*10*0.01^2;% Prior covariance matrix for innovations to the fourth block of S
    TS4_0 = dfS4*S4_0;                     % Prior scaling matrix
    TS4 = TS4_0;  

end
%
% clear initial sample
clear y Y X Y0 X0 X01 T0 T1
[T,N] = size(YS');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Initialize Gibbs arrays:
%
KK = (1+N*L)*N;
% State vector:
SA = zeros(N*(1+N*L),T,NG);         % Draws of the state vector
%QA = zeros(N*(1+N*L),N*(1+N*L),NG); % Draws of covariance matrix for state innovations
QA = eps*ones(T+1,KK,NG);
SVQ = zeros(KK,NG);
%
% Stochastic volatilities: diagonal elements
H = eps*ones(T+1,N,NG); % Stochastic volatilities
SV = zeros(N,NG);       % Standard error for volatility innovation
%
% Stochastic volatilities: off-diagonal elements
AAA = zeros((N*(N-1))/2,T+1,NG); % The off-diagonal elements
SS1 = zeros(1,NG);               % Standard error for volatility innovation
SS2 = zeros(2,2,NG);             % Standard error for volatility innovation
SS3 = zeros(3,3,NG);             % Standard error for volatility innovation
SS4 = zeros(4,4,NG);
%
% Reduced-form VAR residuals:
U=zeros(T,N,NG);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% set 'seed' for random number generator 
seednumber=1111;
rand('seed',seednumber);
randn('seed',seednumber);
%
% initial for log(h(t))
lh(1:2,:) = ones(2,1)*mu0';            %log h
YS1=YS(1:N,:)';
dy = diff(YS1,1);
e(:,1:N) = dy - ones(T-1,1)*mean(dy);  %are considered like "innovations"
lh(3:T+1,:) = log(e.^2);
H0 = exp(lh);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% warm up for stochastic volatilities; shut down var, treat dy as var innovation
%
evar = [zeros(N,1) e'];
H(:,:,1) = H0;
ch0 = zeros(N*(N-1)/2,1);
ch = ch0;
ssc0 = 10000*eye(N*(N-1)/2,N*(N-1)/2);   %here we follow Cogley and Sargent (2005), that's why it is 10000
                                         %we assume: A(t)=B i.e. NOT time-varying during warming-up
CF = chofac(N,ch0); 
yhs = zeros(size(YS,2),N);
CH = zeros(size(ch0,1),NG);


for iter = 2:NG
    % R conditional on states and data (svol programs)
    lh = log(H(:,:,iter-1));
    for i = 1:N,
       eh(:,i) = lh(2:T+1,i) - lh(1:T,i);  % random walk
       v = ig2(v0,d0,eh(:,i));
       SV(i,iter) = v^.5;
    end
     
    % orthogonalize var innovations
    CF = chofac(N,ch);  %creates a lower triangular matrix with the elements of ch below diagonal
    f = (CF*evar)';     %orthogonalized "fake" VAR reduced-form innovations with Var(f)=H
    
    % lh | ch,sv,b,y
     for i = 1:N    %indicates that we are proceeding on UNIVARIATE basis (each equation separately because stochastic volatilities h are independent)
       %beginning of the sample: no observed value of f and no H(t-1) 
       H(1,i,iter) = svmh0(H(2,i,iter-1),0,1,SV(i,iter),mu0(i,1),ss0);
       %sample: take H(t+1) from previous iteration and H(t-1) from
       %previous step
       for t = 2:T 
          H(t,i,iter) = svmh(H(t+1,i,iter-1),H(t-1,i,iter),0,1,SV(i,iter),f(t-1,i),H(t,i,iter-1)); 
       end
       %end of sample: only H(t-1) available
       H(T+1,i,iter) = svmhT(H(T,i,iter),0,1,SV(i,iter),f(T,i),H(T+1,i,iter-1)); 
     end
     
     % ch | sv,b,lh,y
     k = 0;
     for i = 2:N, 
        lhs = H(2:T+1,i,iter).^.5;  
        for n = 1:N,
            yhs(:,n) = (evar(n,:)')./lhs;
        end 
        yr = yhs(:,i);
        xr = -yhs(:,1:i-1);
        j = k+1;
        k = i-1+k;
        ch(j:k,1) = bayesreg(ch0(j:k),ssc0(j:k,j:k),1,yr,xr);
     end
     CH(:,iter) = ch;  
end
%
%
% Initial draw of states (here we follow Cogley and Sargent):
H0 = H(:,:,NG);
Q00 = diag(diag(Q0));
[S0,P0,P1] = kfR(YS,XS,Q00,CF,H0,SI,PI,T,N,L); 
mlmax = 2;
S = [eye(N*(L-1)),zeros(N*(L-1),N)];
lmax = zeros(size(YS,2),1);
while mlmax >= 1,
     SA1 = gibbs1(S0,P0,P1,T,N,L);
     for j = 1:T,
       A = SA1(2:1+N*L,j)';
       for k=1:N-1,
	      A = [A;SA1(2+k*(1+N*L):(k+1)*(1+N*L),j)'];
       end
       A = [A;S];
       lmax(j,1) = max(abs(eig(A)))';
     end 
     mlmax = max(lmax)
end

% initial values for log(q_t)
lq = ones(2,1)*muq0';
SA11 = SA1'; SA11 = SA11(2:T,:)-SA11(1:T-1,:); eq(:,1:KK)=SA11-ones(T-1,1)*mean(SA11);
lq(3:T+1,:) = log(eq.^2);
HQ0 = exp(lq);
evarq = [zeros(KK,1) eq'];
QA(:,:,1) = HQ0;
eq = zeros(T,KK);

bh0 = zeros(KK*(KK-1)/2,1);
bh = bh0;
ssb0 = 10000*eye(N*(N-1)/2,N*(N-1)/2);   %here we follow Cogley and Sargent (2005), that's why it is 10000
                                         %we assume: A(t)=B i.e. NOT time-varying during warming-up
BF = chofac(KK,bh0); 
yhs = zeros(size(YS,2),KK);
BH = zeros(size(ch0,1),NG);

for iter=2:NG
    lq = log(QA(:,:,iter-1));
    for i = 1:KK
        eq(:,i) = lq(2:T+1,i)-lq(1:T,i); % RW
        vq = ig2(vq0,dq0,eq(:,i));
        SVQ(i,iter) = vq^.5;
    end
    EQ(:,:,iter) = eq;
    fq = (BF*evarq)';
        % lh | ch,sv,b,y
     for i = 1:KK    %indicates that we are proceeding on UNIVARIATE basis (each equation separately because stochastic volatilities h are independent)
       %beginning of the sample: no observed value of f and no H(t-1) 
       QA(1,i,iter) = svmh0(QA(2,i,iter-1),0,1,SVQ(i,iter),muq0(i,1),ssq0);
       %sample: take H(t+1) from previous iteration and H(t-1) from
       %previous step
       for t = 2:T 
          QA(t,i,iter) = svmh(QA(t+1,i,iter-1),QA(t-1,i,iter),0,1,SVQ(i,iter),fq(t-1,i),QA(t,i,iter-1)); 
       end
       %end of sample: only H(t-1) available
       QA(T+1,i,iter) = svmhT(QA(T,i,iter),0,1,SVQ(i,iter),fq(T,i),QA(T+1,i,iter-1)); 
     end
          % ch | sv,b,lh,y
     k = 0;
     for i = 2:N, 
        lhs = QA(2:T+1,i,iter).^.5;  
        for n = 1:N,
            yhs(:,n) = (evarq(n,:)')./lhs;
        end 
        yr = yhs(:,i);
        xr = -yhs(:,1:i-1);
        j = k+1;
        k = i-1+k;
        bh(j:k,1) = bayesreg(bh0(j:k),ssb0(j:k,j:k),1,yr,xr);
     end
     BH(:,iter) = ch;  
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initial inputs
SA(:,:,2) = SA1;
H(:,:,1:2) = H(:,:,NG-1:NG);
SV(:,1:2) = SV(:,NG-1:NG);
    for hh=1:T
        SA(:,hh,1)=SI;
    end
clear CH ch ch0 ssc0 CF A
QA(:,:,1:2) = QA(:,:,NG-1:NG);
SVQ(:,1:2) = SVQ(:,NG-1:NG);
%
U(:,:,1)=f;   % only needed if you have to step back --> buffer!
%ET(:,:,1)=fq(2:end,:);
ET(:,:,1)=fq;
clear f fq
%
% hybrid MCMC 'Metropolis-within-Gibbs' sampler for the unrestricted Bayesian VAR: 
% replaces some of the Gibbs steps with Metropolis accept/reject steps
% (involves conditional kernel instead of conditional density)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BURN-IN
% Begin MCMC burn-in: 25 files a 2000 draws for a total burn-in period of
%                     50000 iterations
%
% Variables to be saved:
varname(1,:) = 'SA';
varname(2,:) = 'QA';
varname(3,:) = 'SV';
varname(4,:) = 'UU';
varname(5,:) = 'HH';
varname(6,:) = 'S1';
varname(7,:) = 'S2';
varname(8,:) = 'VQ';
varname(9,:) = 'AA';
varname(10,:) = 'FF';
varname(11,:) = 'II';
varname(12,:) = 'ET';
varname(13,:) = 'S3';

%%
file=1;
%
while file <= NB/NG        
    maxshakes = 200; % maximum number of attempts at stable draw (if stability constraint is imposed)
    iter = 3;        % provide buffer for back steps in subsequent files
    gc = 1;
    while iter <= NG   %previously: NG (=5000)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (1) Drawing states: Kalman filter (forward filtering, backward sampling)
        [S0,P0,P1] = kfP1(YS,XS,QA(:,:,iter-1),AAA(:,:,iter-1),H(:,:,iter-1),SI,PI,T,N,L); 
        SA(:,:,iter) = gibbs1(S0,P0,P1,T,N,L);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (2) Draw Q conditional on states: innovation variance
        % The posterior estimate of the covariance matrix for the innovations to the random walk parameters of
        % the the VAR, the v(t)'s:
        %[TQ,DF] = iwpQ(SA(:,:,iter),T,TQ0,df);
        % Drawing Q:
        %QA(:,:,iter) = gibbs2Q(TQ,DF,N,L);
        lq = log(QA(:,:,iter-1)); 
        for i = 1:KK
            % sv|b,lh,ch,y
            % Conditional on a specific time path of log[h(t)], the innovations to the logs of the stochastic
            % volatilities are directly observable. The innovations are mutually independent, so we consider
            % them one at a time:
            eq(:,i)=diff(lq(:,i));  % The innovations
            v = ig2(v0,d0,eq(:,i)); % The draw for the volatility from an inverse gamma distribution
            SVQ(i,iter) = v^.5;      % The standard deviation of volatility innovations
        end
        % conditional on \Theta^{T} eta_{t} are observable:
        % \eta_{t} = \Theta_{t} - Theta_{t-1}
        % Unsure how to get these, do we want recursive residuals?
        %ET(:,:,iter) = innovmQ2(SA(:,:,iter),SA(:,:,iter)',KK,T)';
        % RECURSIVE RESIDUALS ARE NOT THE CORRECT METHOD TO COMPUTE THESE
        % RESULTS
        ET(2:T,:,iter) = diff(SA(:,:,iter)');
        % LINE 388: in computing eta_{t} we lose an observation. HOWEVER
        % reduced form results in computing this way look extremely similar
        % to that when assuming Q_{t} = Q (possibly not diagonal).
        fq = (eye(KK)*ET(:,:,iter)')';
        
     for i = 1:KK    %indicates that we are proceeding on UNIVARIATE basis (each equation separately because stochastic volatilities h are independent)
       %beginning of the sample: no observed value of f and no H(t-1) 
       QA(1,i,iter) = svmh0(QA(2,i,iter-1),0,1,SVQ(i,iter),muq0(i,1),ssq0);
       %sample: take H(t+1) from previous iteration and H(t-1) from
       %previous step for t = 2:T
       for t = 2:T
          QA(t,i,iter) = svmh(QA(t+1,i,iter-1),QA(t-1,i,iter),0,1,SVQ(i,iter),fq(t-1,i),QA(t,i,iter-1)); 
       end
       %end of sample: only H(t-1) available
       QA(T+1,i,iter) = svmhT(QA(T,i,iter),0,1,SVQ(i,iter),fq(T,i),QA(T+1,i,iter-1)); 
     end
      % T+1
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (3) The residuals from the time-varying VAR. Conditional on data and states, they are observable:
        U(:,:,iter)=innovm(YS,XS1,SA(:,:,iter),N,T,L)'; 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (4) Drawing the matrices S1 and S2 conditional on the previous iteration:
        % S1:
        [TS1,DFS1] = iwpQ(AAA(1,:,iter-1),T,TS1_0,dfS1);
        PS=real(sqrt(inv(TS1)));
        u=randn(1,DFS1);
        SS1(1,iter)=1/(PS*(u*u')*PS');
        % S2:
        [TS2,DFS2] = iwpQ(AAA(2:3,:,iter-1),T,TS2_0,dfS2);
        PS=real(sqrtm(inv(TS2))); 
        u=randn(2,DFS2);
        SS2(:,:,iter) = inv(PS*(u*u')*PS');
        clear PS u
        
        if N==4
            % S3:
            [TS3,DFS3] = iwpQ(AAA(4:6,:,iter-1),T,TS3_0,dfS3);
            PS=real(sqrtm(inv(TS3))); 
            u=randn(3,DFS3);
            SS3(:,:,iter) = inv(PS*(u*u')*PS');
            clear PS u
        elseif N==5
            % S3:
            [TS3,DFS3] = iwpQ(AAA(4:6,:,iter-1),T,TS3_0,dfS3);
            PS=real(sqrtm(inv(TS3))); 
            u=randn(3,DFS3);
            SS3(:,:,iter) = inv(PS*(u*u')*PS');
            clear PS u
            % S4:
            [TS4,DFS4] = iwpQ(AAA(7:10,:,iter-1),T,TS4_0,dfS4);
            PS=real(sqrtm(inv(TS4))); 
            u=randn(4,DFS4);
            SS4(:,:,iter) = inv(PS*(u*u')*PS');
            clear PS u
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (5) Drawing the off-diagonal elements of A(t) conditional on the previous iteration:
        if N==3
            AAA(:,:,iter)=getalpha(U(:,:,iter),SS1(1,iter),SS2(:,:,iter),muA0,ssA0,T,H(:,:,iter-1));
        elseif N==4
            AAA(:,:,iter)=getalphafour(U(:,:,iter),SS1(1,iter),SS2(:,:,iter),SS3(:,:,iter),muA0,ssA0,T,H(:,:,iter-1),N);
        elseif N==5
            AAA(:,:,iter)=getalphafive(U(:,:,iter),SS1(1,iter),SS2(:,:,iter),SS3(:,:,iter),SS4(:,:,iter),muA0,ssA0,T,H(:,:,iter-1),N);
        end       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (6) Orthogonalize the VAR innovations: A orthogonalizes R,
        % this is needed as input for the stochastic volatilities H
        for t=1:T
            CF=chofac(N,AAA(:,t,iter));   % CF is lower triangular
            f(t,:)=(CF*U(t,:,iter)')';    % epsilon=A*VAR innovations U
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (7) Drawing the diagonal of H(t) conditional on the previous iteration:
        % The logarithms of the diagonal elements of H(t) from the previous iteration:
        lh = log(H(:,:,iter-1)); 
        for i = 1:N
            % sv|b,lh,ch,y
            % Conditional on a specific time path of log[h(t)], the innovations to the logs of the stochastic
            % volatilities are directly observable. The innovations are mutually independent, so we consider
            % them one at a time:
            eh(:,i)=diff(lh(:,i));  % The innovations
            v = ig2(v0,d0,eh(:,i)); % The draw for the volatility from an inverse gamma distribution
            SV(i,iter) = v^.5;      % The standard deviation of volatility innovations
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Drawing the volatilities: lh|ch,sv,b1,y (Metropolis step)
        % We do it on a univariate basis (according to the algorithm of Jacquier et al (1994)),
        % as the stochastic volatilities are independent of one another:
        for i = 1:N
            % Drawing the first, based on the distribution:
            H(1,i,iter) = svmh0(H(2,i,iter-1),0,1,SV(i,iter),mu0(i,1),ss0); 
            for t = 2:T
                % Drawing from 2 to T, based on previous iteration:
                H(t,i,iter) = svmh(H(t+1,i,iter-1),H(t-1,i,iter),0,1,SV(i,iter),f(t-1,i),H(t,i,iter-1)); 
            end
            % Drawing the last:
            H(T+1,i,iter) = svmhT(H(T,i,iter),0,1,SV(i,iter),f(T,i),H(T+1,i,iter-1)); 
        end
        
        clear f CF
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Checking stability; reject unstable draws
        if SC==0
           iter = iter+1;
            [file iter]
            gc = 1;
            if (iter/100)==fix(iter/100)
                HH=H;
                UU=U;
                AA=AAA;
                S1=SS1;
                S2=SS2;
                S3=SS3;
                S4=SS4;
                FF=file;
                II=iter;
                VQ = SVQ;
                save(DFILE1,varname(1,:),varname(2,:),varname(3,:),varname(4,:),varname(5,:),varname(6,:),varname(7,:),varname(8,:),varname(9,:),varname(10,:),varname(11,:),varname(12,:),varname(13,:));
                clear HH UU AA S1 S2 S3 S4 FF II VQ
            end 
            
        else
        for j = 1:T
            A=SA(2:1+N*L,j,iter)';
            for k=1:N-1
               A=[A;SA(2+k*(1+N*L):(k+1)*(1+N*L),j,iter)'];
            end
            A=[A;S];
            lmax(j,1) = max(abs(eig(A)))';
        end
        %
        if max(lmax) <= 1
            %[file,iter];
            for i = 2:T
            max(eig(diag(QA(i,:,iter))));
            end
            %trace(QA(:,:,iter));
            iter = iter+1;
            [file iter]
            gc = 1;
            if (iter/100)==fix(iter/100)
                HH=H;
                UU=U;
                AA=AAA;
                S1=SS1;
                S2=SS2;
                S3=SS3;
                FF=file;
                II=iter;
                VQ = SVQ;
                save(DFILE1,varname(1,:),varname(2,:),varname(3,:),varname(4,:),varname(5,:),varname(6,:),varname(7,:),varname(8,:),varname(9,:),varname(10,:),varname(11,:),varname(12,:),varname(13,:));
                clear HH UU AA S1 S2 S3 S4 FF II VQ
            end
        elseif (max(lmax) > 1)&&(gc < maxshakes),
            %disp('States unstable, try again')
            gc = gc + 1;
        elseif (max(lmax) > 1)&&(gc == maxshakes),
            %disp('G1 Fails Repeatedly, Step Back')
            iter = iter-1;
            gc = 1;
        end
        
        end
    end
    % Reinitialize gibbs arrays (buffer for back step)
    SA(:,:,1:2) = SA(:,:,NG-1:NG);
    QA(:,:,1:2) = QA(:,:,NG-1:NG); 
    SV(:,1:2) = SV(:,NG-1:NG);
    SVQ(:,1:2) = SVQ(:,NG-1:NG);
    H(:,:,1:2) = H(:,:,NG-1:NG);
    U(:,:,1:2) = U(:,:,NG-1:NG);
    ET(:,:,1:2) = ET(:,:,NG-1:NG);
    AAA(:,:,1:2) = AAA(:,:,NG-1:NG);
    SS1(:,1:2) = SS1(:,NG-1:NG);
    SS2(:,:,1:2) = SS2(:,:,NG-1:NG);
    file=file+1;
end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%clear DFILE1 varname
a=1;
%
