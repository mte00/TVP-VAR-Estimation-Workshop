function N=TVP_MCMC_Minn(DFILE1,DFILE2,y,D,NG,L,TP,SC,Lambda)

% This function estimates a TVC-BVAR along the lines of Primiceri (2004) and of
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
if N==6
    [X,Y,X1] = lagdep6(N,L,T,y);
else
[X,Y,X1]=lagdep(N,L,T,y);
end
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
% [SI,PI,RI] = surreg(Y0,X0,T0);   % OLS point estimates
%%
[SI,PI,RI]=MINN_BVAR_TRAINING(Y0',X1(1:T0,:),T0,N,L);
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
    Q0 = (0.01^2)*PI; % Primiceri (allows for less time variation)
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
%
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
elseif N==6
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
    % the matrix S5
    dfS5 = 6;
    DFS5 = dfS5 + T1 - T0;
    S5_0 = diag(abs(muA0(11:15)))*10*0.01^2;
    TS5_0 = dfS5*S5_0;
    TS5 = TS5_0;
end
%
% clear initial sample
clear y Y X Y0 X0 X01 T0 T1
[T,N] = size(YS');

S = [eye(N*(L-1)),zeros(N*(L-1),N)];

load(DFILE1)

SA(:,:,1:4) = SA(:,:,NG-3:NG);
QA(:,:,1:4) = QA(:,:,NG-3:NG); 
SV(:,1:4) = SV(:,NG-3:NG);
H(:,:,1:4) = HH(:,:,NG-3:NG);
U(:,:,1:4) = UU(:,:,NG-3:NG);
AAA(:,:,1:4) = AA(:,:,NG-3:NG);
SS1(:,1:4) = S1(:,NG-3:NG);
SS2(:,:,1:4) = S2(:,:,NG-3:NG);
SS3(:,:,1:4) = S3(:,:,NG-3:NG);
SS4(:,:,1:4) = S4(:,:,NG-3:NG);
SS5(:,:,1:4) = S5(:,:,NG-3:NG);

% Variables to be saved:
varname(1,:) = 'SD'; % States (time-varying parameters in the VAR)
varname(2,:) = 'QD'; % State innovation covariance matrix (the matrix Q)
varname(3,:) = 'AA'; % Non-zero and non-one elements in lower diagonal block of matrix A(t) in Primiceri,
%                        stacked in a vector
varname(4,:) = 'UU'; % VAR residuals
varname(5,:) = 'OM'; % The log standard deviations--the log(sigma(t)) in Primiceri
varname(6,:) = 'S1'; % S1
varname(7,:) = 'S2'; % S2
varname(8,:) = 'S3'; % S3
varname(9,:) = 'S4'; % S4
varname(10,:) = 'VD'; % standard deviation of vol. innovation
varname(11,:) = 'DD';
varname(12,:) = 'II';
varname(13,:) = 'S5';
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THE ERGODIC DISTRIBUTION:
maxshakes = 200;   % maximum number of attempts at stable draw (if stability constraint imposed)
gc = 1;
%
SB = zeros(N*(1+N*L),T,NG);         % Draws of the state vector
QB = zeros(N*(1+N*L),N*(1+N*L),NG); % Draws of covariance matrix for state innovations
%
% Stochastic volatilities: diagonal elements
HB = ones(T+1,N,NG); % Stochastic volatilities
SVB = zeros(N,NG);       % Standard error for volatility innovation
%
% Stochastic volatilities: off-diagonal elements
AB = zeros((N*(N-1))/2,T+1,NG); % The off-diagonal elements
SB1 = zeros(1,NG);               % Standard error for volatility innovation
SB2 = zeros(2,2,NG);             % Standard error for volatility innovation
SB3 = zeros(3,3,NG);             % Standard error for volatility innovation
SB4 = zeros(4,4,NG);             % Standard error for volatility innovation
SB5 = zeros(5,5,NG);
%
% Reduced-form VAR residuals:
UB=zeros(T,N,NG);
%
SD = zeros(size(SB));                        % states
QD = zeros(size(QB));                        % state innovation variance
VD = zeros(size(SVB));                        % standard deviation of vol. innovation
OM = zeros(size(HB,1)-1,size(HB,2),size(HB,3)); % stochastic volatilities
UU= zeros(size(UB));                          % VAR residuals
S1= zeros(size(SB1));                        % S1
S2= zeros(size(SB2));                        % S2
S3= zeros(size(SB3));                        % S3
S4= zeros(size(SB4));                        % S4
S5 = zeros(size(SB5));
AA= zeros(size(AB,1),T,size(AB,3));        % Stochastic volatilities: below diagonal elements of A(t)
%
disp('Starting Monte Carlo Simulation')
dd=1;
%
while dd <= D   % to get 2000 simulations keeping every 10th draw (D=20)
    iter = 5;   % provide buffer for back steps in subsequent files
    gc = 1;
    while iter <= NG  
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (1) Drawing states:
        [S0,P0,P1] = kfP(YS,XS,QA(:,:,iter-1),AAA(:,:,iter-1),H(:,:,iter-1),SI,PI,T,N,L);
        SA(:,:,iter) = gibbs1(S0,P0,P1,T,N,L);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (2) Draw Q conditional on states:
        % The posterior estimate of the covariance matrix for the innovations to the random walk parameters of
        % the the VAR, the v(t)'s:
        [TQ,DF] = iwpQ(SA(:,:,iter),T,TQ0,df);
        % Drawing Q:
        QA(:,:,iter) = gibbs2Q(TQ,DF,N,L);
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
        elseif N==6
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
            % S5:
            [TS5,DFS5] = iwpQ(AAA(11:15,:,iter-1),T,TS5_0,dfS5);
            PS=real(sqrtm(inv(TS5))); 
            u=randn(5,DFS5);
            SS5(:,:,iter) = inv(PS*(u*u')*PS');
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
        elseif N==6
            AAA(:,:,iter)=getalphasix(U(:,:,iter),SS1(1,iter),SS2(:,:,iter),SS3(:,:,iter),SS4(:,:,iter),SS5(:,:,iter),muA0,ssA0,T,H(:,:,iter-1),N);
        end 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (6) Orthogonalize the VAR innovations:
        for t=1:T
            CF=chofac(N,AAA(:,t,iter));
            f(t,:)=(CF*U(t,:,iter)')';
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
            v = ig2(v0,d0,eh(:,i)); % The draw for the volatility
            SV(i,iter) = v^.5;      % The standard deviation
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Drawing the volatilities: lh|ch,sv,b1,y
        % We do it on a univariate basis, as the stochastic volatilities are independent of one another:
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
            if (iter/D)-fix(iter/D)==0  %saves every 10th draw to reduce autocorrelation among draws
                % Sample from Markov chain:
                SD(:,:,(dd-1)*(NG/D)+(iter/D)) = SA(:,:,iter); % states
                QD(:,:,(dd-1)*(NG/D)+(iter/D)) = QA(:,:,iter); % state innovation variance
                VD(:,(dd-1)*(NG/D)+(iter/D)) = SV(:,iter); % Standard error for volatility innovation
                OM(:,:,(dd-1)*(NG/D)+(iter/D)) = H(2:size(H,1),:,iter); % stochastic volatilities
                AA(:,:,(dd-1)*(NG/D)+(iter/D)) = AAA(:,2:T+1,iter); % Stochastic volatilities: below diagonal elements of A(t)
                S1(1,(dd-1)*(NG/D)+(iter/D)) = SS1(1,iter); % S1
                S2(:,:,(dd-1)*(NG/D)+(iter/D)) = SS2(:,:,iter); % S2
                if N==4
                S3(:,:,(dd-1)*(NG/D)+(iter/D)) = SS3(:,:,iter); % S3
                elseif N==5
                S3(:,:,(dd-1)*(NG/D)+(iter/D)) = SS3(:,:,iter); % S3
                S4(:,:,(dd-1)*(NG/D)+(iter/D)) = SS4(:,:,iter); % S4
                elseif N==6
                S3(:,:,(dd-1)*(NG/D)+(iter/D)) = SS3(:,:,iter); % S3
                S4(:,:,(dd-1)*(NG/D)+(iter/D)) = SS4(:,:,iter); % S4
                S5(:,:,(dd-1)*(NG/D)+(iter/D)) = SS5(:,:,iter);
                end
                UU(:,:,(dd-1)*(NG/D)+(iter/D)) = U(:,:,iter); % VAR residuals
            end
            iter = iter+1;
            gc = 1;
            [dd iter]
            if (iter/100)-fix(iter/100)==0
                DD=dd;
                II=iter;
                save(DFILE2,varname(1,:),varname(2,:),varname(3,:),varname(4,:),varname(5,:),varname(6,:),varname(7,:),varname(8,:),varname(9,:),varname(10,:),varname(11,:),varname(12,:),varname(13,:));
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
        if max(lmax) < 1
            if (iter/D)-fix(iter/D)==0  %saves every 10th draw to reduce autocorrelation among draws
                % Sample from Markov chain:
                SD(:,:,(dd-1)*(NG/D)+(iter/D)) = SA(:,:,iter); % states
                QD(:,:,(dd-1)*(NG/D)+(iter/D)) = QA(:,:,iter); % state innovation variance
                VD(:,(dd-1)*(NG/D)+(iter/D)) = SV(:,iter); % Standard error for volatility innovation
                OM(:,:,(dd-1)*(NG/D)+(iter/D)) = H(2:size(H,1),:,iter); % stochastic volatilities
                AA(:,:,(dd-1)*(NG/D)+(iter/D)) = AAA(:,2:T+1,iter); % Stochastic volatilities: below diagonal elements of A(t)
                S1(1,(dd-1)*(NG/D)+(iter/D)) = SS1(1,iter); % S1
                S2(:,:,(dd-1)*(NG/D)+(iter/D)) = SS2(:,:,iter); % S2
                if N==4
                S3(:,:,(dd-1)*(NG/D)+(iter/D)) = SS3(:,:,iter); % S3
                elseif N==5
                S3(:,:,(dd-1)*(NG/D)+(iter/D)) = SS3(:,:,iter); % S3
                S4(:,:,(dd-1)*(NG/D)+(iter/D)) = SS4(:,:,iter); % S4
                elseif N==6
                S3(:,:,(dd-1)*(NG/D)+(iter/D)) = SS3(:,:,iter); % S3
                S4(:,:,(dd-1)*(NG/D)+(iter/D)) = SS4(:,:,iter); % S4
                S5(:,:,(dd-1)*(NG/D)+(iter/D)) = SS5(:,:,iter);
                end
                UU(:,:,(dd-1)*(NG/D)+(iter/D)) = U(:,:,iter); % VAR residuals
            end
            iter = iter+1;
            gc = 1;
            [dd iter]
            if (iter/100)-fix(iter/100)==0
                DD=dd;
                II=iter;
                save(DFILE2,varname(1,:),varname(2,:),varname(3,:),varname(4,:),varname(5,:),varname(6,:),varname(7,:),varname(8,:),varname(9,:),varname(10,:),varname(11,:),varname(12,:),varname(13,:));
            end
        elseif (max(lmax) > 1)&&(gc < maxshakes),
           % disp('States unstable, try again')
            gc = gc + 1;
        elseif (max(lmax) > 1)&&(gc == maxshakes),
            %disp('Fails Repeatedly, Step Back')
            iter = iter-1;
            gc = 1;
        end
        end
        
    end
    % Reinitialize gibbs arrays (buffer for back step)
    SA(:,:,1:4) = SA(:,:,NG-3:NG);
    QA(:,:,1:4) = QA(:,:,NG-3:NG);
    SV(:,1:4) = SV(:,NG-3:NG);
    H(:,:,1:4) = H(:,:,NG-3:NG);
    U(:,:,1:4) = U(:,:,NG-3:NG);
    AAA(:,:,1:4) = AAA(:,:,NG-3:NG);
    SS1(:,1:4) = SS1(:,NG-3:NG);
    SS2(:,:,1:4) = SS2(:,:,NG-3:NG);
    if N==4
    SS3(:,:,1:4) = SS3(:,:,NG-3:NG);
    elseif N==5
    SS3(:,:,1:4) = SS3(:,:,NG-3:NG);
    SS4(:,:,1:4) = SS4(:,:,NG-3:NG);
    elseif N==6
    SS3(:,:,1:4) = SS3(:,:,NG-3:NG);
    SS4(:,:,1:4) = SS4(:,:,NG-3:NG);
    SS5(:,:,1:4) = SS5(:,:,NG-3:NG);
    end
    dd=dd+1;
end

%tijd=etime(clock,t0);   %elapsed time
