% EXAMPLE4 QBLL TVP predictive regression

clear; clc;
seed=07012020;
rng(seed);
%addpath('src')

% load data
data=xlsread('Example_4_data','Sheet1','B2:I1002');

y=100*data(:,2:end); % get returns on same scale as market risk premium.
X=data(:,1);

% generate figure
figure(1)
subplot(1,2,1)
plot(y(:,:))
axis tight
ylabel('%')
legend('AAPL','AIG','AMZN','GOOGL','NFLX','NKE','UPS','Location','SouthOutside')
subplot(1,2,2)
plot(X(:,:),'b-')
axis tight
ylabel('%')
legend('Market Risk Premium','Location','SouthOutside')
%matlab2tikz('EX4_1.tex')
%%
tic;
[betas,vols,b0]=QBLL_univariate(y,X);
toc;
%%
figure(2)
subplot(2,7,1)
plot(squeeze(betas(1,1,2,:)),'r-','LineWidth',1.2)
hold on,
plot(squeeze(betas(1,1,1,:)),'r--')
plot(squeeze(betas(1,1,3,:)),'r--')
axis tight
ylabel('Intercept')
title('AAPL')
subplot(2,7,2)
plot(squeeze(betas(1,2,2,:)),'r-','LineWidth',1.2)
hold on,
plot(squeeze(betas(1,2,1,:)),'r--')
plot(squeeze(betas(1,2,3,:)),'r--')
axis tight
title('AIG')
subplot(2,7,3)
plot(squeeze(betas(1,3,2,:)),'r-','LineWidth',1.2)
hold on,
plot(squeeze(betas(1,3,1,:)),'r--')
plot(squeeze(betas(1,3,3,:)),'r--')
axis tight
title('AMZN')
subplot(2,7,4)
plot(squeeze(betas(1,4,2,:)),'r-','LineWidth',1.2)
hold on,
plot(squeeze(betas(1,4,1,:)),'r--')
plot(squeeze(betas(1,4,3,:)),'r--')
axis tight
title('GOOGL')
subplot(2,7,5)
plot(squeeze(betas(1,5,2,:)),'r-','LineWidth',1.2)
hold on,
plot(squeeze(betas(1,5,1,:)),'r--')
plot(squeeze(betas(1,5,3,:)),'r--')
axis tight
title('NFLX')
subplot(2,7,6)
plot(squeeze(betas(1,6,2,:)),'r-','LineWidth',1.2)
hold on,
plot(squeeze(betas(1,6,1,:)),'r--')
plot(squeeze(betas(1,6,3,:)),'r--')
axis tight
title('NKE')
subplot(2,7,7)
plot(squeeze(betas(1,7,2,:)),'r-','LineWidth',1.2)
hold on,
plot(squeeze(betas(1,7,1,:)),'r--')
plot(squeeze(betas(1,7,3,:)),'r--')
axis tight
title('UPS')
subplot(2,7,8)
plot(squeeze(betas(2,1,2,:)),'b-','LineWidth',1.2)
hold on,
plot(squeeze(betas(2,1,1,:)),'b--')
plot(squeeze(betas(2,1,3,:)),'b--')
axis tight
ylabel('\beta_{m}')
subplot(2,7,9)
plot(squeeze(betas(2,2,2,:)),'b-','LineWidth',1.2)
hold on,
plot(squeeze(betas(2,2,1,:)),'b--')
plot(squeeze(betas(2,2,3,:)),'b--')
axis tight
subplot(2,7,10)
plot(squeeze(betas(2,3,2,:)),'b-','LineWidth',1.2)
hold on,
plot(squeeze(betas(2,3,1,:)),'b--')
plot(squeeze(betas(2,3,3,:)),'b--')
axis tight
subplot(2,7,11)
plot(squeeze(betas(2,4,2,:)),'b-','LineWidth',1.2)
hold on,
plot(squeeze(betas(2,4,1,:)),'b--')
plot(squeeze(betas(2,4,3,:)),'b--')
axis tight
subplot(2,7,12)
plot(squeeze(betas(2,5,2,:)),'b-','LineWidth',1.2)
hold on,
plot(squeeze(betas(2,5,1,:)),'b--')
plot(squeeze(betas(2,5,3,:)),'b--')
axis tight
subplot(2,7,13)
plot(squeeze(betas(2,6,2,:)),'b-','LineWidth',1.2)
hold on,
plot(squeeze(betas(2,6,1,:)),'b--')
plot(squeeze(betas(2,6,3,:)),'b--')
axis tight
subplot(2,7,14)
plot(squeeze(betas(2,7,2,:)),'b-','LineWidth',1.2)
hold on,
plot(squeeze(betas(2,7,1,:)),'b--')
plot(squeeze(betas(2,7,3,:)),'b--')
axis tight
%matlab2tikz('EX4_2.tex')

%%

figure(3)
plot(sqrt(vols(:,:,2)),'LineWidth',1.5)
axis tight
title('Time-varying volatilities')
legend('AAPL','AIG','AMZN','GOOGL','NFLX','NKE','UPS','Location','SouthOutside')
%matlab2tikz('EX4_3.tex')
