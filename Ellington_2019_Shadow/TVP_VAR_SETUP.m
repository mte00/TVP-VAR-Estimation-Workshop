% ESTIMATION CODE TO GET REDUCED FORM PARAMETERS OF ELLINGTON (2019) 
% CAN ESTIMATE STANDARD PRIMICERI (2005) MODEL AND AN EXTENDED TVP VAR MODEL
% THAT ALLOWS PARAMETER INNOVATIONS TO FOLLOW STOCHASTIC VOLATILITY PROCESS

clear all; clc; 

addpath('Matlab_Programs');
addpath('Matlab_Programs\convergence','Matlab_Programs\estimate','Matlab_Programs\reduced','Matlab_Programs\utilities');
addpath('Results');
addpath('data')
warning('off', 'all');
% USE BVAR MINNESOTA PRIOR ON PARAMETERS AND IW PRIOR ON COVARIANCE MATRIX
L = 2;
TP = 20; % Training Sample in Years
ident = 0; % 0 Sign restrictions, 1 Cholesky
NB=10000;
D=10;
NG=1000;
SC=1;
Lambda='CogSa'; R=500; N=3;

time = (1956:0.25:2017.75)';
data= xlsread('US','Sheet2','B2:E249');
SHADOW =0;


G1=1; % 1=Standard TVP VAR as in Primiceri (2005). 
      % 0=add extra layer of stochastic volatility to parameters.
if G1==1
    if SHADOW==0
        DFILE1 = '<ENTER YOUR PATH HERE>\Results\Reduced\US_Burn';
        DFILE2 = '<ENTER YOUR PATH HERE>\Results\Reduced\US_MCMC';
        y=data(:,1:3);
    elseif SHADOW == 1
        DFILE1 = '<ENTER YOUR PATH HERE>\Results\Reduced\US_SHADBurn';
        DFILE2 = '<ENTER YOUR PATH HERE>\Results\Reduced\US_SHADMCMC';
        y=[data(:,1:2), data(:,4)];
    end
             
    
 a = TVP_BURN_Minn(DFILE1, y, NB, NG, L, TP, SC, Lambda);
 c = TVP_MCMC_Minn(DFILE1, DFILE2, y, D, NG, L, TP, SC, Lambda);

if SHADOW == 0
save G1_US_BASE
elseif SHADOW == 1
save G1_US_SHAD
end
elseif G1==0;
        if SHADOW==0
            DFILE1 = '<ENTER YOUR PATH HERE>\Results\Reduced\US_BurnG2';
            DFILE2 = '<ENTER YOUR PATH HERE>\Results\Reduced\US_MCMCG2';
        y=data(:,1:3);
        elseif SHADOW == 1
            DFILE1 = '<ENTER YOUR PATH HERE>\Results\Reduced\US_SHADBurnG2';
            DFILE2 = '<ENTER YOUR PATH HERE>\TVP_VAR_N3_US\Results\Reduced\US_SHADMCMCG2';
        y=[data(:,1:2), data(:,4)];
        end
        
a = TVP_BURN_G2(DFILE1, y, NB, NG, L, TP, SC, Lambda);
c = TVP_MCMC_G2(DFILE1, DFILE2, y, D, NG, L, TP, SC, Lambda);

if SHADOW == 0
save G2_US_BASE
elseif SHADOW == 1
save G2_US_SHAD
end
end




