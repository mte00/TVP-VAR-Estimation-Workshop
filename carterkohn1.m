function [beta2,errors]=carterkohn1(beta0,p00,hlast,Q,Y,X)

%%Step 1 Set up matrices for the Kalman Filter
ns=cols(beta0);
beta_tt=[];          %will hold the filtered state variable
t=rows(Y);
ptt=zeros(t,ns,ns);    % will hold its variance
mu=0;
F=eye(ns);
%initialise the state variable
beta11=beta0; 
p11=p00;



for i=1:t
    x=X(i,:);
    R=hlast(i+1);
    %Prediction
beta10=mu+beta11*F';
p10=F*p11*F'+Q;
yhat=(x*(beta10)')';                                               
eta=Y(i,:)-yhat;
feta=(x*p10*x')+R;
%updating
K=(p10*x')*inv(feta);
beta11=(beta10'+K*eta')';
p11=p10-K*(x*p10);

ptt(i,:,:)=p11;
beta_tt=[beta_tt;beta11];

end
%%%%%%%%%%%end of Kalman Filter%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Carter and Kohn Backward recursion to calculate the mean and variance of the distribution of the state
%vector
beta2 = zeros(t,ns);   %this will hold the draw of the state variable
wa=randn(t,ns);
errors=zeros(t,1);
i=t;  %period t
p00=squeeze(ptt(i,:,:)); 
beta2(i,:)=beta_tt(i:i,:)+(wa(i:i,:)*cholx(p00));   
errors(i,:)=Y(i,:)-X(i,:)*beta2(i,:)';
%periods t-1..to 1

for i=t-1:-1:1
pt=squeeze(ptt(i,:,:));

bm=beta_tt(i:i,:)+(pt*F'*inv(F*pt*F'+Q)*(beta2(i+1:i+1,:)-mu-beta_tt(i,:)*F')')';  

pm=pt-pt*F'*inv(F*pt*F'+Q)*F*pt;  

beta2(i:i,:)=bm+(wa(i:i,:)*cholx(pm));  
errors(i,:)=Y(i,:)-X(i,:)*beta2(i,:)';

end