function AA=getalphatwo(u,s1,muA0,ssA0,T,h,N)
%
u=[zeros(1,N); u];
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% alpha21:
Q=s1;
R=h(:,2);
st=zeros(T+1,1);
Pt=zeros(T+1,1);
sT=zeros(T+1,1);
PT=zeros(T+1,1);
s=zeros(T+1,1);
P=zeros(T+1,1);
st(1)=muA0(1);   % This is s(0|0)
Pt(1)=ssA0(1);   % This is P(0|0)
s(1)=muA0(1);    % This is s(1|0)
P(1)=ssA0(1)+Q;  % This is P(1|0)
for tt=2:T+1
    H=-u(tt,1);
    uhat=u(tt,2)-H'*s(tt-1);
    G=(P(tt-1)*H)*inv((R(tt)+H'*P(tt-1)*H));
    st(tt)=s(tt-1)+G*uhat;          % s(t|t)
    Pt(tt)=P(tt-1)-G*(H'*P(tt-1));  % P(t|t)
    s(tt)=st(tt);                   % s(t+1|t)
    P(tt)=Q+P(tt-1)-G*(H'*P(tt-1)); % P(t+1|t)
end
PT(T+1,1)=Pt(T+1);
for tt=1:T-1
    PT(T+1-tt,1)=Pt(T+1-tt,1)-Pt(T+1-tt,1)*(P(T+1-tt,1)\Pt(T+1-tt,1));
end
R=randn(T+1,1);
sT(T+1,1)=st(T+1)+R(T+1)*sqrt(PT(T+1));
for tt=1:T-1
    sT(T+1-tt,1)=st(T+1-tt,1)+Pt(T+1-tt,1)*(P(T+1-tt,1)\(sT(T+1-tt+1,1)-s(T+1-tt,1)))+R(T+1-tt)*sqrt(PT(T+1-tt));
end
%
AA=sT';
%
clear st s Pt P sT PT
%
