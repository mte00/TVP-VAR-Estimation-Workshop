function AA=getalphafive(u,s1,s2,s3,s4,muA0,ssA0,T,h,N)
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
st(1)=muA0(1);    % This is s(0|0)
Pt(1)=ssA0(1);    % This is P(0|0)
s(1)=muA0(1);     % This is s(1|0)
P(1)=ssA0(1)+Q;   % This is P(1|0)

for tt=2:T+1
    H=-u(tt,1);
    uhat=u(tt,2)-H'*s(tt-1);
    G=(P(tt-1)*H)*inv((R(tt)+H'*P(tt-1)*H));
    st(tt)=s(tt-1)+G*uhat;               % s(t|t)
    Pt(tt)=P(tt-1)-G*(H'*P(tt-1));       % P(t|t)
    s(tt)=st(tt);                        % s(t+1|t)
    P(tt)=Q+P(tt-1)-G*(H'*P(tt-1));      % P(t+1|t)
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  alpha31 and alpha32:
Q=s2;
R=h(:,3);
st=zeros(2,1,T+1);
Pt=zeros(2,2,T+1);
sT=zeros(2,1,T+1);
PT=zeros(2,2,T+1);
s=zeros(2,1,T+1);
P=zeros(2,2,T+1);
st(:,:,1)=muA0(2:3);        % This is s(0|0)
Pt(:,:,1)=ssA0(2:3,2:3);    % This is P(0|0)
s(:,:,1)=st(:,:,1);         % This is s(1|0)
P(:,:,1)=Pt(:,:,1)+Q;       % This is P(1|0)

for tt=2:T+1
    H=[-u(tt,1) -u(tt,2)]';
    uhat=u(tt,3)-H'*s(:,:,tt-1);
    G=(P(:,:,tt-1)*H)*inv((R(tt)+H'*P(:,:,tt-1)*H));
    st(:,:,tt)=s(:,:,tt-1)+G*uhat;              % s(t|t)
    Pt(:,:,tt)=P(:,:,tt-1)-G*(H'*P(:,:,tt-1));  % P(t|t)
    s(:,:,tt)=st(:,:,tt);                       % s(t+1|t)
    P(:,:,tt)=Q+P(:,:,tt-1)-G*(H'*P(:,:,tt-1)); % P(t+1|t)
end
PT(:,:,T+1)=Pt(:,:,T+1);
for tt=1:T-1
    PT(:,:,T+1-tt)=Pt(:,:,T+1-tt)-Pt(:,:,T+1-tt)*(P(:,:,T+1-tt)\Pt(:,:,T+1-tt));
end
R=randn(2,T+1);
sT(:,:,T+1)=st(:,:,T+1)+real(sqrtm(PT(:,:,T+1)))*R(:,T+1);
for tt=1:T-1
    sT(:,:,T+1-tt)=st(:,:,T+1-tt)+Pt(:,:,T+1-tt)*(P(:,:,T+1-tt)\(sT(:,:,T+1-tt+1)-s(:,:,T+1-tt)))+real(sqrtm(PT(:,:,T+1-tt)))*R(:,T+1-tt);
end
%
AA=[AA; squeeze(sT(:,:,1:T+1))];
%
clear st s Pt P sT PT
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% alpha 41, alpha42 and alpha43:
%
Q=s3;
R=h(:,4);
st=zeros(3,1,T+1);
Pt=zeros(3,3,T+1);
sT=zeros(3,1,T+1);
PT=zeros(3,3,T+1);
s=zeros(3,1,T+1);
P=zeros(3,3,T+1);
st(:,:,1)=muA0(4:6);      %this is s(0|0)
Pt(:,:,1)=ssA0(4:6,4:6);  %this is P(0|0)
s(:,:,1)=st(:,:,1);       %this is s(1|0)
P(:,:,1)=Pt(:,:,1)+Q;     %this is P(1|0)
%forward recursion
for tt=2:T+1
    H=[-u(tt,1) -u(tt,2) -u(tt,3)]';
    uhat=u(tt,4)-H'*s(:,:,tt-1);
    G=(P(:,:,tt-1)*H)*inv((R(tt)+H'*P(:,:,tt-1)*H));
    st(:,:,tt)=s(:,:,tt-1)+G*uhat;                % s(t|t)
    Pt(:,:,tt)=P(:,:,tt-1)-G*(H'*P(:,:,tt-1));    % P(t|t)
    s(:,:,tt)=st(:,:,tt);                         % s(t+1|t)
    P(:,:,tt)=Q+P(:,:,tt-1)-G*(H'*P(:,:,tt-1));   % P(t+1|t)
end
%backward recursion
PT(:,:,T+1)=Pt(:,:,T+1);
for tt=1:T-1
    PT(:,:,T+1-tt)=Pt(:,:,T+1-tt)-Pt(:,:,T+1-tt)*(P(:,:,T+1-tt)\Pt(:,:,T+1-tt));
end
R=randn(3,T+1);
sT(:,:,T+1)=st(:,:,T+1)+real(sqrtm(PT(:,:,T+1)))*R(:,T+1);
for tt=1:T-1
    sT(:,:,T+1-tt)=st(:,:,T+1-tt)+Pt(:,:,T+1-tt)*(P(:,:,T+1-tt)\(sT(:,:,T+1-tt+1)-s(:,:,T+1-tt)))+real(sqrtm(PT(:,:,T+1-tt)))*R(:,T+1-tt);
end
%
AA=[AA; squeeze(sT(:,:,1:T+1))];   %this is a 6 x (T+1) matrix
%
clear st s Pt P sT PT
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% alpha 51, alpha 52, alpha 53 and alpha 54:
%
Q=s4;
R=h(:,5);
st=zeros(4,1,T+1);
Pt=zeros(4,4,T+1);
sT=zeros(4,1,T+1);
PT=zeros(4,4,T+1);
s=zeros(4,1,T+1);
P=zeros(4,4,T+1);
st(:,:,1)=muA0(7:10);       %this is s(0|0)
Pt(:,:,1)=ssA0(7:10,7:10);  %this is P(0|0)
s(:,:,1)=st(:,:,1);         %this is s(1|0)
P(:,:,1)=Pt(:,:,1)+Q;       %this is P(1|0)
%forward recursion
for tt=2:T+1
    H=[-u(tt,1) -u(tt,2) -u(tt,3) -u(tt,4)]';
    uhat=u(tt,5)-H'*s(:,:,tt-1);
    G=(P(:,:,tt-1)*H)*inv((R(tt)+H'*P(:,:,tt-1)*H));
    st(:,:,tt)=s(:,:,tt-1)+G*uhat; % s(t|t)
    Pt(:,:,tt)=P(:,:,tt-1)-G*(H'*P(:,:,tt-1)); % P(t|t)
    s(:,:,tt)=st(:,:,tt); % s(t+1|t)
    P(:,:,tt)=Q+P(:,:,tt-1)-G*(H'*P(:,:,tt-1)); % P(t+1|t)
end
%backward recursion
PT(:,:,T+1)=Pt(:,:,T+1);
for tt=1:T-1
    PT(:,:,T+1-tt)=Pt(:,:,T+1-tt)-Pt(:,:,T+1-tt)*(P(:,:,T+1-tt)\Pt(:,:,T+1-tt));
end
R=randn(4,T+1);
sT(:,:,T+1)=st(:,:,T+1)+real(sqrtm(PT(:,:,T+1)))*R(:,T+1);
for tt=1:T-1
    sT(:,:,T+1-tt)=st(:,:,T+1-tt)+Pt(:,:,T+1-tt)*(P(:,:,T+1-tt)\(sT(:,:,T+1-tt+1)-s(:,:,T+1-tt)))+real(sqrtm(PT(:,:,T+1-tt)))*R(:,T+1-tt);
end
%
AA=[AA; squeeze(sT(:,:,1:T+1))];    %this is a 10 x (T+1) matrix