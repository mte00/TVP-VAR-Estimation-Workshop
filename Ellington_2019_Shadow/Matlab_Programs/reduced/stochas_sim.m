function [Y_B] = stochas_sim(sd, aa, qd, s1, s2, vd, om, YY, HOR, N, L, SC)


% Michael Ellington 12/01/2017

% Step 1: Simulate VAR elements into the future
%        1) Simulate VAR covariance matrix
% Shocks to A_{t} matrix
% NOTE: since we are simulating the responses in each point in time A_{t} =
% A at each observation.
aa_e = real(sqrtm([s1 zeros(1,2); zeros(2,1) s2]))*randn((N*(N-1))/2,HOR+L+1);
% This puts shocks in the block diagonal matrix S (see Eq(9) in paper)
% Now simulate elements of matrix through the horizon
for jj = 1:HOR+L+1
    aa(:,:,jj+1) = aa(:,:,jj) + aa_e(:,jj);
end

% Shocks to the H_{t} matrix 
om_e = diag(vd)*randn(N,HOR+L+1);
% Simulate H_{t} matrix over the horizon
om = log(om);
for jj = 1:HOR+L+1
    om(:,:,jj+1) = om(:,:,jj) + om_e(:,jj);
end
om = exp(om);
% Compute the VARs reduced-form covariance matrix
VAR=zeros(N,N,HOR+L+1);
for jj = 1:HOR+L+1
    iA = inv(chofac(N, aa(:,:,jj)));
    VAR(:,:,jj) = iA*diag(om(:,:,jj))*iA'; % \Omega_{t}
end
SI=sd;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                       (2) Simulating the matrix B:
B=zeros(N,N*L+1,HOR+L);
jj=2; % Simulating the matrix B:
trial=0;
maxtrial=50;
while jj<=HOR+L
    dd=sd(:,:,jj-1)+real(mysqrt_varm(qd))*randn(N*(1+N*L),1);
    b = [dd(1:1+N*L,1)'; dd((1+N*L)+1:2*(1+N*L),1)';...
        dd(2*(1+N*L)+1:3*(1+N*L),1)'];
    
    if SC==0
        sd(:,:,jj)=dd;
        B(:,:,jj)=b;
        jj=jj+1;
        
    else
    if max(abs(varroots(L,N,b)))<1
        sd(:,:,jj)=dd;
        B(:,:,jj)=b;
        jj=jj+1;
        trial=0;
    elseif max(abs(varroots(L,N,b)))>1 && trial<maxtrial
        trial=trial+1;
    elseif max(abs(varroots(L,N,b)))>1 && trial==maxtrial
        y_sim = zeros(N,HOR+L);
        return
    end
    end
end

Y_B = [YY zeros(N, HOR)]; 
shocks = randn(N,HOR);
for tt = 1+L:HOR+L   
    epsilon = mysqrt_varm(VAR(:,:,tt-L))*shocks(:,tt-L);
    Y_B(:,tt) = B(:,:,tt-L)*[1; myvec(myfliplr(Y_B(:,tt-L:tt-1)))] + epsilon;
end