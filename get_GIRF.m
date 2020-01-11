function [ir, wold] = get_GIRF(B,A0,ND,L,HORZ)
% Get GIRFs as in Equation (10) KPP(1996)
% B is N,N*L+1 coefficient matrix
% A0 is N x N covariance matrix 
% ND=1 which corresponds to whether the model has a constant or not.
% L is lag length
% HORZ is impulse horizon
N=size(B,1);
B=varcompanion(B,ND,N,L);
J=[eye(N) zeros(N, N*(L-1))];
ir=zeros(HORZ,N,N);
wold=[];

% GET MA coefficients
for h=0:HORZ
   wold = cat(3,wold,(J*(B^h)*J')); 
end

for i = 1:N
    sv=false(1,N);
    sv(i)=true;
    ss=(1/sqrt(A0(i,i)));
    for h=1:HORZ+1
        ir(h,:,i)=ss*sv*A0*wold(:,:,h)';
    end
end

ir=permute(ir,[2 3 1]);