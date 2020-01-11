function vardecomp = vardecomp(nvars,nsteps,ir)

%calculates variance decomposition and accumulate vardeco
resp6=zeros(nvars,nvars,nsteps);
resp7=zeros(nsteps,1);
vardecomp=zeros(nvars,nvars,nsteps);

for i=1:nvars
    for j=1:nvars
        resp = squeeze(ir(i,j,:));
        vardeco = resp.*resp;                 %corresponds to resp5 in RATS
        resp6(i,j,:)=cumsum(vardeco);  %variance of the forecast error: conditional variance
    end
    
    for k=1:nsteps
        temp = resp6(i,:,k);
        temp = sum(temp);
        resp7(k,1)=temp;
        % resp7(k,1)=resp6(i,1,k) + resp6(i,2,k) + resp6(i,3,k) + resp6(i,4,k);
    end
    resp8=resp7';   %unconditional variance (total variance of each variable)
   
    for j=1:nvars
        vardecomp(i,j,:) = squeeze(resp6(i,j,:))'./resp8; %conditional/unconditional variance       
    end
    
end