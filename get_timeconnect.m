function [tc,tr,tt,nd,pwc]=get_timeconnect(N,HO,irf)

fev=vardecomp(N,HO,irf);
fev=fev(:,:,end);
FF=sum(sum(fev));
tc=100*(1-trace(fev)/FF);

for i=1:N
   temp1(i,:)=sum(fev(i,:))-fev(i,i);
   temp2(i,:)=sum(fev(:,i))-fev(i,i);
   tr(i,:)=100*temp1(i,:)./FF;
   tt(i,:)=100*temp2(i,:)./FF;
end
   nd=tt-tr;
   
% get pairwise net directional connectedness

pwc=zeros(N,N);
for j=1:N
    for i=1:N
       pwc(i,j)=100*(fev(j,i)-fev(i,j))/N; 
    end
end
   
   
   
end