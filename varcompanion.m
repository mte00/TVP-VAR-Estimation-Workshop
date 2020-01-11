function A=varcompanion(A,ndet,n,p)
%creates companion matrix of A 
A=A(:,ndet+1:end);   
A=[A; eye(n*(p-1)) zeros(n*(p-1),n)];