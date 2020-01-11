function em=varroots(L,N,b)

S=[eye(N*(L-1)),zeros(N*(L-1),N)];
A=[b(:,2:end);S];

em=eig(A);


        
       
