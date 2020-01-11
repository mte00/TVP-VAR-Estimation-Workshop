function [ adjw ] = normker( T, H )
% get weighting adjusted matrix from a normal kernel with size TxT, and
% Bandwidth H; the sum of weights on each row, normalised to sum up to 2H+1

ww=zeros(T,T);
for i=1:T
for j=1:T
z=(i-j)/H;      

ww(i,j)=(1./sqrt(2*pi))*exp(-(1/2).*(z^ 2));

end
end

s=sum(ww,2);
adjw=zeros(T,T); 

for k=1:T
    adjw(k,:)=(ww(k,:)/s(k));
end
cons=sum(adjw.^2,2);

for k=1:T
    adjw(k,:)=(1/cons(k))*(adjw(k,:));
end

end

