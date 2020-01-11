function Y=demean(X)
% This code demeans a matrix X.
[R,C]=size(X);
for jj=1:C
    Y(:,jj)=X(:,jj)-mean(X(:,jj));
end