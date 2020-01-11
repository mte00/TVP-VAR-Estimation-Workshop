function XX=minindc(X)
%
% Luca Benati
% Bank of England
% Monetary Assessment and Strategy Division
% August 2003
%
% function x=minindc(X)
% This function is modelled on the Gauss function minindc, and it returns a column vector x containing the index (i.e., the row number)
% of the smallest element in each column of a matrix X.
%
[R,C]=size(X);
IND=(1:1:R)';
%
for j=1:C
    X(:,j)=(X(:,j)==min(X(:,j)));
end
%
for j=1:C
    if sum(X(:,j))>1
        for t=1:R
            if sum(X(1:t,j))>1
                X(t,j)=0;
            end
        end
    end
end
%
for j=1:C
    XX(j,1)=IND'*X(:,j);
end