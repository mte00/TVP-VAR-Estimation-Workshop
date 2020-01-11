function data=lagn(data,m) 

%input: data matrix and lag m

data=data(m+1:end,:)-data(1:end-m,:);