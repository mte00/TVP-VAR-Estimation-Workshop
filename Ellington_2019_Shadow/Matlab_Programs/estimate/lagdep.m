function [X,Y,X1]=lagdep(N,L,T,y)

X1 = zeros(T-L,1+(N*L));
X1(1:T-L,1) = ones(T-L,1);
for i = 1:L
    X1(1:T-L,2+(N*(i-1)):1+(i*N)) = y(1+L-i:T-i,:);
end
Y = y(1+L:T,:)';
%
[rx1,cx1] = size(X1');
[rx2,cx2] = size(X1');
[rx3,cx3] = size(X1');
[rx4,cx4] = size(X1');
[rx5,cx5] = size(X1');
[rx6,cx6] = size(X1');
%
if N==3
    data = [X1';X1';X1'];
    rd = size(data,1);
    onemat = zeros(rd,size(y,2));
    onemat(1:rx1,1) = ones(rx1,1);
    onemat(rx1+1:rx1+rx2,2) = ones(rx2,1);
    onemat(rx1+rx2+1:rx1+rx2+rx3,3) = ones(rx3,1);
    %
    temp=zeros(rd,N,length(Y));
    X=zeros(rd,N,length(Y));
    for i = 1:cx1
        temp(:,:,i) = ndgrid(data(:,i),ones(1,N));
        X(:,:,i) = temp(:,:,i).*onemat;
    end
    
elseif N==4
    data = [X1';X1';X1';X1'];
    rd = size(data,1);
    onemat = zeros(rd,size(y,2));
    onemat(1:rx1,1) = ones(rx1,1);
    onemat(rx1+1:rx1+rx2,2) = ones(rx2,1);
    onemat(rx1+rx2+1:rx1+rx2+rx3,3) = ones(rx3,1);
    onemat(rx1+rx2+rx3+1:rd,4) = ones(rx4,1);
    %
    temp=zeros(rd,N,length(Y));
    X=zeros(rd,N,length(Y));
    for i = 1:cx1
        temp(:,:,i) = ndgrid(data(:,i),ones(1,N));
        X(:,:,i) = temp(:,:,i).*onemat;
    end
    
elseif N==5
    data = [X1';X1';X1';X1';X1'];
    rd = size(data,1);
    onemat = zeros(rd,size(y,2));
    onemat(1:rx1,1) = ones(rx1,1);
    onemat(rx1+1:rx1+rx2,2) = ones(rx2,1);
    onemat(rx1+rx2+1:rx1+rx2+rx3,3) = ones(rx3,1);
    onemat(rx1+rx2+rx3+1:rx1+rx2+rx3+rx4,4) = ones(rx4,1);
    onemat(rx1+rx2+rx3+rx4+1:rd,5) = ones(rx5,1);
    %
    temp=zeros(rd,N,length(Y));
    X=zeros(rd,N,length(Y));
    for i = 1:cx1
        temp(:,:,i) = ndgrid(data(:,i),ones(1,N));
        X(:,:,i) = temp(:,:,i).*onemat;
    end
end

    
    

