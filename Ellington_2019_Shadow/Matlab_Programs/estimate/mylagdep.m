function [X, Y, X1] = mylagdep(N, L, T, y)
% Extends Baumeister's function to allow for 8-variable model 

X1 = zeros(T-L, 1+(N*L)); % matrix to store RHS variables
X1(1:end,1) = ones(T-L,1); % put intercept in first column
for i = 1:L
    X1(1:T-L,2+(N*(i-1)):1+(i*N)) = y(1+L-i:T-i,:);
end

Y = y(1+L:T,:)'; % accounts for lags in the X1 matrix

[rx1, cx1] = size(X1');
[rx2, ~] = size(X1');
[rx3, ~] = size(X1');
[rx4, ~] = size(X1');
[rx5, ~] = size(X1');
[rx6, ~] = size(X1');
[rx7, ~] = size(X1');
[rx8, ~] = size(X1');

if N == 2
    data = [X1'; X1'];
    rd = size(data,1);
    onemat = zeros(rd,size(y,2));
    onemat(1:rx1,1) = ones(rx1,1);
    onemat(rx1+1:rx1+rx2,2) = ones(rx2,1);
    temp = zeros(rd, N, length(Y));
    X = zeros(rd, N, length(Y));
    for i = 1:cx1
        temp(:,:,i) = ndgrid(data(:,i), ones(N,1));
        X(:,:,i) = temp(:,:,i).*onemat;
    end
    % each row represents lagged values of each variable, the pages
    % represent t=0,...,t=T for each X lagged dependent variable. if N=2
    % and L=2 then in we have a block diagonal matrix where in the first
    % column. In X(:,:,1) where page 1 represents time t=0,
    % we have intercept in row 1, X1(-1) in row 2, X2(-1), X1(-2) in row 3
    % and X2(-2) in row 2 at time t=0!
elseif N== 3
    data = [X1'; X1'; X1'];
    rd = size(data,1);
    onemat = zeros(rd,size(y,2));
    onemat(1:rx1,1) = ones(rx1,1);
    onemat(rx1+1:rx1+rx2,2) = ones(rx2,1);
    onemat(rx1+rx2+1:rx1+rx2+rx3,3) = ones(rx3,1);
    %
    temp=zeros(rd, N, length(Y));
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
elseif N == 6
    data = [X1';X1';X1';X1';X1';X1'];
    rd = size(data,1);
    onemat = zeros(rd,size(y,2));
    onemat(1:rx1,1) = ones(rx1,1);
    onemat(rx1+1:rx1+rx2,2) = ones(rx2,1);
    onemat(rx1+rx2+1:rx1+rx2+rx3,3) = ones(rx3,1);
    onemat(rx1+rx2+rx3+1:rx1+rx2+rx3+rx4,4) = ones(rx4,1);
    onemat(rx1+rx2+rx3+rx4+1:rx1+rx2+rx3+rx4+rx5,5) = ones(rx5,1);
    onemat(rx1+rx2+rx3+rx4+rx5+1:rd,6) = ones(rx6,1);
    %
    temp=zeros(rd,N,length(Y));
    X=zeros(rd,N,length(Y));
    for i = 1:cx1
        temp(:,:,i) = ndgrid(data(:,i),ones(1,N));
        X(:,:,i) = temp(:,:,i).*onemat;
    end
elseif N == 7
    data = [X1';X1';X1';X1';X1';X1';X1'];
    rd = size(data,1);
    onemat = zeros(rd,size(y,2));
    onemat(1:rx1,1) = ones(rx1,1);
    onemat(rx1+1:rx1+rx2,2) = ones(rx2,1);
    onemat(rx1+rx2+1:rx1+rx2+rx3,3) = ones(rx3,1);
    onemat(rx1+rx2+rx3+1:rx1+rx2+rx3+rx4,4) = ones(rx4,1);
    onemat(rx1+rx2+rx3+rx4+1:rx1+rx2+rx3+rx4+rx5,5) = ones(rx5,1);
    onemat(rx1+rx2+rx3+rx4+rx5+1:rx1+rx2+rx3+rx4+rx5+rx6,6) = ones(rx6,1);
    onemat(rx1+rx2+rx3+rx4+rx5+rx6+1:rd,7) = ones(rx7,1);
    %
    temp=zeros(rd,N,length(Y));
    X=zeros(rd,N,length(Y));
    for i = 1:cx1
        temp(:,:,i) = ndgrid(data(:,i),ones(1,N));
        X(:,:,i) = temp(:,:,i).*onemat;
    end
elseif N==8
    data = [X1';X1';X1';X1';X1';X1';X1';X1'];
    rd = size(data,1);
    onemat = zeros(rd,size(y,2));
    onemat(1:rx1,1) = ones(rx1,1);
    onemat(rx1+1:rx1+rx2,2) = ones(rx2,1);
    onemat(rx1+rx2+1:rx1+rx2+rx3,3) = ones(rx3,1);
    onemat(rx1+rx2+rx3+1:rx1+rx2+rx3+rx4,4) = ones(rx4,1);
    onemat(rx1+rx2+rx3+rx4+1:rx1+rx2+rx3+rx4+rx5,5) = ones(rx5,1);
    onemat(rx1+rx2+rx3+rx4+rx5+1:rx1+rx2+rx3+rx4+rx5+rx6,6) = ones(rx6,1);
    onemat(rx1+rx2+rx3+rx4+rx5+rx6+1:rx1+rx2+rx3+rx4+rx5+rx6+rx7,7) = ones(rx7,1);
    onemat(rx1+rx2+rx3+rx4+rx5+rx6+rx7+1:rd,8) = ones(rx8,1);
    %
    temp=zeros(rd,N,length(Y));
    X=zeros(rd,N,length(Y));
    for i = 1:cx1
        temp(:,:,i) = ndgrid(data(:,i),ones(1,N));
        X(:,:,i) = temp(:,:,i).*onemat;
    end   
end