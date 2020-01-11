function Y = sortir(X, a)

Y = zeros(size(X));

for i = 1:size(X,1)
    for j = 1:size(X,2)
        Y(i,j) = ((1+X(i,j)).^a-1)*100;
    end
end
