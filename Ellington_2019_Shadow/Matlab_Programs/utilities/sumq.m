function m=sumq(x)

c=1;
for jx=1:3:size(x,1)-2
    m(c,1)=sum(x(jx:jx+2,1));
    c=c+1;
end