function Index=DetectNaN(x)

[R,C]=size(x);
if sum(sum((x<0)+(x==0)+(x>0)))==R*C
    Index=0;
else
    Index=1;
end
