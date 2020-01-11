function a=vol_roll(data,k)

%computes standard deviation over rolling window
%Christiane Baumeister
%May 2011

%k is the length of the rolling window
for jx=1:size(data,1)-k
    a(jx,1)=std(data(jx:jx+k,1));
end
