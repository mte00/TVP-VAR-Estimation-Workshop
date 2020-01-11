function BART=bartwind(M)
% This program creates a Bartlett spectral window for a pre-defined bandwidth M
w=(linspace(-pi,pi,2*M+1))';
w(M+1,1)=eps;
BART=(1/(2*pi*M))*((sin(0.5*w*M)./(sin(0.5*w))).^2);
BART=BART/sum(BART);
