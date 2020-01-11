function M=beltbloom(X)
%
% Luca Benati
% Bank of England
% Monetary Assessment and Strategy Division
% October 2004
%
% function M=beltbloom(X)
% This program selects the optimal bandwidth for a Bartlett spectral window estimator of the spectral density matrix based on
% a multivariate version of the algorithm proposed by  K. I. Beltrao and P. Bloomfield" (1987), "Determining the Bandwidth of
% a Kernel Spectrum Estimate", Journal of Time Series Analysis, 21--38. 
%
%                                             Input of the program is:
% X    = a TxN matrix of series
%
%                                             Output of the program is:
% M    = the estimated bandwidth
%
X=demean(X);
[T,N]=size(X);
%
% Fourier-transforming:
for jj=1:N
    x=fft(X(:,jj));
    FF(:,jj)=x(1:(fix(T/2)+1));
end
clear x
%
for mm=4:round(T^(1/2))
    LL(mm,1)=beltbloomL(mm,X,FF);
end
M=maxindc(real(LL(4:round(T^(1/2)))))+3;