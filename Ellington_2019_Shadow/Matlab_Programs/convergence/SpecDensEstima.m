function [omegak,Py,Pyl,Pyu]=SpecDensEstima(y,alpha,M,PLOT,CONF,BOOT,L)
%
%		Luca Benati
%		Bank of England
%		Monetary Assessment and Strategy Division
%		Last modified: April 2004
%
% function [omegak,Py,Pyl,Pyu]=SpecDensEstima(y,alpha,M,PLOT,CONF,BOOT,L)
% This program estimates the spectrum for a previously defined time series y, by smoothing the estimated periodogram in the 
% frequency domain via a Bartlett window, and computes confidence bands for a pre-specified significance level alpha. M is the
% parameter which controls the width of the spectral window. The window width is equal to 2M+1. The width of the spectral window
% is determined automatically via the procedure proposed by Beltrao and Bloomfield.
%
%                                          Input of the program
%
% y       = the series whose spectrum we want to compute
% alpha   = the significance level for the confidence bands
% M       = the parameter governing the bandwidth of the spectral window (bandwidth is equal to 2*M+1); if M==-9999, M is chosen
%           automatically according to the procedure introduced by Beltrao and Bloomfield
% PLOT    = 'Y' if you want a plot, 'N' if not
% CONF    = 'BOOT' if you want bootstrapped confidence intervals, 'NOBO' if not.
% BOOT    = the bootstrapping procedure of choice, 'CHISQ' (based on the chi-square distribution), and 'NOCHI' (the second,
%           alternative procedure in Berkowitz and Diebold, section on the Franke-Hardle bootstrap)
% L       = the number of bootstrap replications
%
%
%                                          Output of the program
% omegak  = the Fourier frequencies
% Py      = the spectrum
% Pyl,Pyu = the lower and upper (alpha/2) and (1-alpha/2) % confidence bands for the spectrum
%
y=y(:);
n=length(y);
y=y-mean(y);
%
if M==-9999
    % M=beltrao(y);
    % M=fix(M/2);
    M=beltbloom(y);
else
    M=M;
end
%
BART=bartwind(M);
% Here we set V, the 'equivalent degrees of freedom' parameter for the spectral window (see Fuller, 'Introduction to Statistical
% Time Series', 1996, p. 374), or Koopmans (1974), "The Spectral Analysis of Time Series", p. 273, formula 8.39.
V=fix(2/sum(BART.^2));
%
% Here we compute the Fourier transforms of y and x and we keep only the first half.
%
Y=fft(y);
Y=Y(2:(fix(length(y)/2)+1))/n;
%
% The spectrum for y.
% (1) This is the program for the case in which the number of observations is odd.
for j=1:length(Y)
    Py(j,1)=2*Y(j,1)*conj(Y(j,1));
end
Py=(Py*n)/(4*pi);
Iy=Py;
% Here we smooth.
Py=specsmooth(Iy,M,BART);
%
% The Fourier frequencies.
omegak=linspace(0,pi,length(Py))';
% The confidence intervals.
Pyl=exp(log(Py)+log(V/chi2inv((alpha/2),V)));
Pyu=exp(log(Py)+log(V/chi2inv((1-alpha/2),V)));
%
if CONF=='BOOT'
    % Here the bootstrapped confidence intervals, and the median estimates:
    E=2*(Iy./Py);
    E=(2*E)/mean(E);
    for z=1:L
        if BOOT=='CHISQ'
            IBOOT=0.5*(chi2rnd(2,length(Py),1)).*Py;
        else
            for j=1:length(Py)
                IBOOT(j,1)=0.5*Py(j,1)*bootstrap(E);
            end
        end
        Pyboot(z,:)=specsmooth(IBOOT,M,BART)';
        clear IBOOT
    end
    Py=median(Pyboot)';
    Pyboot=sort(Pyboot);
    Pyl=Pyboot(fix(L*(alpha/2)),:)';
    Pyu=Pyboot(fix(L*(1-(alpha/2))),:)';
end
%
if PLOT=='N'
    return
end
%
% Here we plot the results.
plot(omegak,log(Py),'k',omegak,log([Pyl Pyu]),'r')
title('Log estimated spectrum and (1-alpha)% confidence bands (Window=Bartlett)')
xlabel('Omega')
axis([0 pi min(min(log([Pyl Pyu])))-0.1*abs(min(min(log([Pyl Pyu])))) max(max(log([Pyl Pyu])))-0.1*abs(max(max(log([Pyl Pyu]))))])