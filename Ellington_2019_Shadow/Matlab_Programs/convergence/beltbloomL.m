function LL=beltbloomL(M,X,FF)
%
% Luca Benati
% Bank of England
% Monetary Assessment and Strategy Division
% October 2004
%
% function LL=beltbloomL(M,X,FF)
% This program computes an approximated cross-validated log-likelihood in the frequency domain for a matrix of series X, based on
% the expression you find, e.g., in Hansen and Sargent, "Formulating and Estimating Dynamic Linear Rational Expectations Models",
% JEDC 1980, equation (34). The approximated cross-validated log-likelihood is computed as in Beltrao and P. Bloomfield (1987),
% "Determining the Bandwidth of a Kernel Spectrum Estimate", Journal of Time Series Analysis, based on the 'leave-one-out' method
% also discussed, e.g., in Marron, JS (1985), "An Asymptotically Efficient Solution to the Bandwidth Problem of Kernel Density
% Estimation", Annals of Statistics", 1011--1023.
%
[T,N]=size(X);
%
BART=bartwind(M);
% The spectral density matrix.
for hh=1:size(FF,2)
    for kk=1:size(FF,2)
        if hh==kk
            Y=FF(:,hh);
            % The spectra:
            % (1) This is the program for the case in which the number of observations is odd.
            if ((T/2)-fix(T/2))==0.5
                % The power spectrum.
                Py(1,1)=Y(1,1)^2;
                for j=2:length(Y)
                    Py(j,1)=2*Y(j,1)*conj(Y(j,1));
                end
            else
                % (2) This is the program for the case in which the number of observations is even.
                % The power spectrum.
                Py(1,1)=Y(1,1)^2;
                for j=2:length(Y)-1
                    Py(j,1)=2*Y(j,1)*conj(Y(j,1));
                end
                Py(length(Y),1)=Y(length(Y),1)*conj(Y(length(Y),1));
            end
            Py=(Py*T)/(4*pi);
            Iy=Py;
            % Here we smooth, and we get the consistent estimate of the spectrum:
            Py=specsmooth(Iy,M,BART,1);
            for ww=1:length(Py)
                S(hh,kk,ww)=Py(ww);
                I(hh,kk,ww)=Iy(ww);
            end
            clear Y,clear Py,clear Iy
        else
            Y=FF(:,hh);            
            Z=FF(:,kk);
            % The cross-periodogram:
            for j=1:length(Y)
                Ixy(j,1)=2*T*Y(j,1)*conj(Z(j,1));
            end
            % The cross-spectrum.
            Ixy=Ixy/(4*pi);
            % The co-spectrum and the quadrature spectrum.
            Cxy=real(Ixy);
            Qxy=imag(Ixy);
            [Cxy,Qxy]=cospsmooth(Cxy,Qxy,BART,M,1);
            for ww=1:length(Cxy)
                S(hh,kk,ww)=Cxy(ww)-i*Qxy(ww);
                I(hh,kk,ww)=real(Ixy(ww))-i*imag(Ixy(ww));
            end
            clear Y,clear Z,clear Cxy,clear Qxy,clear Ixy
        end
    end
end
S=S/T^2;
I=I/T^2;
%
% The approximated log-likelihood (this is based on expression 34 of Hansen and Sargent,
% "Formulating and Estimating Dynamic Linear Rational Expectations Models", JEDC 1980
%
for hh=1:size(S,3)
    lambda=real(eig(S(:,:,hh)));
    T1(hh,1)=log(prod(lambda(1:rank(S(:,:,hh)))));
end
for hh=1:size(S,3)
    if rank(S(:,:,hh))<size(S(:,:,hh),1)
        T2(hh,1)=trace(pinv(S(:,:,hh))*I(:,:,hh));
    else
        T2(hh,1)=trace(S(:,:,hh)\I(:,:,hh));
    end
end
T2=real(T2);
LL=-0.5*(sum(T1)+sum(T2));