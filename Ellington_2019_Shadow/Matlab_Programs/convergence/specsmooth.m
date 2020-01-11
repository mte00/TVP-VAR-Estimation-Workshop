function Py=specsmooth(Iy,M,WIND,CROSSVAL)
%
% Luca Benati
% Bank of England
% Monetary Assessment and Strategy Division
% February 2003
%
% This program smooths a periodogram Iy to get the estimated spectrum, Py, by means of a window WIND of bandwidth M.
%                                          Input of the program is:
% Iy   = the periodogram
% WIND = the spectral window
% M    = the bandwidth
%                                          Output of the program is:
% 
% Py   = the spectrum
%
if nargin<4
    CROSSVAL=0;
end
if CROSSVAL>0
    WIND(fix(size(WIND,1)/2)+1)=0;
    WIND=WIND/sum(WIND);
end
%
Py=Iy;
Py(1,1)=2*Py(1,1);
Py(length(Py),1)=2*Py(length(Py),1);
% Here we save the length of the spectrum for future use.
l=length(Py);
% Here we 'augment' the spectrum vector in order to have enough
% observations to smooth it.
P_t=[(flipud(Py(2:M+1)))' Py' (flipud(Py(length(Py)-M-1:length(Py)-1)))']';
clear Py
% Here we actually do the smoothing.
for k=1:l
    Py(k,1)=sum(WIND.*P_t(k:2*M+k));
end
