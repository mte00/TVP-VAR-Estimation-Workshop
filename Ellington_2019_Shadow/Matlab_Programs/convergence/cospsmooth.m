function [Cxy,Qxy]=cospsmooth(Cxy,Qxy,WIND,M,CROSSVAL)
%
% Luca Benati
% Bank of England
% Monetary Assessment and Strategy Division
% February 2003
%
% This program smooths the real and the imaginary parts of the cross-periodogram, thus getting the co-spectrum and the
% quadrature spectrum. Smoothing is performed by means of a window WIND of bandwidth M.
%                                 Input of the program is:
% Cxy  = the real part of the cross-periodogram
% Qxy  = the imaginary part of the cross-periodogram
% WIND = the spectral window
% M    = the bandwidth
%                                 Output of the program is:
% 
% Cxy  = the co-spectrum
% Qxy  = the quadrature spectrum
%
if nargin<5
    CROSSVAL=0;
end
if CROSSVAL>0
    WIND(fix(size(WIND,1)/2)+1)=0;
    WIND=WIND/sum(WIND);
end
%
% Here we smooth the real part of the cross-periodogram, thus getting the co-spectrum.
l=length(Cxy);
P_t=[(flipud(Cxy(2:M+1)))' Cxy' (flipud(Cxy(length(Cxy)-M-1:length(Cxy)-1)))']';
clear Cxy
for k=1:l
    Cxy(k,1)=sum(WIND.*P_t(k:2*M+k));
end
clear P_t
% Here we smooth the imaginary part of the cross-periodogram, thus getting the quadrature spectrum.
P_t=[(flipud(Qxy(2:M+1)))' Qxy' (flipud(Qxy(length(Qxy)-M-1:length(Qxy)-1)))']';
clear Qxy
for k=1:l
    Qxy(k,1)=sum(WIND.*P_t(k:2*M+k));
end