function [Nspy, Nspp, Nspi, Nspm, Nspyy, Nsppp, Nspii, Nspmm, COH_MAT, bus_freq, VFy, VFpi, VFi, VFm] =...
    N_spec_persistence(DFILE2,N,L)
% This function computes the normalised spectral density of variables in
% the VAR at omega=0. Also we retrieve the coherence between the series
% based on the estimated coefficients in the VAR model. We follow Hamilton
% (1994) to compute this in the following manner: 
%
%   \hat{kappa}_ij(w) = \frac{[\hat{c}_ij(w)]^2 + \hat[\hat{q}_{ij}(w)]^2}{\hat{f}^{ii}_{t|T}(w)*\hat{f}^{jj}_{t|T}(w)}
%

% Also churns out the unconditional variance of our variables which are the
% diagonal elements of the integral of the spectral densities

load(DFILE2)
NN = size(SD,3)-1; % Number of gibbs sampler outputs 5000 draws

% Get Spectral densities
clear QD VD UU S1 S2 S3 SS % Clear stuff you dont want from loading the output form TVP_MCMC

i = sqrt(-1);
T = size(SD,2);
omeg = linspace(0,pi,T/2)';
omeg2 = (2*pi)./omeg;
omeg3 = round(omeg2);

for ij = 1:length(omeg3)
if omeg3(ij,:)==10
omeg4(ij,:)=1;
else
omeg4(ij,:)=0;
end
end
omeg5=omeg4==1;
[bus_freq,~] = find(omeg5==1);
bus_freq = bus_freq(1);

Nspy = zeros(T, NN);
Nspp = Nspy;
Nspi = Nspy;
Nspm = Nspy;

Spy = zeros(length(omeg),NN,T);
Spp = Spy;
Spi = Spy;
Spm = Spy;


Nspyy = zeros(T,length(omeg),NN); 
Nsppp = Nspyy;
Nspii = Nspyy; 
Nspmm = Nspyy;

% Unconditional variances
VFy = zeros(T,NN);
VFpi = VFy;
VFi = VFy;
VFm = VFy;

COH_MAT = zeros(N,N,length(omeg),T,NN);

kk = 1;
while kk<=size(SD,2)
    T-kk+1;
    ll=1;
    while ll <=NN
        BETA = SD(:,kk,ll);
        bet = reshape(BETA',1+N*L,N)';
        bet = bet(:,2:end);
        for jl = 1:L
            B(:,:,jl) = bet(:,(jl-1)*N+1:(jl-1)*N+N); % 3rd dimension of B 
            % is the lagged coefficients in each equation e.g. B(:,:,2)
            % contains the coefficients of the second lag of endogenous
            % variables.
        end
        iA = inv(chofac(N,AA(:,kk,ll))); % Lower triangular A_{t}
        VAR = iA*diag(OM(kk,:,ll))*iA'; % This is the reduced form covariance matrix of the VAR
        SSS = zeros(N,N,length(omeg));
        C = zeros(N,N,length(omeg));
        for w = 1:length(omeg)
            GI = eye(N);
            for bl = 1:size(B,3)
                GI = GI - B(:,:,bl)*exp(-i*bl*omeg(w));
            end
            SSS(:,:,w) = (GI\VAR)*inv(GI');
            % SSS(:,:,w) = (1/(2*pi))*(GI\VAR)*inv(GI');
            temp = zeros(N,N);
            for vl = 1:N
                for xl = 1:N
                    temp(vl,xl) = ((real(SSS(vl,xl,w))^2) + (imag(SSS(vl,xl,w))^2))/(real(SSS(vl,vl,w))*real(SSS(xl,xl,w)));
                end
            end
            C(:,:,w) = temp;
        end
        Spy(:,ll,kk) = real(squeeze(SSS(1,1,:)));
        Spp(:,ll,kk) = real(squeeze(SSS(2,2,:)));
        Spi(:,ll,kk) = real(squeeze(SSS(3,3,:)));
        Spm(:,ll,kk) = real(squeeze(SSS(4,4,:)));
        clear SSS
        
        VFy(kk,:) = sum(Spy(:,ll,kk));
        VFpi(kk,:) = sum(Spp(:,ll,kk));
        VFi(kk,:) = sum(Spi(:,ll,kk));
        VFm(kk,:) = sum(Spm(:,ll,kk));
        Nspy(kk,ll) = Spy(1,ll,kk)/sum(Spy(:,ll,kk));
        Nspp(kk,ll) = Spp(1,ll,kk)/sum(Spp(:,ll,kk));
        Nspi(kk,ll) = Spi(1,ll,kk)/sum(Spi(:,ll,kk));
        Nspm(kk,ll) = Spm(1,ll,kk)/sum(Spm(:,ll,kk));
        
        %for jl = 1:length(omeg)
        Nspyy(kk,:,ll) = Spy(:,ll,kk)./sum(Spy(:,ll,kk));
        Nsppp(kk,:,ll) = Spp(:,ll,kk)./sum(Spp(:,ll,kk));
        Nspii(kk,:,ll) = Spi(:,ll,kk)./sum(Spi(:,ll,kk));
        Nspmm(kk,:,ll) = Spm(:,ll,kk)./sum(Spm(:,ll,kk));
        %end
        COH_MAT(:,:,:,kk,ll) = C;
        ll = ll + 1;
    end
    kk = kk + 1;
end

        