function SA = gibbs1(S0,P0,P1,T,N,L)
% function SA = GIBBS1(YS,XS,S0,P0,P1,T,N,L);
% This file performs the first step of the Gibbs sampler, drawing from p(theta | Q,R,YS) 
% initialize arrays for Gibbs sampler
SA = zeros(N*(1+N*L),T);  % artificial states
%SM = zeros(N*(1+N*L),1); % backward update for conditional mean of state vector
%PM = zeros(N*(1+N*L),N*(1+N*L)); % backward update for projection matrix
%P = PM; % backward update for covariance matrix
wa = randn(N*(1+N*L),T); % draws for state innovations
% Backward recursions and sampling (smoothed estimates)
% Terminal state: draw from N(theta T|T, P T|T)
SA(:,T) = S0(:,T) + real(sqrtm(P0(:,:,T)))*wa(:,T);
% iterating back through the rest of the sample back to the beginning of
% the sample
for i = 1:T-1,
   PM = P0(:,:,T-i)*inv(P1(:,:,T-i+1));
   P = P0(:,:,T-i) - PM*P0(:,:,T-i);
   SM = S0(:,T-i) + PM*(SA(:,T-i+1) - S0(:,T-i));
   SA(:,T-i) = SM + real(sqrtm(P))*wa(:,T-i);
end

% The backward recursion updates the conditional means and variances to
% reflect the additional information about theta(t) contained in theta(t+1)