function V=doublej(a1,b1);
%DOUBLEJ.M
%  function V=doublej(a1,b1);
%  Computes infinite sum V given by
%
%         V = SUM (a1^j)*b1*(a1^j)'
%
%  where a1 and b1 are each (n X n) matrices with eigenvalues whose moduli are
%  bounded by unity, and b1 is an (n X n) matrix.
%  The sum goes from j = 0 to j = infinity.  V is computed by using
%  the following "doubling algorithm".  We iterate to convergence on
%  V(j) on the following recursions for j = 1, 2, ..., starting from
%  V(0) = b1:
%
%       a1(j) = a1(j-1)*a1(j-1)
%       V(j) = V(j-1) + a1(j-1)*V(j-1)*a1(j-1)'
%
%  The limiting value is returned in V.
%
alpha0=a1;

gamma0=b1;

diff=5;  ijk=1;

while diff > 1e-15;

  alpha1=alpha0*alpha0;
  gamma1=gamma0+alpha0*gamma0*alpha0';
  diff=max(max(abs(gamma1-gamma0)));
  gamma0=gamma1;
  alpha0=alpha1;

  ijk=ijk+1; if ijk > 50;
             disp('Error: check aopt and c for proper configuration')
             end

end;

V=gamma1;