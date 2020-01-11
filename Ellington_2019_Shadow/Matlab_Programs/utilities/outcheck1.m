function y=outcheck1(x)
%
% is based on function y=outcheck(x) by Luca Benati
% This function checks a series for outliers similar to the procedure described in Stock, J., and Watson, M. (2002)
% "Has the Business Cycle Changed and Why?", NBER Macroeconomics Annual 2002. 
% 
%                                        Input of the program is:
% x = the series of interest
%                                        Output of the program is:
% y = if there is no outlier, y=x; otherwise, y is the transformed series
%
T=length(x);
M=median(x);
X=sort(x);
Q1=X(fix(T*0.25),1);
Q3=X(fix(T*0.75),1);
range=Q3-Q1;
for t=4:T-4
    if abs(x(t,1)-M)<3*range
        x(t,1)=x(t,1);
    else
        x(t,1)=median([x(t-3:t-1,1)' x(t+1:t+3,1)']');
    end
end
y=x;