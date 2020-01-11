function X = ann_dat(FD, a, Y);

% This function creates annualised growth rates of matrix Y
% FD input is whether you want first differences, fourth differences or
% twelvth differences corresponding to the data you have and the growth
% rate you want annualising.
% a gives the power with which we are annualising the data, e.g. if i have
% quarterly data and i want quarterly growth rates annualsing: FD =1; a =
% 4;



    for i = 1:size(Y,2)
        X(:,i) = ((Y(FD+1:end,i)./Y(1:end-FD,i)).^a-1)*100;
    end


    