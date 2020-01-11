function [X1, X2, X3, X4] = TVP_vol_variable(L,N,NN,DFILE2)

% compute the unconditional standard deviations of variables from the TVP
% VAR model. Based on Baumeister and Peersman (2013) JAE.

load(DFILE2)
clear D DD II QD UU VD S1 S4 S3 S4

e_y = [1, zeros(1,7)]';
e_pi = [0 1 zeros(1,6)]';
e_i = [zeros(1,2) 1 zeros(1,5)]';
e_m = [zeros(1,3) 1 zeros(1,4)]';

cover = 0.16*2;
DELTOL = 1e-12;

X1 = zeros(size(SD,2), 3);
X2 = X1;
X3 = X1;
X4 = X1;

kk = 1
while kk <=size(SD,2)
    x1 = zeros(NN,1);
    x2 = x1; x3 = x1; x4 = x1;
    for ll = 1:NN
        BET = SD(:,kk,ll);
        bet = reshape(BET',1+N*L,N)';
        bet = bet(:,2:end);
        iA = inv(chofac(N,AA(:,kk,ll)));
        VAR = iA*diag(OM(kk,:,ll))*iA';
        
        BB = [bet; eye(N) zeros(N,N)];
        VAR = [VAR zeros(N,N); zeros(N,N) zeros(N,N)];
        % Variance of first variable
        V_old = 0;
        DEL = 1;
        zh = 0;
        while DEL > DELTOL
            BBzh = BB^zh;
            V_new = V_old + e_y'*BBzh*VAR*BBzh'*e_y;
            if zh >0
                DEL = (V_new - V_old)/V_old;
            end
            V_old = V_new;
            zh = zh + 1;
        end
        x1(ll) = sqrt(V_old);
        % Variance of second variable
        V_old = 0;
        DEL = 1;
        zh = 0;
        while DEL > DELTOL
            BBzh = BB^zh;
            V_new = V_old + e_pi'*BBzh*VAR*BBzh'*e_pi;
            if zh >0
                DEL = (V_new - V_old)/V_old;
            end
            V_old = V_new;
            zh = zh + 1;
        end
        x2(ll) = sqrt(V_old);
        % Variance of third variable
        V_old = 0;
        DEL = 1;
        zh = 0;
        while DEL > DELTOL
            BBzh = BB^zh;
            V_new = V_old + e_i'*BBzh*VAR*BBzh'*e_i;
            if zh >0
                DEL = (V_new - V_old)/V_old;
            end
            V_old = V_new;
            zh = zh + 1;
        end
        x3(ll) = sqrt(V_old);
        % Variance of fourth variable
        V_old = 0;
        DEL = 1;
        zh = 0;
        while DEL > DELTOL
            BBzh = BB^zh;
            V_new = V_old + e_m'*BBzh*VAR*BBzh'*e_m;
            if zh >0
                DEL = (V_new - V_old)/V_old;
            end
            V_old = V_new;
            zh = zh + 1;
        end
        x4(ll) = sqrt(V_old);
    end
    
    x1 = sort(x1);
    x2 = sort(x2);
    x3 = sort(x3);
    x4 = sort(x4);
    X1(kk,:) = x1(fix(NN*[cover/2 0.5 (1-cover)/2]')); 
    X2(kk,:) = x2(fix(NN*[cover/2 0.5 (1-cover)/2]')); 
    X3(kk,:) = x3(fix(NN*[cover/2 0.5 (1-cover)/2]')); 
    X4(kk,:) = x4(fix(NN*[cover/2 0.5 (1-cover)/2]')); 
    kk = kk + 1
end
save VOL_TVP_VARIABLES X1 X2 X3 X4
