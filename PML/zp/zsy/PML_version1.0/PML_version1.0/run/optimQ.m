function [Q] = optimW(X,Y,P,Q,W,beta,sigma,gamma)
[d,~] = size(X);
[q,~] = size(Y);
%OPTIMW 此处显示有关此函数的摘要
%   X:d*m Y:q*m
[IDX,D] = knnsearch(X',X','k',11); % IDX存储的是每个样本点的近邻点的行，D保存的是每个近邻点的距离

[kmat,~] = knnlabel(Y,Q,IDX,D); % temp(i,:)保存的是第i个样本的所有近邻点标签的平均值
    % (0)
    thetaPre = 1;
    thetaCurr = 1;
    Qpre = Q;
    Qcurr = Qpre;
    L = 2;
    eta = 2;
    F1 = X*X';
    F2 = Q*Y* X';
    for i=1:50
        Z = Qcurr + thetaCurr*((1/thetaPre)-1)*(Qcurr-Qpre);
        
        svd_obj_temp = (W*X - Z*Y)*Y'+beta*(Z-P)+ sigma * kmat;
%         svd_obj_temp=svd_obj_temp_temp+lambda*(Z+W);
        svd_obj=Z-1/L*svd_obj_temp;
    
        Qpre=Qcurr;
        Qcurr = max(svd_obj-gamma/L,0)+min(svd_obj+gamma/L, 0);
        [~, traceNorm] = svdThreshold(svd_obj,beta/L);
        % 循环确定W
        % g(Wk+1)
        gQ = (1/2) * (norm(W * X - Q * Y, 'fro' ))^2 + (beta/2) * (norm(Q-P, 'fro' ))^2 +  sigma * kmat;
        a = gQ + gamma * traceNorm;    % 左边
        
        % h(Wk+1,Zk)
        gZ = (1/2) * (norm(W * X - Z * Y, 'fro' ))^2 + (beta/2) * (norm(Q-P, 'fro' ))^2 +  sigma * norm(kmat, 'fro');
        delta = sum(dot(((W*X-Z*Y)*Y'+beta * (Q-Z)+sigma+kmat) , (Q-Z)));
        h = gZ + delta + gamma * traceNorm;
        last = (L/2) * (norm(Q-Z, 'fro' ))^2;
        b = h + last; % 右边
        while (a > b)
            L = eta * L;
            Z = Qcurr + thetaCurr*((1/thetaPre)-1)*(Qcurr-Qpre);
        
            svd_obj_temp = (W*X - Z*Y)*Y'+beta*(Z-P)+ sigma * kmat;
    %         svd_obj_temp=svd_obj_temp_temp+lambda*(Z+W);
            svd_obj=Z-1/L*svd_obj_temp;
    
            
            
            Q = max(obj-gamma/L,0)+min(svd_obj+gamma/L, 0);
            [~, traceNorm] = svdThreshold(svd_obj,beta/L);
            
             gQ = (1/2) * (norm(W * X - Q * Y, 'fro' ))^2 + (beta/2) * (norm(Q-P, 'fro' ))^2 +  sigma * kmat;
            a = gQ + gamma * traceNorm;    % 左边

            % h(Wk+1,Zk)
            gZ = (1/2) * (norm(Z * X - Q * Y, 'fro' ))^2 + (beta/2) * (norm(Q-P, 'fro' ))^2 +  sigma * norm(kmat, 'fro');
            delta = sum(dot(((W*X-Z*Y)*Y'+beta * (Z-P)+sigma+kmat) , (W-Z)));
            h = gZ + delta + gamma * traceNorm;
            last = (L/2) * (norm(W-Z, 'fro' ))^2;
            b = h + last; % 右边
        end
        
        thetaPre = thetaCurr;
        thetaCurr = (sqrt(thetaCurr^4 + 4*(thetaCurr^2)) - (thetaCurr^2))/2;
        Qpre = Qcurr;
        Qcurr = Q;
        
    end
end
