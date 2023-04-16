function [Q] = optimW(X,Y,P,Q,W,beta,sigma,gamma)
[d,~] = size(X);
[q,~] = size(Y);


[IDX,D] = knnsearch(X',X','k',11); 

[kmat,~] = knnlabel(Y,Q,IDX,D); 
    
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

        svd_obj=Z-1/L*svd_obj_temp;
    
        Qpre=Qcurr;
        Qcurr = max(svd_obj-gamma/L,0)+min(svd_obj+gamma/L, 0);
        [~, traceNorm] = svdThreshold(svd_obj,beta/L);
       
        gQ = (1/2) * (norm(W * X - Q * Y, 'fro' ))^2 + (beta/2) * (norm(Q-P, 'fro' ))^2 +  sigma * kmat;
        a = gQ + gamma * traceNorm;    
        
        gZ = (1/2) * (norm(W * X - Z * Y, 'fro' ))^2 + (beta/2) * (norm(Q-P, 'fro' ))^2 +  sigma * norm(kmat, 'fro');
        delta = sum(dot(((W*X-Z*Y)*Y'+beta * (Q-Z)+sigma+kmat) , (Q-Z)));
        h = gZ + delta + gamma * traceNorm;
        last = (L/2) * (norm(Q-Z, 'fro' ))^2;
        b = h + last; 
        while (a > b)
            L = eta * L;
            Z = Qcurr + thetaCurr*((1/thetaPre)-1)*(Qcurr-Qpre);
        
            svd_obj_temp = (W*X - Z*Y)*Y'+beta*(Z-P)+ sigma * kmat;
 
            svd_obj=Z-1/L*svd_obj_temp;
    
            
            
            Q = max(obj-gamma/L,0)+min(svd_obj+gamma/L, 0);
            [~, traceNorm] = svdThreshold(svd_obj,beta/L);
            
             gQ = (1/2) * (norm(W * X - Q * Y, 'fro' ))^2 + (beta/2) * (norm(Q-P, 'fro' ))^2 +  sigma * kmat;
            a = gQ + gamma * traceNorm;    

            
            gZ = (1/2) * (norm(Z * X - Q * Y, 'fro' ))^2 + (beta/2) * (norm(Q-P, 'fro' ))^2 +  sigma * norm(kmat, 'fro');
            delta = sum(dot(((W*X-Z*Y)*Y'+beta * (Z-P)+sigma+kmat) , (W-Z)));
            h = gZ + delta + gamma * traceNorm;
            last = (L/2) * (norm(W-Z, 'fro' ))^2;
            b = h + last; 
        end
        
        thetaPre = thetaCurr;
        thetaCurr = (sqrt(thetaCurr^4 + 4*(thetaCurr^2)) - (thetaCurr^2))/2;
        Qpre = Qcurr;
        Qcurr = Q;
        
    end
end
