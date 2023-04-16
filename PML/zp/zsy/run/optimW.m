function [W] = optimW(X,Y,Q,W,lambda4,lambda5)
[d,~] = size(X);
[q,~] = size(Y);

    thetaPre = 1;
    thetaCurr = 1;
    Wpre = W;
    Wcurr = Wpre;
    L = 2;
    eta = 2;
    F1 = X*X';
    F2 = Q*Y* X';
    for i=1:200
        
        
        Z = Wcurr + thetaCurr*((1/thetaPre)-1)*(Wcurr-Wpre);
        
        prepare = Z - (1/L)*((Z*F1 - F2) + lambda4 * Z);
        [W,traceNorm]=svdThreshold(prepare,lambda5/L);


        gW = (1/2) * (norm(W * X - Q * Y, 'fro' ))^2 + (lambda4/2) * (norm(W, 'fro' ))^2;
        a = gW + lambda5 * traceNorm;   
     
        gZ = (1/2) * (norm(Z * X - Q * Y, 'fro' ))^2 + (lambda4/2) * (norm(Z, 'fro' ))^2;
        delta = sum(dot(((Z*X-Q*Y)*X'+lambda4 * Z) , (W-Z)));
        h = gZ + delta + lambda5 * traceNorm;
        last = (L/2) * (norm(W-Z, 'fro' ))^2;
        b = h + last; 
        while (a > b)
            L = eta * L;
            prepare = Z - (1/L)*((Z*X - Q*Y)*X' + lambda4 * Z);
            [W,traceNorm]=svdThreshold(prepare,lambda5/L);
            
            gW = (1/2) * (norm(W * X - Q * Y, 'fro' ))^2 + (lambda4/2) * (norm(W, 'fro' ))^2;
            a = gW + lambda5 * traceNorm;    
            gZ = (1/2) * (norm(Z * X - Q * Y, 'fro' ))^2 + (lambda4/2) * (norm(Z, 'fro' ))^2;
            delta = sum(dot(((Z*X-Q*Y)*X'+lambda4 * Z) , (W-Z)));
            h = gZ + delta + lambda5 * traceNorm;
            last = (L/2) * (norm(W-Z, 'fro' ))^2;
            b = h + last; 
        end
        
        thetaPre = thetaCurr;
        thetaCurr = (sqrt(thetaCurr^4 + 4*(thetaCurr^2)) - (thetaCurr^2))/2;
        Wpre = Wcurr;
        Wcurr = W;
        
    end
end
