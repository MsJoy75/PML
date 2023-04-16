function [W,L] = train(Xtr, Yp, epoch, lambda1, lambda2, lambda3, lambda4, lambda5)

[m,d] = size(Xtr);
[~,q] = size(Yp);

Xtr = Xtr'; 
Yp = Yp'; 

W = 0.01 * randi([-1,1],q,d);
Q = randi([0,1],q,q);

P = requestP(Yp')';  
Ymulti = Yp * Yp';

[IDX,D] = knnsearch(Xtr',Xtr','k',11); 

[kmat,~] = knnlabel(Yp,Q,IDX,D); 

L = zeros(20, 1);

for i=1:epoch
    
    W = optimW(Xtr,Yp,Q,W,lambda4,lambda5);

    H = Q - ((W*Xtr*Yp'-Q*Ymulti) + lambda1*(Q-P) +lambda2* kmat)/100000;
    
   
        for k=1:q
            for j=1:q
                if (H(k,j)>lambda3/100000)
                    Q(k,j)=H(k,j)-lambda3/100000;
                elseif (H(k,j) < -lambda3/100000)
                    Q(k,j)=H(k,j)+lambda3/100000;
                else
                    Q(k,j)=0;
                end
            end
        end
   
    
    [kmat,z] = knnlabel(Yp,Q,IDX,D);
    loss = 1/2 * norm((W*Xtr - Q*Yp), 'fro') + lambda1/2*norm((Q-P), 'fro') +  lambda3*norm(Q, 1) + (lambda2/2)*z + lambda5*(sum(svd(W))) + lambda4*(norm(W, 'fro'));
    L(i, 1) = loss;
    if i>1
        if abs( L(i-1,1) -L(i,1))<0.0001
            break;
        end
    end
end


end

