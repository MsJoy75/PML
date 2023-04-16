function [kmat,z] = knnlabel(Yp,Q, IDX, D)

Yp = Yp';
[m,q] = size(Yp);
[~,num] = size(IDX);
ynn = zeros(m,q); 
for i = 1:m
    y = zeros(1, q);
    for j = 1:num
        y = y + Yp(IDX(i, j),:) * exp(-(D(i,j))/2);
    end
    ynn(i,:) = y/num;
end
kmat = zeros(q,q);
z = 0; 
for i = 1:m
    dif = Yp(i,:) - ynn(i,:); % 1*q
    temp = Q' * dif';
    
    z = z + norm(temp, 2);
    
    kmat = kmat + temp * dif;
end
kmat = kmat/m;
z = z/m;
end

