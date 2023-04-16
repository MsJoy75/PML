function [P] = requestP(Yp)


[m,q] = size(Yp);
P = zeros(q);
number = zeros(1,q);

for i = 1:q
    [~,~,v] = find(Yp(:,i)>0);
    [number(1, i),~] = size(v);
end

for i = 1:q
    for j = 1:q
        num = 0;
        for k = 1:m
            if Yp(k,i) == 1 && Yp(k,j)==1
                num = num + 1;
            end
        end
        p = num/number(1, i);
        P(i,j) = p;
    end
end


end

