function [RankingLosses,OneErrors,AveragePrecisions,coverage1,HammingLoss] = test(W,Xte,Yte,Weights_sizepre,Bias_sizepre)
%TEST 此处显示有关此函数的摘要
%   
labels = (W * Xte');
Yte(Yte == 0) = -1;
RankingLosses= Ranking_loss(labels,Yte');
OneErrors= One_error(labels,Yte');
% AveragePrecisions= Average_precision1(labels,Yte');
AveragePrecisions= Average_precision1(labels,Yte');
coverage1 = coverage(labels,Yte');


[num_test,~]=size(Yte);
[~,num_class]=size(Yte);

Outputs=labels';
Threshold=([Outputs,ones(num_test,1)]*[Weights_sizepre,Bias_sizepre]')';
Pre_Labels=zeros(num_test,num_class);


for i=1:num_test
    for k=1:num_class
        if(Outputs(i,k)>=Threshold(1,i))
            Pre_Labels(i,k)=1;
        else
            Pre_Labels(i,k)=-1;
        end
    end
end


HammingLoss=Hamming_loss(Pre_Labels',Yte');
end

