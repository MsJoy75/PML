clear;

% if you have any question about this paper or this code, please contact zhao_shiyi@163.com.


load('birds2.mat');
n_fold = 5;
[N, dim] = size(data);

target_partial(target_partial == -1) = 0;
partial_labels = target_partial;
target(target == -1) = 0;

indices = zeros(N, 1);

astep = int16(N/n_fold);
indices(1:1*(astep), :) = 1;
indices(1*(astep) +1:2*(astep), :) = 2;
indices(2*(astep) +1:3*(astep), :) = 3;
indices(3*(astep) +1:4*(astep), :) = 4;
indices(4*(astep) +1:end, :) = 5;

result = zeros(n_fold,5);
for k=1:n_fold
    test_idxs = (indices == k);
    train_idxs = ~test_idxs;

    Xtr=data(train_idxs,:);Yp=partial_labels(:,train_idxs)';Ytr=target(:,train_idxs)';
    Xte=data(test_idxs,:);
    Yte=target(:,test_idxs)';
        
    [Xtr, settings]=mapminmax(Xtr');
    Xte=mapminmax('apply',Xte',settings);
    Xtr(find(isnan(Xtr)))=0;
    Xte(find(isnan(Xte)))=0;
    Xtr=Xtr';
    Xte=Xte';
    
    [num_train,dim]=size(Xtr);
    [num_test,~]=size(Xte);

    Xtr = [Xtr, ones(num_train,1)];
    Xte = [Xte, ones(num_test,1)];
 
    [num_test,num_label] = size(Yte);
    

iter = 5;
lambda1 = 1000;
lambda2 = 1000;
lambda3 = 400;
lambda4 = 1;
lambda5 = 1;

    [W,L] = train(Xtr, Yp, iter, lambda1, lambda2,  lambda3, lambda4, lambda5);
    YPredict = W * Xte';
    
    %% Computing the size predictor using linear least squares model
    Outputs = (W*Xtr')';
    Left=Outputs;
    Right=zeros(num_train,1);
    for i=1:num_train
        temp=Left(i,:);
        [temp,index]=sort(temp);
        candidate=zeros(1,num_label+1);
        candidate(1,1)=temp(1)-0.1;
        for j=1:num_label-1
            candidate(1,j+1)=(temp(j)+temp(j+1))/2;
        end
        candidate(1,num_label+1)=temp(num_label)+0.1;
        miss_class=zeros(1,num_label+1);
        for j=1:num_label+1
            temp_notlabels=index(1:j-1);
            temp_labels=index(j:num_label);
            [~,false_neg]=size(setdiff(temp_notlabels,find(Ytr(i,:)==0)));
            [~,false_pos]=size(setdiff(temp_labels,find(Ytr(i,:)==1)));
            miss_class(1,j)=false_neg+false_pos;
        end
        [~,temp_index]=min(miss_class);
        Right(i,1)=candidate(1,temp_index);
    end
    Left=[Left,ones(num_train,1)];
    tempvalue=(Left\Right)';
    Weights_sizepre=tempvalue(1:num_label);
    Bias_sizepre=tempvalue(num_label+1);
    
    [RankingLosses,OneErrors,AveragePrecisions,coverage1,HammingLoss] = ...
    test(W,Xte,Yte,Weights_sizepre,Bias_sizepre);
    result(k,:) = [RankingLosses,OneErrors,AveragePrecisions,coverage1,HammingLoss];
    fprintf('Cross Validation: %d, rLoss: %.3f, oError: %.3f,avgPre : %.3f, conv: %.3f, hLoss: %.3f\n', ...
        k,RankingLosses,OneErrors,AveragePrecisions,coverage1,HammingLoss);
    
    
    
    
end
rr=sum(result)/n_fold;
rrstd = std(result,0,1);
