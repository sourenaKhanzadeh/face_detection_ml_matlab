run('../vlfeat-0.9.21/toolbox/vl_setup')
load('pos_neg_feats.mat')

n_train = round(length(pos_feats) * .8, 0);
n_test = round(length(pos_feats) * .2, 0);

% sample = randsample(1:length(data), n_train);

feats = cat(1,pos_feats(1:n_train, :),neg_feats(1:n_train, :));
labels = cat(1,ones(n_train,1),-1*ones(n_train,1));

% testDataSet = cat(1,pos_feats(n_train+1:end, :),neg_feats(n_train+1:end, :));
testDataSet = cat(1,ones(n_test,1),-1*ones(n_test,1));

lambda = 0.1;
[w,b] = vl_svmtrain(feats',labels',lambda);

fprintf('Classifier performance on train data:\n')
confidences = [pos_feats(n_train+1:end, :); neg_feats(n_train+1:end, :)]*w + b;
% confidences = [pos_feats(1:n_train, :); neg_feats(1:n_train, :)]*w + b;


[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, testDataSet);
