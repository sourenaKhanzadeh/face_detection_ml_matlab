run('../vlfeat-0.9.21/toolbox/vl_setup')
load('pos_neg_feats.mat')
load('pos_neg_valid_feats.mat');

feats = cat(1,x_pos_train,x_neg_train);
labels = cat(1,ones(pos_nImages,1),-1*ones(neg_nImages,1));

lambda = 0.1;
[w,b] = vl_svmtrain(feats',labels',lambda);

fprintf('Classifier performance on train data:\n')
confidences = [x_pos_train; x_neg_train]*w + b;

[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, labels);
