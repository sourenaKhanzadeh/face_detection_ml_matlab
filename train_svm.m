run('../vlfeat-0.9.21/toolbox/vl_setup')
load('pos_neg_feats.mat')
load('pos_neg_valid_feats.mat');

feats = cat(1,x_pos_train,x_neg_train);
labels = cat(1,ones(pos_nImages,1),-1*ones(neg_nImages,1));

valid_feats = cat(1,x_pos_valid,x_neg_valid);
valid_labels = cat(1,ones(valid_pos_nImages,1),-1*ones(valid_neg_nImages,1));


lambda = 0.1;
[w,b] = vl_svmtrain(feats',labels',lambda);

fprintf('Classifier performance on train data:\n')
confidences = [x_pos_train; x_neg_train]*w + b;

[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, labels);


fprintf('Classifier performance on test data:\n')
valid_confidences = [x_pos_valid; x_neg_valid]*w + b;

[tp_rate_valid, fp_rate_valid, tn_rate_valid, fn_rate_valid] =  report_accuracy(valid_confidences, valid_labels);


save('my_svm.mat', 'w', 'b');