run('../vlfeat-0.9.21/toolbox/vl_setup')

clf = Classifier; 
[w, b, acc] = clf.train('pos_neg_feats.mat', 'pos_neg_valid_feats.mat');

save('my_svm.mat', 'w', 'b');