close all
clear
run('../vlfeat-0.9.21/toolbox/vl_setup')

clf = Classifier;

pos_imageDir = 'train_pos_images';
[x_pos_train, pos_nImages] = clf.get_feature(pos_imageDir);

valid_pos_imageDir = 'validate_pos_images';
[x_pos_valid, valid_pos_nImages] = clf.get_feature(valid_pos_imageDir);

neg_imageDir = 'train_neg_images';
[x_neg_train, neg_nImages] = clf.get_feature(neg_imageDir);

valid_neg_imageDir = 'validate_neg_images';
[x_neg_valid, valid_neg_nImages] = clf.get_feature(valid_neg_imageDir);


save('pos_neg_feats.mat','x_pos_train','x_neg_train','pos_nImages','neg_nImages')
save('pos_neg_valid_feats.mat','x_pos_valid','x_neg_valid','valid_pos_nImages','valid_neg_nImages')