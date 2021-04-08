run('../vlfeat-0.9.21/toolbox/vl_setup')


params = containers.Map(...
    {'feat_cellSize', 'feat_n_cell',...
    'lambda', 'scale',...
    'downsample', 'cellSize',...
    'marg', 'factor',...
    'thresh', 'ov_factor',...
    'top'},...
    {3, 12, 0.0001, 1.3, 0.9, 3, 11, 20, 1, 0.01, 200});


clf = Classifier(params);

% pos_imageDir = 'train_pos_images';
% [x_pos_train, pos_nImages] = clf.get_feature(pos_imageDir);
% 
% valid_pos_imageDir = 'validate_pos_images';
% [x_pos_valid, valid_pos_nImages] = clf.get_feature(valid_pos_imageDir);
% 
% neg_imageDir = 'train_neg_images';
% [x_neg_train, neg_nImages] = clf.get_feature(neg_imageDir);
% 
% valid_neg_imageDir = 'validate_neg_images';
% [x_neg_valid, valid_neg_nImages] = clf.get_feature(valid_neg_imageDir);
% 
% 
% save('class_feats.mat','x_pos_train','x_neg_train','pos_nImages','neg_nImages')
% save('class_valid_feats.mat','x_pos_valid','x_neg_valid','valid_pos_nImages','valid_neg_nImages')
% 
% [w, b, acc] = clf.train('class_feats.mat', 'class_valid_feats.mat');
% 
% save('class_svm.mat', 'w', 'b');

imageDir = 'final_test';

[bboxes, confidences, image_names]  = clf.detect(imageDir, false, 'class_svm.mat', true);
