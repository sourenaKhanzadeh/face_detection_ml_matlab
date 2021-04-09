run('../vlfeat-0.9.21/toolbox/vl_setup')


imageDir = 'test_images';

lambda = 0.00005;
scale = 1.5;
downsample = 0.9;
cellSize = 6;
marg = 11;
dim = 36;                                                                                                                                               
thresh = 1;

params = containers.Map(...
    {'feat_cellSize', 'feat_n_cell', 'lambda', 'scale',...
    'downsample', 'cellSize', 'marg', 'factor', 'thresh', 'ov_factor',...
    'top'},...
    {3, 12, lambda, scale, downsample, cellSize, marg, dim, thresh, 0.01, 200});


clf = Classifier(params);
% to pause change clf.detect(imageDir, true)
[bboxes, confidences, image_names]  = clf.detect(imageDir); 


% evaluate
label_path = 'test_images_gt.txt';
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections_on_test(bboxes, confidences, image_names, label_path);
