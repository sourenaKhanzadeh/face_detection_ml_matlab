run('../vlfeat-0.9.21/toolbox/vl_setup')


imageDir = 'test_images';

params = containers.Map(...
    {'feat_cellSize', 'feat_n_cell', 'lambda', 'scale',...
    'downsample', 'cellSize', 'marg', 'factor', 'thresh', 'ov_factor',...
    'top'},...
    {3, 12, 0.0001, 1.5, 0.8, 6, 11, 10, 0.8, 0.01, 20});

clf = Classifier(params);
[bboxes, confidences, image_names]  = clf.detect(imageDir); 


% evaluate
label_path = 'test_images_gt.txt';
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections_on_test(bboxes, confidences, image_names, label_path);
