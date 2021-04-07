run('../vlfeat-0.9.21/toolbox/vl_setup')


imageDir = 'test_images';

clf = Classifier;
[bboxes, confidences, image_names]  = clf.detect(imageDir); 


% evaluate
label_path = 'test_images_gt.txt';
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections_on_test(bboxes, confidences, image_names, label_path);
