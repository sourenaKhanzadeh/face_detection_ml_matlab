run('../vlfeat-0.9.21/toolbox/vl_setup')

% load w and b
load("my_svm.mat");

imageDir = 'test_images';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

bboxes = zeros(0,4);
ib = 0;
confidences = zeros(0,1);
image_names = cell(0,1);

facts = [0.9 0.8 0.7 0.5 0.6 0.4 0.3];
cellSize = 6;
dim = 36;
for i=1:nImages
    % load and show the image
    image = im2single(imread(sprintf('%s/%s',imageDir,imageList(i).name)));
    imshow(image);
    hold on;
    aboxes = [];
    aconfs = [];
        
    for fact=facts
        im = imresize(image,fact);
        
        % generate a grid of features across the entire image. you may want to 
        % try generating features more densely (i.e., not in a grid)
        feats = vl_hog(im,cellSize);

        % concatenate the features into 6x6 bins, and classify them (as if they
        % represent 36x36-pixel faces)
        [rows,cols,~] = size(feats);    
        confs = zeros(rows,cols);

        for r=1:rows-5
            for c=1:cols-5

            % create feature vector for the current window and classify it using the SVM model, 
            x = feats(r:r+5, c:c+5, :);
            % take dot product between feature vector and w and add b,
            pred = dot(w, x(:)) + b;
            % store the result in the matrix of confidence scores confs(r,c)
            confs(r, c)= pred;
            end
        end

        % get the most confident predictions 
        [~,inds] = sort(confs(:),'descend');
        inds = inds(1:floor((rows*cols)/dim)); % (use a bigger number for better recall)
        for n=1:numel(inds)        
            [row,col] = ind2sub([size(feats,1) size(feats,2)],inds(n));

            bbox = [ col*cellSize./fact ...
                     row*cellSize./fact ...
                    (col+cellSize-1)*cellSize./fact ...
                    (row+cellSize-1)*cellSize./fact];
            conf = confs(row,col);
            % save    
            aboxes = [aboxes; bbox];
            aconfs = [aconfs; conf];

        end
    end
    
    for index = 1:size(aboxes, 1)
        conf = aconfs(index);
        bbox = aboxes(index, :);
        if (conf > 0.9)
            for k=1:size(aconfs, 1)
                pbox = aboxes(k, :);
                bi=[max(bbox(1),pbox(1)) ; max(bbox(2),pbox(2)) ; min(bbox(3),pbox(3)) ; min(bbox(4),pbox(4))];
                iw=bi(3)-bi(1)+1;
                ih=bi(4)-bi(2)+1;
                if iw>0 && ih>0       
                    % compute overlap as area of intersection / area of union
                    ua=(bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1)+...
                       (pbox(3)-pbox(1)+1)*(pbox(4)-pbox(2)+1)-...
                       iw*ih;
                    ov=iw*ih/ua;
                    if ov > 0.01
                        if conf < aconfs(k)
                            bbox = [];
                            conf = [];
                            break
                        end
                    end
                end
            end

            image_name = {imageList(i).name};

            % plot
            if length(bbox) > 1
                plot_rectangle = [bbox(1), bbox(2); ...
                    bbox(1), bbox(4); ...
                    bbox(3), bbox(4); ...
                    bbox(3), bbox(2); ...
                    bbox(1), bbox(2)];
                plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-');
            end
            ib = ib + 1;
            bboxes = [bboxes; bbox];
            confidences = [confidences; conf];
            image_names = [image_names; image_name];
        end
    end

%     pause;
    fprintf('got preds for image %d/%d\n', i,nImages);
end

% evaluate
label_path = 'test_images_gt.txt';
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections_on_test(bboxes, confidences, image_names, label_path);
