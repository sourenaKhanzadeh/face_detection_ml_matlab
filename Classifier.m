classdef Classifier
    properties
        feat_cellSize = 3;
        feat_n_cell = 12;
        featSize = 31;
        lambda = 0.0001
        scale = 1.5;
        og_scale = 0;
        downsample = 0.8;
        cellSize = 6;
        marg = 11;
        dim = 36;
        factor = 10;
        thresh = 0.8;
        ov_factor = 0.01;
        top = 20;
    end
    
    methods
        function obj = Classifier(dict)
            if nargin == 0
                dict = container.Map();
            end
            if ~isempty(dict)
                dict_keys = keys(dict);
                dict_value = values(dict);
                i = 1;
                for key=dict_keys
                    val = dict_value{i};
                    switch key{1}
                        case ['feat_cellSize']
                            obj.feat_cellSize = val;
                        case ['feat_n_cell']
                            obj.feat_n_cell = val;
                        case ['lambda']
                            obj.lambda = val;
                        case ['scale']
                            obj.scale = val;
                        case ['downsample']
                            obj.downsample = val;
                        case ['cellSize']
                            obj.cellSize = val;
                        case ['marg']
                            obj.marg = val;
                        case ['factor']
                            obj.factor = val;
                        case ['thresh']
                            obj.thresh = val;
                        case ['ov_factor']
                            obj.ov_factor = val;
                        case ['top']
                            obj.top = val;
                    end
                    i = i + 1;
                end
            end
            obj.featSize = obj.featSize * obj.feat_n_cell ^ 2;
            obj.og_scale = obj.scale;
            
        end
        
        function [train, n_images] = get_feature(self, image_dir, debug)
            if nargin == 2
                debug = false;
            end
            imageList = dir(sprintf('%s/*.jpg',image_dir));
            n_images = length(imageList);
            
            train = zeros(n_images,self.featSize);
            
            for i=1:n_images
                im = im2single(imread(sprintf('%s/%s',image_dir,imageList(i).name)));
                feat = vl_hog(im,self.feat_cellSize);
                train(i,:) = feat(:);
                fprintf('got feat for train %s %d/%d\n',image_dir, i,n_images);
                if debug
                    self.debug_features(im, feat)
                end
            end
        end
        
        function [w, b, accs] = train(self, load_feats, load_valid_feats)
            load(load_feats);
            load(load_valid_feats);

            feats = cat(1,x_pos_train,x_neg_train);
            labels = cat(1,ones(pos_nImages,1),-1*ones(neg_nImages,1));

            valid_feats = cat(1,x_pos_valid,x_neg_valid);
            valid_labels = cat(1,ones(valid_pos_nImages,1),-1*ones(valid_neg_nImages,1));


            [w,b] = vl_svmtrain(feats',labels',self.lambda);

            fprintf('Classifier performance on train data:\n')
            confidences = [x_pos_train; x_neg_train]*w + b;

            [tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, labels);


            fprintf('Classifier performance on test data:\n')
            valid_confidences = [x_pos_valid; x_neg_valid]*w + b;

            [tp_rate_valid, fp_rate_valid, tn_rate_valid, fn_rate_valid] =  report_accuracy(valid_confidences, valid_labels);
            
            accs = [tp_rate fp_rate tn_rate fn_rate; tp_rate_valid fp_rate_valid tn_rate_valid fn_rate_valid];
        end
        
        function [bboxes, confidences, image_names] = detect(self, image_dir, paused, load_w_b, g)
            % load w and b
            if nargin == 2
                g = false;
                paused = false;
                load("my_svm.mat");
            elseif nargin == 3
                g = false;
                load("my_svm.mat");
            elseif nargin == 4
                g = false;
                load(load_w_b);
            else
                load(load_w_b);
            end
            
            imageList = dir(sprintf('%s/*.jpg',image_dir));
            nImages = length(imageList);
            
            bboxes = zeros(0,4);
            confidences = zeros(0,1);
            image_names = cell(0,1);
            
            for i=1:nImages
                image = im2single(imread(sprintf('%s/%s',image_dir,imageList(i).name)));
                imshow(image);
                hold on;
                aboxes = [];
                aconfs = [];
                [n, m] = size(image); 
                if g
                    image = rgb2gray(image);
                end
                while self.scale* min(n, m) >= self.factor
                    im = imresize(image,self.scale);
                    % generate a grid of features across the entire image. you may want to 
                    % try generating features more densely (i.e., not in a grid)
                    feats = vl_hog(im,self.cellSize);

                    % concatenate the features into 6x6 bins, and classify them (as if they
                    % represent 36x36-pixel faces)
                    [rows,cols,~] = size(feats);    
                    confs = zeros(rows,cols);

                    for r=1:rows-self.marg
                        for c=1:cols-self.marg

                        % create feature vector for the current window and classify it using the SVM model, 
                        x = feats(r:r+self.marg, c:c+self.marg, :);
                        % take dot product between feature vector and w and add b,
                        pred = dot(w, x(:)) + b;
                        % store the result in the matrix of confidence scores confs(r,c)
                        confs(r, c)= pred;
                        end
                    end

                    % get the most confident predictions 
                    [~,inds] = sort(confs(:),'descend');
                    if (rows * cols) < self.top
                        inds = inds(1:floor((rows*cols))); % (use a bigger number for better recall)
                    else
                        inds = inds(1:self.top); % (use a bigger number for better recall)
                    end
                    for n=1:numel(inds)        
                        [row,col] = ind2sub([size(feats,1) size(feats,2)],inds(n));

                        bbox = [ col*self.cellSize/self.scale ...
                                 row*self.cellSize/self.scale ...
                                (col+self.marg)*self.cellSize/self.scale ...
                                (row+self.marg)*self.cellSize/self.scale];
                        conf = confs(row,col);
                        % save    
                        aboxes = [aboxes; bbox];
                        aconfs = [aconfs; conf];

                    end
                    self.scale = self.scale * self.downsample;
                end
                self.scale = self.og_scale;
                
                [bboxes, confidences, image_names] = self.non_max_suppression...
                                                    (aconfs, aboxes, bboxes,...
                                                    confidences, image_names, imageList, i);
                fprintf('got preds for image %d/%d\n', i,nImages);
                
                if paused
                    pause;
                end
            end
            
        end
        
        function [bboxes, confidences, image_names] = non_max_suppression(self, aconfs, aboxes,...
                                                        bboxes, confidences,image_names,  imageList, i)
            for index = 1:size(aboxes, 1)
                conf = aconfs(index);
                bbox = aboxes(index, :);
                if (conf > self.thresh)
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
                            if ov > self.ov_factor
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

                        bboxes = [bboxes; bbox];
                        confidences = [confidences; conf];
                        image_names = [image_names; image_name];
                    end
                end
            end
        end
        
        function debug_features(~, im, feat)
                imhog = vl_hog('render', feat);
                subplot(1,2,1);
                imshow(im);
                subplot(1,2,2);
                imshow(imhog)
                pause;
        end
        
        
    end
end