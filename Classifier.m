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
        %{
        Classifier
        =====================
        params: 
            dict : container.Map
        Dynamically change Classifier properties
        and initialize static parameters
        %}
        function obj = Classifier(dict)
            % if no arg is passed then set
            % dict to default Map type
            if nargin == 0
                dict = containers.Map();
            end
            if ~isempty(dict)
                % take the dictionary
                % partition to keys => values
                dict_keys = keys(dict);
                dict_value = values(dict);
                i = 1;
                % check every key
                for key=dict_keys
                    % get corrosponding value
                    val = dict_value{i};
                    % see if key matches any of the following
                    % change Classifier property
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
            % save default scale value for
            % later use, initialize feat size for more use
            obj.featSize = obj.featSize * obj.feat_n_cell ^ 2;
            obj.og_scale = obj.scale;
            
        end
        
        %{
        Classifier.get_feature
        =====================
        predifinitions:
            let Z = integers
            let n and m in Z where
                n = length of total images
                m = Classifier.featSize
        params:
            self: Classifier
            image_dir: string
                image directory path
            debug: boolean
                set true for debug mode
        returns:
            train: nxm array
                features
            n_images: int
                number of images in the image_dir
        get the features after splitting the 
        dataset, get validation and training features
        from the right directory, for validation pass
        validation dir and for training pass training dir
        %}
        function [train, n_images] = get_feature(self, image_dir, debug)
            % if debug not specify set it to false
            if nargin == 2
                debug = false;
            end
            % get all the images and take its lenght
            imageList = dir(sprintf('%s/*.jpg',image_dir));
            n_images = length(imageList);
            
            % initialize feature set
            train = zeros(n_images,self.featSize);
            for i=1:n_images
                % for every image take its HOG and save in train
                % as a feature
                im = im2single(imread(sprintf('%s/%s',image_dir,imageList(i).name)));
                feat = vl_hog(im,self.feat_cellSize);
                train(i,:) = feat(:);
                fprintf('got feat for train %s %d/%d\n',image_dir, i,n_images);
                if debug
                    % check the image of features
                    % go to Classifier.debug_features for more info
                    self.debug_features(im, feat)
                end
            end
        end
        
        %{
        Classifier.train
        =====================
        predifinitions:
            let Z = integers
            let n in Z
        params: 
            self: Classifier
            load_feats: string
                string to the path of the features
            load_valid_feats: string
                string to the path of validation features
        returns:
            w: nx1 array
            b: float
            accs: 2x4 array
                test and train accuracy details
        train the features and make a SVM model by returning
        the w and b as f(x) = wx + b and get accuracy detail 
        of the training and testing models
        %}
        function [w, b, accs] = train(self, load_feats, load_valid_feats)
            % load features
            load(load_feats);
            load(load_valid_feats);
            
            % get training and its specific label
            feats = cat(1,x_pos_train,x_neg_train);
            labels = cat(1,ones(pos_nImages,1),-1*ones(neg_nImages,1));
            
            % get valaidation and its specific label
            valid_feats = cat(1,x_pos_valid,x_neg_valid);
            valid_labels = cat(1,ones(valid_pos_nImages,1),-1*ones(valid_neg_nImages,1));

            % train feature and get w and b
            [w,b] = vl_svmtrain(feats',labels',self.lambda);

            fprintf('Classifier performance on train data:\n')
            confidences = [x_pos_train; x_neg_train]*w + b;

            [tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, labels);


            fprintf('Classifier performance on test data:\n')
            valid_confidences = [x_pos_valid; x_neg_valid]*w + b;

            [tp_rate_valid, fp_rate_valid, tn_rate_valid, fn_rate_valid] =  report_accuracy(valid_confidences, valid_labels);
            
            accs = [tp_rate fp_rate tn_rate fn_rate; tp_rate_valid fp_rate_valid tn_rate_valid fn_rate_valid];
        end
        
        %{
        Classifier.detect
        =====================
        predifinitions:
            let Z = integers
            let n in Z
        params:
            self: Classifier
            image_dir: string
                path to image directory
            paused: boolean
                set true if you want to pause to see each image
            load_w_b: string
                path to load the w and b file as *.mat
            g: boolean
                set true if image needs to be grayscaled
        returns:
            bboxes: nx4 array
                detected faces as bounding boxes
            confidences: nx1
                detected faces confidences
            image_names: nx1
                detected faces image names
        detect the faces of each images in the specified directory
        save bounding boxes and its confidences for further use
        to see every image set @parameter: 'paused' to true
        %}
        function [bboxes, confidences, image_names] = detect(self, image_dir, paused, load_w_b, g)
            % init g and paused to false
            % and load my_svm.mat by default
            % if no argument specified
            if nargin == 2
                g = false;
                paused = false;
                load_w_b = "my_svm.mat"; 
            elseif nargin == 3
                g = false;
                load_w_b = "my_svm.mat";
            elseif nargin == 4
                g = false;
            end
            % load w and b
            load(load_w_b);
            
            % get images in the directory
            imageList = dir(sprintf('%s/*.jpg',image_dir));
            nImages = length(imageList);
            
            % initialize 
            bboxes = zeros(0,4);
            confidences = zeros(0,1);
            image_names = cell(0,1);
            
            for i=1:nImages
                % for every image
                image = im2single(imread(sprintf('%s/%s',image_dir,imageList(i).name)));
                imshow(image);
                hold on;
                
                % initialize a temorary boundin box and
                % confidence holder for every image
                aboxes = [];
                aconfs = [];
                [n, m] = size(image);
                % convert image to grayscale if @param: 'g' is
                % set to true
                if g
                    image = rgb2gray(image);
                end
                % multi scale the image until a certain factor
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
                        % set the bounding box
                        bbox = [ col*self.cellSize/self.scale ...
                                 row*self.cellSize/self.scale ...
                                (col+self.marg)*self.cellSize/self.scale ...
                                (row+self.marg)*self.cellSize/self.scale];
                        conf = confs(row,col);
                        % temorary save    
                        aboxes = [aboxes; bbox];
                        aconfs = [aconfs; conf];

                    end
                    % down scale
                    self.scale = self.scale * self.downsample;
                end
                % set scale to its original for the use of next image
                self.scale = self.og_scale;
                
                % run non_maximum_suppression on the image
                [bboxes, confidences, image_names] = self.non_max_suppression...
                                                    (aconfs, aboxes, bboxes,...
                                                    confidences, image_names, imageList, i);
                fprintf('got preds for image %d/%d\n', i,nImages);
                
                % pause to see image if @parameter: 'paused' is set to true
                if paused
                    pause;
                end
            end
            
        end
        
        %{
        Classifier.non_max_suppression
        =====================
        predefinitions:
            let Z = integers
            let n and m and p in Z where m <= p <= n 
            let k in Z where k != n or k != m or k == n or k == m
        parameters:
            self: Classifier
            aconfs: nx1 array
                temporary confidences of the detected boxes
            aboxes: nx4 array
                temporary bounding box of each detection
            bboxes: mx4 array
                real bounding boxes for detected faces
            confidences: mx1 array
                real confidences for each detected faces
            image_names: nx1 array
                image name as strings
            imageList: kx1 struct
                list of image files
        returns:
            bboxes: px4 array
                real bounding boxes for detected faces
            confidences: px1
                real confidences for each detected faces
            image_names: px1
                image names as string 
        implement the non suppression max given the confidences and
        bounding boxes to minimize false posetives
        %}
        function [bboxes, confidences, image_names] = non_max_suppression(self, aconfs, aboxes,...
                                                        bboxes, confidences,image_names,  imageList, i)
            for index = 1:size(aboxes, 1)
                % check all boxes
                conf = aconfs(index);
                bbox = aboxes(index, :);
                % keep only the boxes with higher accuracies than
                % the given threshold
                if (conf > self.thresh)
                    for k=1:size(aconfs, 1)
                        % check if any of the boxes overlaps
                        pbox = aboxes(k, :);
                        bi=[max(bbox(1),pbox(1)) ; max(bbox(2),pbox(2)) ; min(bbox(3),pbox(3)) ; min(bbox(4),pbox(4))];
                        iw=bi(3)-bi(1)+1;
                        ih=bi(4)-bi(2)+1;
                        % the box overlaps
                        if iw>0 && ih>0       
                            % compute overlap as area of intersection / area of union
                            ua=(bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1)+...
                               (pbox(3)-pbox(1)+1)*(pbox(4)-pbox(2)+1)-...
                               iw*ih;
                            ov=iw*ih/ua;
                            % choose the one with higher confidence
                            if ov > self.ov_factor
                                if conf < aconfs(k)
                                    bbox = [];
                                    conf = [];
                                    break
                                end
                            end
                        end
                    end
                    % save the image name
                    image_name = {imageList(i).name};

                    % plot
                    if length(bbox) > 1
                        plot_rectangle = [bbox(1), bbox(2); ...
                            bbox(1), bbox(4); ...
                            bbox(3), bbox(4); ...
                            bbox(3), bbox(2); ...
                            bbox(1), bbox(2)];
                        plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-');
                        
                        % save the faces detected
                        bboxes = [bboxes; bbox];
                        confidences = [confidences; conf];
                        image_names = [image_names; image_name];
                    end
                end
            end
        end
        
        %{
        Classifier.debug_features
        =====================
        predifinitions:
            let Z = integers
            let n and m in Z
            let q and s in Z
        ~: ignore self
        parameters:
            im: nxm array
                image
            feat: qxsx31 array
                HOG feature
        debug and see each HOG feature
        see how they look like
        %}
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