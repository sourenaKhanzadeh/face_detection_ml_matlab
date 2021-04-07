classdef Classifier
    properties
        feat_cellSize = 3;
        feat_n_cell = 12;
        featSize = 31;
        lambda = 0.0001
    end
    
    methods
        function obj = Classifier()
            obj.featSize = obj.featSize * obj.feat_n_cell ^ 2;
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
        
        function [w, b, accs] = train(self)
            load('pos_neg_feats.mat');
            load('pos_neg_valid_feats.mat');

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