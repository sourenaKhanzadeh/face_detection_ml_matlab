classdef Classifier
    properties
        feat_cellSize = 3;
        feat_n_cell = 12;
        featSize = 31;
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
        
        function debug_features(self, im, feat)
                imhog = vl_hog('render', feat);
                subplot(1,2,1);
                imshow(im);
                subplot(1,2,2);
                imshow(imhog)
                pause;
        end
        
        
    end
end