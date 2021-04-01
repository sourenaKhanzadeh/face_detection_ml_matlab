close all
clear
run('../vlfeat-0.9.21/toolbox/vl_setup')

pos_imageDir = 'train_pos_images';
pos_imageList = dir(sprintf('%s/*.jpg',pos_imageDir));
pos_nImages = length(pos_imageList);

valid_pos_imageDir = 'validate_pos_images';
valid_pos_imageList = dir(sprintf('%s/*.jpg',valid_pos_imageDir));
valid_pos_nImages = length(valid_pos_imageList);

neg_imageDir = 'train_neg_images';
neg_imageList = dir(sprintf('%s/*.jpg',neg_imageDir));
neg_nImages = length(neg_imageList);

valid_neg_imageDir = 'validate_neg_images';
valid_neg_imageList = dir(sprintf('%s/*.jpg',valid_neg_imageDir));
valid_neg_nImages = length(valid_neg_imageList);

cellSize = 6;
featSize = 31*cellSize^2;

x_pos_train = zeros(pos_nImages,featSize);
for i=1:pos_nImages
    im = im2single(imread(sprintf('%s/%s',pos_imageDir,pos_imageList(i).name)));
    feat = vl_hog(im,cellSize);
    x_pos_train(i,:) = feat(:);
    fprintf('got feat for train pos image %d/%d\n',i,pos_nImages);
%     imhog = vl_hog('render', feat);
%     subplot(1,2,1);
%     imshow(im);
%     subplot(1,2,2);
%     imshow(imhog)
%     pause;
end

x_pos_valid = zeros(valid_pos_nImages,featSize);
for i=1:valid_pos_nImages
    im = im2single(imread(sprintf('%s/%s',valid_pos_imageDir,valid_pos_imageList(i).name)));
    feat = vl_hog(im,cellSize);
    x_pos_valid(i,:) = feat(:);
    fprintf('got feat for valid pos image %d/%d\n',i,valid_pos_nImages);
%     imhog = vl_hog('render', feat);
%     subplot(1,2,1);
%     imshow(im);
%     subplot(1,2,2);
%     imshow(imhog)
%     pause;
end

x_neg_train = zeros(neg_nImages,featSize);
for i=1:neg_nImages
    im = im2single(imread(sprintf('%s/%s',neg_imageDir,neg_imageList(i).name)));
    feat = vl_hog(im,cellSize);
    x_neg_train(i,:) = feat(:);
    fprintf('got feat for neg image %d/%d\n',i,neg_nImages);
%     imhog = vl_hog('render', feat);
%     subplot(1,2,1);
%     imshow(im);
%     subplot(1,2,2);
%     imshow(imhog)
%     pause;
end

x_neg_valid = zeros(valid_neg_nImages,featSize);
for i=1:valid_neg_nImages
    im = im2single(imread(sprintf('%s/%s',valid_neg_imageDir,valid_neg_imageList(i).name)));
    feat = vl_hog(im,cellSize);
    x_neg_valid(i,:) = feat(:);
    fprintf('got feat for valid neg image %d/%d\n',i,valid_neg_nImages);
%     imhog = vl_hog('render', feat);
%     subplot(1,2,1);
%     imshow(im);
%     subplot(1,2,2);
%     imshow(imhog)
%     pause;
end

save('pos_x_neg_train.mat','x_pos_train','x_neg_train','pos_nImages','neg_nImages')
save('pos_x_neg_valid.mat','x_pos_valid','x_neg_valid','valid_pos_nImages','valid_neg_nImages')