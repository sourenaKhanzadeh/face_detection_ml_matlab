% you might want to have as many negative examples as positive examples
n_have = 0;
n_want = numel(dir('cropped_training_images_faces/*.jpg'));


imageDir = 'images_notfaces';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

new_imageDir = 'cropped_training_images_notfaces';
mkdir(new_imageDir);

dim = 36;

while n_have < n_want
    % generate random 36x36 crops from the non-face images
    % choose image based on the modulus time
    % to loop over all images
    index = mod(n_have, nImages) + 1;
    image = imread(strcat(imageList(index).folder, "\\",imageList(index).name));
    image = rgb2gray(image);
    
    % randomly crop a part of an image
    win =  randomCropWindow2d(size(image), [dim , dim]);
    crop =  imcrop(image, win);
    % save to directory
    imwrite(crop, strcat(new_imageDir, "\\img_",string(n_have), ".jpg"));
    % go to next frame
    n_have = n_have + 1;
end