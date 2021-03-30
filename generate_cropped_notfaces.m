% you might want to have as many negative examples as positive examples
n_have = 0;
n_want = numel(dir('cropped_training_images_faces/*.jpg'));


imageDir = 'images_notfaces';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

new_imageDir = 'cropped_training_images_notfaces';
new_validDir = "validate_images";
new_trainDir = "train_images";

if ~isfolder(new_imageDir)
    mkdir(new_imageDir);
end
if ~isfolder(new_trainDir)
    mkdir(new_trainDir);
end
if ~isfolder(new_validDir)
    mkdir(new_validDir);
end

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
    filename = strcat(new_imageDir, "\\img_",string(n_have), ".jpg");
    imwrite(crop, filename);
    
    % splitting data 80-20 train and valid
    n_train = round(n_want *.8, 0);
    n_test = round(n_want *.2, 0);
    
    % save valid and train images to corrosponding folders
    if (n_have <= n_train)
        filename = strcat(new_trainDir, "\\img_",string(n_have), ".jpg");
    else
        filename = strcat(new_validDir, "\\img_",string(n_have-n_train), ".jpg");
    end
    imwrite(crop, filename);
    imwrite(image, filename);
    
    
    fprintf("Saving(%d/%d): %s\n", n_have, n_want, filename);
    
    % go to next frame
    n_have = n_have + 1;
end