% you might want to have as many negative examples as positive examples
n_have = 0;
n_want = numel(dir('cropped_training_images_faces/*.jpg'));


imageDir = 'images_notfaces';
imageList = dir(sprintf('%s/*.jpg',imageDir));
croppedfaceList = dir("cropped_training_images_faces/*.jpg");
nImages = length(imageList);

new_imageDir = 'cropped_training_images_notfaces';
new_validPosDir = "validate_pos_images";
new_validNegDir = "validate_neg_images";
new_trainPosDir = "train_pos_images";
new_trainNegDir = "train_neg_images";

if ~isfolder(new_imageDir)
   mkdir(new_imageDir);
end
if ~isfolder(new_validPosDir)
   mkdir(new_validPosDir);
end
if ~isfolder(new_validNegDir)
   mkdir(new_validNegDir);
end
if ~isfolder(new_trainPosDir)
   mkdir(new_trainPosDir); 
end
if ~isfolder(new_trainNegDir)
   mkdir(new_trainNegDir); 
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
    
    crop_face = imread(strcat(croppedfaceList(n_have+1).folder, "\\",croppedfaceList(n_have+1).name));
    
    % save to directory
    filename = strcat(new_imageDir, "\\img_",string(n_have), ".jpg");
    imwrite(crop, filename);
    
    % splitting data 80-20 train and valid
    n_train = round(n_want *.8, 0);
    n_test = round(n_want *.2, 0);
    
    % save valid and train images to corrosponding folders
    if (n_have <= n_train)
        filename = strcat(new_trainNegDir, "\\img_",string(n_have), ".jpg");
        filename2 = strcat(new_trainPosDir, "\\", croppedfaceList(n_have+1).name);
    else
        filename = strcat(new_validNegDir, "\\img_",string(n_have-n_train), ".jpg");
        filename2 = strcat(new_validPosDir, "\\",croppedfaceList(n_have+1).name);
    end
    imwrite(crop, filename);
    imwrite(crop_face, filename2);
    
    
    fprintf("Saving(%d/%d): %s\n", n_have, n_want, filename);
    
    % go to next frame
    n_have = n_have + 1;
end