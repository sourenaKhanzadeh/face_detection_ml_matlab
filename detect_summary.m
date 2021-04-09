%% CPS843/CP8307 Assignment 3 - MATLAB Machine Learning
% Include this file in your submission. Do not modify it, other than to add 
% your name and student number. The grader should be able to run this 
% script to step through the entire assignment. 
% - For multi-part questions, insert your own pauses, where appropriate.
% - For written answer questions, print your answer in the MATLAB terminal.
% Student Name: Sourena Khanzadeh
% Student Number: 500929191
% Student Name: Dylan Donsky
% Student Number: 500700489 

fprintf("for our approach on the detect faces the first thing we did was \n")
fprintf("one we loaded in our svm W and B values we get all the images and initialize \n")
fprintf("the confidences and and bounding boxes to an array of zeros\n")
fprintf("after which we itterate through each image converting them to grayscale\n")
fprintf("and setting a temporary bounding box and confidence holder for each one\n")
fprintf("we then use a scaling factor so for each image we take them at varing sizes\n")
fprintf("starting at around a factor of 1.5 times the image size to make sure we hit\n")
fprintf("the smaller faces in the image, and going down all the way till the min size\n")
fprintf("of the image is no smaller than the size of the bounding box itself\n")
fprintf("for each scale of the image we use a sliding window to go accross the image in\n")
fprintf("a grid like pattern and storing temporary confidences and bounding box values\n")
fprintf("we then take those values, sort them, and then take up to the top 100 values\n")
fprintf("and store their index values into an array, which then goes through it\n")
fprintf("and stores said values into their respective bounding box\n")
fprintf("that has its row and col values adjusted by the scale so it scales back up/down\n")
fprintf("with the image. after going through all the scales and storing all the bboxes and confs\n")
fprintf("we send that data to a maximum suppression algorithm that takes the overlapping boxes \n")
fprintf("and deletes all but the one with the best confidences. we made sure to not run\n")
fprintf("the maximumn suppression progressively as we got each box value as it could scew the\n")
fprintf("results so we ran it only after we got all the box values for each scale the image.\n")
fprintf("after running the non-max suppression algorithm we store the remaing bboxes for the image\n")
fprintf("and move on to the next one if there is any.\n")
fprintf(" \n")
fprintf("we found using a smaller cell size as well as a starting scale of 1.5\n")
fprintf("with a reduction factor of 0.9 per scale tended to give us very good accuracy\n")
fprintf("we also prioritized accuracy over recall with a theta of 1, so we only took confidences\n")
fprintf("that had a value of 1 or greater\n")
fprintf("we wanted to prioritize the accuracy over the most recall\n")
fprintf("and ended up with a ending average precision score of 0.701\n")
fprintf("but we had a high ending accuracy that was slightly over 0.9 .\n")
fprintf("and a recall which was slightly over 0.7\n")
fprintf(" \n")
fprintf("we ended up with a surprising good performance on class.jpg \n")
fprintf("we found that we got pretty much every face with the exception\n")
fprintf("of a few which were sideways and one with longish hair covering the eyes\n")
fprintf("there were surpirsingly few false postives, only 5 in total.\n")