run('../vlfeat-0.9.21/toolbox/vl_setup')

clf = Classifier; 
[w, b, acc] = clf.train();

save('my_svm.mat', 'w', 'b');