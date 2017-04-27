close all, clear all, clc

fontSize = 16;
pathToFig = '../report_src/figures/lsvc_matlab/';
fig = 0;

%% Data Preparation

fprintf('Preparing dataset... \n')
% Go over the folders and find files
Dir   = fullfile('../database/chars74k-lite/');
imSet = imageSet(Dir,   'recursive');

% Plot random images
randImagesN = 8;

fig = newFig(fig);
for i = 1:randImagesN
    subplot(2,(randImagesN/2),i);
    label = randi(length(imSet));
    imshow(imSet(label).ImageLocation{randi(imSet(label).Count)});
    title(imSet(label).Description);
    ax = gca;
    ax.FontSize = fontSize;
end;

saveas(fig, [pathToFig 'random_chars'] , 'epsc')

% Split the dataset into training and test
[trainingSet, testSet] = partition(imSet, [0.8, 0.2], 'randomized');
fprintf('Done! \n')

%% HOG

fprintf('Extracting HOG features... \n')
% Select random image
labl = randi(length(trainingSet));
img = read(trainingSet(labl),(randi(trainingSet(labl).Count)));
% Extract HOG features and HOG visualization
[hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',[2 2]);
[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
[hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);
% Show the original image
fig = newFig(fig); 
subplot(2,3,1:3); imshow(img);
% Visualize the HOG features
subplot(2,3,4);  
plot(vis2x2); 
title({'Cell size = [2 2]'; ['Feature length = ' num2str(length(hog_2x2))]});
subplot(2,3,5);
ax = gca;
ax.FontSize = fontSize;

plot(vis4x4); 
title({'Cell size = [4 4]'; ['Feature length = ' num2str(length(hog_4x4))]});
subplot(2,3,6);
ax = gca;
ax.FontSize = fontSize;

plot(vis8x8); 
title({'Cell size = [8 8]'; ['Feature length = ' num2str(length(hog_8x8))]});
% ax = gca;
% ax.FontSize = 10;

saveas(fig, [pathToFig 'hog_features'] , 'epsc')

% We see that 4-by-4 cell size setting encodes enough spatial information
% to visually identify a digit shape while limiting the number of
% dimensions in the HOG feature vector, which helps speed up training.

cellSize = [4 4];
hogFeatureSize = length(hog_4x4);
fprintf('Done! \n')

%% Train the Multiclass Classifier

fprintf('Training the classifier... \n')
% Loop over the trainingSet and extract HOG features from each image.

trainingFeatures = [];
trainingLabels   = [];
for digit = 1:numel(trainingSet)
    numImages = trainingSet(digit).Count;           
    features  = zeros(numImages, hogFeatureSize, 'single');
    for i = 1:numImages
        img = (read(trainingSet(digit), i));%rgb2gray
        % Apply pre-processing steps
        img = imbinarize(img);
        features(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
    end
    % Use the imageSet Description as the training labels
    labels = repmat(trainingSet(digit).Description, numImages, 1);
    trainingFeatures = [trainingFeatures; features];   %#ok<AGROW>
    trainingLabels   = [trainingLabels;   labels  ];   %#ok<AGROW>
end

%% Train a classifier using the extracted features. 

% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
classifier = fitcecoc(trainingFeatures, trainingLabels);
fprintf('Done! \n')

%% Evaluate the Classifier
fprintf('Evaluating the Classifier... \n')
% Evaluate the classifier using images from the test set, and
% generate a confusion matrix to quantify the classifier accuracy.
% 
% As in the training step, first extract HOG features from the test images.
% These features will be used to make predictions using the trained
% classifier.

% Extract HOG features from the test set. The procedure is similar to what
% was shown earlier and is encapsulated as a helper function for brevity.
[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(testSet, hogFeatureSize, cellSize);

% Make class predictions using the test features.
predictedLabels = predict(classifier, testFeatures);

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

helperDisplayConfusionMatrix(confMat);
fprintf('Done! \n')

%% Predict the letters on test images

fprintf('Analysing test images... \n')

% Test images directory
DirTI   = fullfile('../database/detection-images/');
% Define horizontal and vertical step (pixels) for the sliding window
% Small steps are computationally expensive but yield better accuracy
step = [3,3];
% First or second test image?
imNum = 1;
% Run the Sliding window function
[L] = slidingWindowAndPlot(DirTI, imNum, step, cellSize, classifier, confMat);
% First or second test image?
imNum = 2;
% Run the Sliding window function
[L] = slidingWindowAndPlot(DirTI, imNum, step, cellSize, classifier, confMat);

fprintf('Done! \n')

%% Calculate total error
error = (1-sum(testLabels == predictedLabels)/length(testLabels));
disp(['Total error: ' num2str(100*error) ' %.']);
saveVar(error, 'error_matlab');

%% References
% [1] N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human
% Detection", Proc. IEEE Conf. Computer Vision and Pattern Recognition,
% vol. 1, pp. 886-893, 2005. 
%
% [2] LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998).
% Gradient-based learning applied to document recognition. Proceedings of
% the IEEE, 86, 2278-2324.
%
% [3] Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, A.Y. Ng, Reading
% Digits in Natural Images with Unsupervised Feature Learning NIPS Workshop
% on Deep Learning and Unsupervised Feature Learning 2011.