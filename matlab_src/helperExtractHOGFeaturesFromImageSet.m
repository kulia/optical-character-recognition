function [featureVector, setLabels] = helperExtractHOGFeaturesFromImageSet(imgSet, hogFeatureSize, cellSize)
% Extract HOG features from an imageSet.

featureVector = [];
setLabels     = [];

% Iterate over an array of imageSets
for idx = 1:numel(imgSet)
    
    numImages = imgSet(idx).Count;
    features  = zeros(numImages, hogFeatureSize, 'single');
    
    % Process each image and extract features
    for j = 1:numImages
        img = (read(imgSet(idx), j));%rgb2gray
        
        % Apply pre-processing steps
        img = imbinarize(img);
        
        features(j, :) = extractHOGFeatures(img,'CellSize',cellSize);
    end
    
    % Use the Description from the imgSet as the labels
    labels = repmat(imgSet(idx).Description, numImages, 1);
    
    featureVector = [featureVector; features];
    setLabels     = [setLabels;     labels];
    
end
