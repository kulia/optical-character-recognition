function [L] = slidingWindowAndPlot(DirTI, imNum, step, cellSize, classifier, confMat)

imTestSet = imageSet(DirTI);
figure; imshow(imTestSet.ImageLocation{imNum});
I = imread(imTestSet.ImageLocation{imNum});
% Apply sliding window approach and identify features

numTestImages = 1;
coordXY = [];

for i = 1:step(1):(size(I,1) - 20)
    for j = 1:step(2):(size(I,2) - 20)
        imT = I(i:i+19,j:j+19);
        featuresTest(numTestImages,:) = extractHOGFeatures(imT,'CellSize',cellSize);
        coordXY(numTestImages,:) = [i,j];
        numTestImages = numTestImages + 1;
    end;
end;

% Find the likelyhood of each letter to be present in each image
[predictTestLabel,scores] = predict(classifier, featuresTest);

maxScoresIm = max(scores);
count = 0;

for i = 1:length(maxScoresIm)
    if maxScoresIm(i) < 0
        continue;
    else
        count = count+1;
        LableFound(count) = char(i+96);
        C = coordXY(find(scores(:,i) == maxScoresIm(i)),:);
        coord4Lable(count,:) = C(randi(size(C,1)),:);
        position(count,:) = [(coord4Lable(count, 1)) (coord4Lable(count,2)) 20 20];
    end;
end;

figure;

for i = 1:count
    subplot(1,count,i);
    imshow(I(position(i,1):position(i,1)+20,position(i,2):position(i,2)+20));
    cmPos = double(LableFound(i))-96;
    confid(i) = confMat(cmPos,cmPos)/sum(confMat(cmPos,:))*100;
    title(LableFound(i));
    xlabel({round(confid(i)) '%'});
end;

L = LableFound;

end
