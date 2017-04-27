function [features, varargout] = extractHOGFeatures(I,varargin)
%extractHOGFeatures Extract HOG features.
%  features = extractHOGFeatures(I) extracts HOG features from a truecolor
%  or grayscale image I and returns the features in a 1-by-N vector. These
%  features encode local shape information from regions within an image and
%  can be used for many tasks including classification, detection, and
%  tracking. 
%
%  The HOG feature length, N, is based on the image size and the parameter
%  values listed below. See the <a href="matlab:helpview(fullfile(docroot,'toolbox','vision','vision.map'),'extractHOGFeatures')" >documentation</a> for more information. 
%
%  [features, validPoints] = extractHOGFeatures(I, points) returns HOG
%  features extracted around point locations within I. The function also
%  returns validPoints, which contains the input point locations whose
%  surrounding [CellSize.*BlockSize] region is fully contained within I.
%  The input points can be specified as an M-by-2 matrix of [x y]
%  coordinates, SURFPoints, cornerPoints, MSERRegions, or BRISKPoints. Any
%  scale information associated with the points is ignored. The class of
%  validPoints is the same as the input points.
%
%  [..., visualization] = extractHOGFeatures(I, ...) optionally returns a
%  HOG feature visualization that can be shown using plot(visualization).
%
%  [...] = extractHOGFeatures(..., Name, Value) specifies additional
%  name-value pairs described below:
%
%  'CellSize'     A 2-element vector that specifies the size of a HOG cell
%                 in pixels. Select larger cell sizes to capture large
%                 scale spatial information at the cost of loosing small
%                 scale detail.
%                 
%                 Default: [8 8]
%
%  'BlockSize'    A 2-element vector that specifies the number of cells in
%                 a block. Large block size values reduce the ability to
%                 minimize local illumination changes.
%
%                 Default: [2 2]
%
%  'BlockOverlap' A 2-element vector that specifies the number of
%                 overlapping cells between adjacent blocks. Select an
%                 overlap of at least half the block size to ensure
%                 adequate contrast normalization. Larger overlap values
%                 can capture more information at the cost of increased
%                 feature vector size. This property has no effect when
%                 extracting HOG features around point locations.
% 
%                 Default: ceil(BlockSize/2)
%                  
%  'NumBins'      A positive scalar that specifies the number of bins in
%                 the orientation histograms. Increase this value to encode
%                 finer orientation details.
%                 
%                 Default: 9
%
%  'UseSignedOrientation' A logical scalar. When true, orientation
%                         values are binned into evenly spaced bins
%                         between -180 and 180 degrees. Otherwise, the
%                         orientation values are binned between 0 and
%                         180 where values of theta less than 0 are
%                         placed into theta + 180 bins. Using signed
%                         orientations can help differentiate light to
%                         dark vs. dark to light transitions within
%                         an image region.
%
%                         Default: false
%
% Class Support
% -------------
% The input image I can be uint8, int16, double, single, or logical, and it
% must be real and non-sparse. POINTS can be SURFPoints, cornerPoints,
% MSERRegions, BRISKPoints, int16, uint16, int32, uint32, single, or
% double.
%
%
% Example 1 - Extract HOG features from an image.
% -----------------------------------------------
%
%    I1 = imread('gantrycrane.png');
%    [hog1, visualization] = extractHOGFeatures(I1,'CellSize',[32 32]);
%    subplot(1,2,1);
%    imshow(I1);
%    subplot(1,2,2);
%    plot(visualization);
%
% Example 2 - Extract HOG features around corner points.
% ------------------------------------------------------
%
%    I2 = imread('gantrycrane.png');
%    corners   = detectFASTFeatures(rgb2gray(I2));
%    strongest = selectStrongest(corners, 3);
%    [hog2, validPoints, ptVis] = extractHOGFeatures(I2, strongest);
%    figure;
%    imshow(I2); hold on;
%    plot(ptVis, 'Color','green');
% 
% See also extractFeatures, extractLBPFeatures, detectHarrisFeatures,
% detectFASTFeatures, detectMinEigenFeatures, detectSURFFeatures,
% detectMSERFeatures, detectBRISKFeatures

% Copyright 2012 The MathWorks, Inc.
%
% References
% ----------
% N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human
% Detection", Proc. IEEE Conf. Computer Vision and Pattern Recognition,
% vol. 1, pp. 886-893, 2005. 
%

%#codegen
%#ok<*EMCA>

notCodegen = isempty(coder.target);

[points, isPoints, params, maxargs] = parseInputs(I,varargin{:});

% check number of outputs 
if notCodegen
    nargoutchk(0,maxargs);
else    
    checkNumOutputsForCodegen(nargout, maxargs);
end

if isPoints

    [features, validPoints] = extractHOGFromPoints(I, points, params);
    
    if nargout >= 2
        varargout{1} = validPoints;
    end
    
    if notCodegen
        if nargout == 3
            params.Points = validPoints;
            varargout{2}  = vision.internal.hog.Visualization(features, params);
        end
    end
else   
   
    features = extractHOGFromImage(I, params);  
   
    if notCodegen
        if nargout == 2
            varargout{1} = vision.internal.hog.Visualization(features, params);
        end
    end
end
 
% -------------------------------------------------------------------------
% Extract HOG features from whole image 
% -------------------------------------------------------------------------
function features = extractHOGFromImage(I, params)
[gMag, gDir] = hogGradient(I);

[gaussian, spatial] = computeWeights(params);

features = extractHOG(gMag, gDir, gaussian, spatial, params);

% -------------------------------------------------------------------------
% Extract HOG features from point locations 
% -------------------------------------------------------------------------
function [features, validPoints] = extractHOGFromPoints(I, points, params)

featureClass = coder.internal.const('single');
uintClass    = coder.internal.const('uint32');

blockSizeInPixels = params.CellSize.*params.BlockSize;

% compute weights
[gaussian, spatial] = computeWeights(params);

if ~isnumeric(points)
    xy = points.Location;
else
    xy = points;
end

featureSize = vision.internal.hog.getFeatureSize(params);

halfSize = (single(blockSizeInPixels) - mod(single(blockSizeInPixels),2))./2;

roi = [1 1 blockSizeInPixels]; % [r c height width]

numPoints       = cast(size(xy,1), uintClass);
validPointIdx   = zeros(1, numPoints , uintClass);
validPointCount = zeros(1, uintClass);

features = zeros(numPoints, featureSize, featureClass);
for i = 1:numPoints
    
    % ROI centered at point location
    roi(1:2) = cast(round(xy(i,[2 1])), featureClass) - halfSize;
    
    % only process if ROI is fully contained within the image
    if all(roi(1:2) >= 1) && ...
            roi(1)+roi(3)-1 <= params.ImageSize(1) && ...
            roi(2)+roi(4)-1 <= params.ImageSize(2)
        
        validPointCount = validPointCount + 1;
        
        [gMag, gDir] = hogGradient(I, roi);
               
        hog = extractHOG(gMag, gDir, gaussian, spatial, params);
        
        features(validPointCount,:) = hog(:);
        validPointIdx(validPointCount) = i; % store valid indices
    end
    
end

features = features(1:validPointCount,:);

validPoints = extractValidPoints(points, validPointIdx(1:validPointCount));

% -------------------------------------------------------------------------
% Extract HOG features given gradient magnitudes and directions
% -------------------------------------------------------------------------
function hog = extractHOG(gMag, gDir, gaussianWeights, weights, params)
    
if isempty(coder.target)         
    hog = visionExtractHOGFeatures(gMag, gDir, gaussianWeights, params, weights);                  
else
    featureClass = 'single';
    
    if params.UseSignedOrientation
        % make gDir range from [0 360]
        histRange = single(360);
    else
        % convert to unsigned orientation, range [0 180]
        histRange = single(180);
    end
    
    % range of gDir is [-180 180], convert range to [0 180] or [0 360]
    negDir = gDir < 0;
    gDir(negDir) = histRange + gDir(negDir);
    
    % orientation bin locations for all cells
    binWidth = histRange/cast(params.NumBins, featureClass);
    [x1, b1] = computeLowerHistBin(gDir, binWidth);
    wDir = 1 - (gDir - x1)./binWidth;
    
    blockSizeInPixels = params.CellSize.*params.BlockSize;
    blockStepInPixels = params.CellSize.*(params.BlockSize - params.BlockOverlap);
    
    r = 1:blockSizeInPixels(1);
    c = 1:blockSizeInPixels(2);
    
    nCells  = params.BlockSize;
    nBlocks = vision.internal.hog.getNumBlocksPerWindow(params);
    
    numCellsPerBlock = nCells(1)*nCells(2);
    hog = coder.nullcopy(...
        zeros([params.NumBins*numCellsPerBlock, nBlocks],...
        featureClass));   
    % scan across all blocks
    for j = 1:nBlocks(2)
        
        for i = 1:nBlocks(1)
            
            wz1 = wDir(r,c);
            
            w = trilinearWeights(wz1, weights);
                        
            % apply gaussian weights
            m = gMag(r,c) .* gaussianWeights;
            
            % interpolate magnitudes for binning
            mx1y1z1 = m .* w.x1_y1_z1;
            mx1y1z2 = m .* w.x1_y1_z2;
            mx1y2z1 = m .* w.x1_y2_z1;
            mx1y2z2 = m .* w.x1_y2_z2;
            mx2y1z1 = m .* w.x2_y1_z1;
            mx2y1z2 = m .* w.x2_y1_z2;
            mx2y2z1 = m .* w.x2_y2_z1;
            mx2y2z2 = m .* w.x2_y2_z2;
                                    
            orientationBins = b1(r,c);
            
            % initialize block histogram to zero
            h = zeros(params.NumBins+2, nCells(1)+2, nCells(2)+2, featureClass);
            
            % accumulate interpolated magnitudes into block histogram
            for x = 1:blockSizeInPixels(2)
                cx = weights.cellX(x);
                for y = 1:blockSizeInPixels(1)
                    z  = orientationBins(y,x);
                    cy = weights.cellY(y);
                    
                    h(z,   cy,   cx  ) = h(z,   cy,   cx  ) + mx1y1z1(y,x);
                    h(z+1, cy,   cx  ) = h(z+1, cy,   cx  ) + mx1y1z2(y,x);
                    h(z,   cy+1, cx  ) = h(z,   cy+1, cx  ) + mx1y2z1(y,x);
                    h(z+1, cy+1, cx  ) = h(z+1, cy+1, cx  ) + mx1y2z2(y,x);
                    h(z,   cy,   cx+1) = h(z,   cy,   cx+1) + mx2y1z1(y,x);
                    h(z+1, cy,   cx+1) = h(z+1, cy,   cx+1) + mx2y1z2(y,x);
                    h(z,   cy+1, cx+1) = h(z,   cy+1, cx+1) + mx2y2z1(y,x);
                    h(z+1, cy+1, cx+1) = h(z+1, cy+1, cx+1) + mx2y2z2(y,x);
                end
            end
            
            % wrap orientation bins
            h(2,:,:)     = h(2,:,:)     + h(end,:,:);
            h(end-1,:,:) = h(end-1,:,:) + h(1,:,:);
            
            % only keep valid portion of the block histogram
            h = h(2:end-1,2:end-1,2:end-1);
            
            % normalize and add block to feature vector            
            hog(:,i,j) = normalizeL2Hys(h(:));
            
            r = r + blockStepInPixels(1);
        end
        r = 1:blockSizeInPixels(1);
        c = c + blockStepInPixels(2);
    end
    
    hog = reshape(hog, 1, []);
end

% -------------------------------------------------------------------------
% Normalize vector using L2-Hys
% -------------------------------------------------------------------------
function x = normalizeL2Hys(x)
classToUse = class(x);
x = x./(norm(x,2) + eps(classToUse)); % L2 norm
x(x > 0.2) = 0.2;                     % Clip to 0.2
x = x./(norm(x,2) + eps(classToUse)); % repeat L2 norm

% -------------------------------------------------------------------------
% Compute the interpolation weights for the spatial histogram over cells
% -------------------------------------------------------------------------
function weights = spatialHistWeights(params)
% 2D interpolation weights are computed for 4 points surrounding (x,y)
%
% (x1,y1) o---------o (x2,y1)
%         |         |
%         |  (x,y)  |
%         |         |
% (x1,y2) o---------o (x2,y2)
%
% (x,y) are the pixel centers within a HOG Block
%
% (x1,y1); (x2,y1); (x1,y2); (x2,y2) are cell centers within a block

width  = single(params.BlockSize(2)*params.CellSize(2));
height = single(params.BlockSize(1)*params.CellSize(1));

x = 0.5:1:width;
y = 0.5:1:height;

[x1, cellX1] = computeLowerHistBin(x, params.CellSize(2));
[y1, cellY1] = computeLowerHistBin(y, params.CellSize(1));

wx1 = 1 - (x - x1)./single(params.CellSize(2));
wy1 = 1 - (y - y1)./single(params.CellSize(1));

weights.x1y1 = wy1' * wx1;
weights.x2y1 = wy1' * (1-wx1);
weights.x1y2 = (1-wy1)' * wx1;
weights.x2y2 = (1-wy1)' * (1-wx1);

% also store the cell indices
weights.cellX = cellX1;
weights.cellY = cellY1;

% -------------------------------------------------------------------------
% Compute tri-linear weights
% -------------------------------------------------------------------------
function weights = trilinearWeights(wz1, spatialWeights)

% define struct fields before usage
weights.x1_y1_z1 = coder.nullcopy(wz1);
weights.x1_y1_z2 = coder.nullcopy(wz1);
weights.x2_y1_z1 = coder.nullcopy(wz1);
weights.x2_y1_z2 = coder.nullcopy(wz1);
weights.x1_y2_z1 = coder.nullcopy(wz1);
weights.x1_y2_z2 = coder.nullcopy(wz1);
weights.x2_y2_z1 = coder.nullcopy(wz1);
weights.x2_y2_z2 = coder.nullcopy(wz1);

weights.x1_y1_z1 = wz1 .* spatialWeights.x1y1;
weights.x1_y1_z2 = spatialWeights.x1y1 - weights.x1_y1_z1;
weights.x2_y1_z1 = wz1 .* spatialWeights.x2y1;
weights.x2_y1_z2 = spatialWeights.x2y1 - weights.x2_y1_z1;
weights.x1_y2_z1 = wz1 .* spatialWeights.x1y2;
weights.x1_y2_z2 = spatialWeights.x1y2 - weights.x1_y2_z1;
weights.x2_y2_z1 = wz1 .* spatialWeights.x2y2;
weights.x2_y2_z2 = spatialWeights.x2y2 - weights.x2_y2_z1;

% -------------------------------------------------------------------------
% Compute the closest bin center x1 that is less than or equal to x
% -------------------------------------------------------------------------
function [x1, b1] = computeLowerHistBin(x, binWidth)
% Bin index
width    = single(binWidth);
invWidth = 1./width;
bin      = floor(x.*invWidth - 0.5);

% Bin center x1
x1 = width * (bin + 0.5);

% add 2 to get to 1-based indexing
b1 = int32(bin + 2);

% -------------------------------------------------------------------------
% Compute Gaussian and spatial weights
% -------------------------------------------------------------------------
function [gaussian, spatial] = computeWeights(params)
blockSizeInPixels = params.CellSize.*params.BlockSize;
gaussian = gaussianWeights(blockSizeInPixels);
spatial  = spatialHistWeights(params);

% -------------------------------------------------------------------------
% Gradient computation using central difference filter [-1 0 1]. Gradients
% at the image borders are computed using forward difference. Gradient
% directions are between -180 and 180 degrees measured counterclockwise
% from the positive X axis.
% -------------------------------------------------------------------------
function [gMag, gDir] = hogGradient(img,roi)

if nargin == 1
    roi = [];    
    imsize = size(img);
else
    imsize = roi(3:4);
end

img = single(img);

if ndims(img)==3
    rgbMag = zeros([imsize(1:2) 3], 'like', img);
    rgbDir = zeros([imsize(1:2) 3], 'like', img);
    
    for i = 1:3
        [rgbMag(:,:,i), rgbDir(:,:,i)] = computeGradient(img(:,:,i),roi);
    end
    
    % find max color gradient for each pixel
    [gMag, maxChannelIdx] = max(rgbMag,[],3);
    
    % extract gradient directions from locations with maximum magnitude
    sz = size(rgbMag);
    [rIdx, cIdx] = ndgrid(1:sz(1), 1:sz(2));
    ind  = sub2ind(sz, rIdx(:), cIdx(:), maxChannelIdx(:));
    gDir = reshape(rgbDir(ind), sz(1:2));
else
    [gMag,gDir] = computeGradient(img,roi);
end

% -------------------------------------------------------------------------
% Gradient computation for ROI within an image.
% -------------------------------------------------------------------------
function [gx, gy] = computeGradientROI(img, roi)
img    = single(img);
imsize = size(img);

% roi is [r c height width]
rIdx = roi(1):roi(1)+roi(3)-1;
cIdx = roi(2):roi(2)+roi(4)-1;

imgX = coder.nullcopy(zeros([roi(3)   roi(4)+2], 'like', img)); %#ok<NASGU>
imgY = coder.nullcopy(zeros([roi(3)+2 roi(4)  ], 'like', img)); %#ok<NASGU>

% replicate border pixels if ROI is on the image border. 
if rIdx(1) == 1 || cIdx(1)==1  || rIdx(end) == imsize(1) ...
        || cIdx(end) == imsize(2)
    
    if rIdx(1) == 1
        padTop = img(rIdx(1), cIdx);
    else
        padTop = img(rIdx(1)-1, cIdx);
    end
    
    if rIdx(end) == imsize(1)
        padBottom = img(rIdx(end), cIdx);
    else
        padBottom = img(rIdx(end)+1, cIdx);
    end
    
    if cIdx(1) == 1
        padLeft = img(rIdx, cIdx(1));
    else
        padLeft = img(rIdx, cIdx(1)-1);
    end
    
    if cIdx(end) == imsize(2)
        padRight = img(rIdx, cIdx(end));
    else
        padRight = img(rIdx, cIdx(end)+1);
    end
    
    imgX = [padLeft img(rIdx,cIdx) padRight];
    imgY = [padTop; img(rIdx,cIdx);padBottom];
else  
    imgX = img(rIdx,[cIdx(1)-1 cIdx cIdx(end)+1]);
    imgY = img([rIdx(1)-1 rIdx rIdx(end)+1],cIdx);
end

gx = conv2(imgX, [1 0 -1], 'valid');
gy = conv2(imgY, [1;0;-1], 'valid');

% -------------------------------------------------------------------------
function [gMag,gDir] = computeGradient(img,roi)

if isempty(roi)
    gx = zeros(size(img), 'like', img);
    gy = zeros(size(img), 'like', img);
    
    gx(:,2:end-1) = conv2(img, [1 0 -1], 'valid');
    gy(2:end-1,:) = conv2(img, [1;0;-1], 'valid');
    
    % forward difference on borders
    gx(:,1)   = img(:,2)   - img(:,1);
    gx(:,end) = img(:,end) - img(:,end-1);
    
    gy(1,:)   = img(2,:)   - img(1,:);
    gy(end,:) = img(end,:) - img(end-1,:);
else
    [gx, gy] = computeGradientROI(img, roi);
end

% return magnitude and direction
gMag = hypot(gx,gy);
gDir = atan2d(-gy,gx);

% -------------------------------------------------------------------------
% Compute spatial weights for HOG blocks.
% -------------------------------------------------------------------------
function h = gaussianWeights(blockSize)

sigma = 0.5 * cast(blockSize(1), 'double');

h = fspecial('gaussian', double(blockSize), sigma);

h = cast(h, 'single');

% -------------------------------------------------------------------------
% Extract valid points 
% -------------------------------------------------------------------------
function validPoints = extractValidPoints(points, idx)
if isnumeric(points)
    validPoints = points(idx,:);
else    
    if isempty(coder.target)
        validPoints = points(idx);
    else
        validPoints = getIndexedObj(points, idx);
    end
end

% -------------------------------------------------------------------------
% Input parameter parsing and validation
% -------------------------------------------------------------------------
function [points, isPoints, params, maxargs] = parseInputs(I, varargin)

notCodegen = isempty(coder.target);

sz = size(I);
validateImage(I);

if mod(nargin-1,2) == 1
    isPoints = true;
    points = varargin{1};
    checkPoints(points);
else
    isPoints = false;
    points = ones(0,2);
end

if notCodegen
    p = getInputParser();    
    parse(p, varargin{:});
    userInput = p.Results;          
    validate(userInput);    
    autoOverlap =  ~isempty(regexp([p.UsingDefaults{:} ''],...
        'BlockOverlap','once'));
else
    if isPoints
        [userInput, autoOverlap] = codegenParseInputs(varargin{2:end});
    else
        [userInput, autoOverlap] = codegenParseInputs(varargin{:});    
    end   
    validate(userInput);      
end

params = setParams(userInput,sz);
if autoOverlap
    params.BlockOverlap = getAutoBlockOverlap(params.BlockSize); 
end
crossValidateParams(params);

if isPoints
    maxargs = 3;
    params.WindowSize = params.BlockSize .* params.CellSize;
else
    maxargs = 2;
    params.WindowSize = params.ImageSize;
end

% -------------------------------------------------------------------------
% Input image validation
% -------------------------------------------------------------------------
function validateImage(I)
% validate image
validateattributes(I, {'double','single','int16','uint8','logical'},...
    {'nonempty','real', 'nonsparse','size', [NaN NaN NaN]},...
    'extractHOGFeatures');

sz = size(I);
coder.internal.errorIf(ndims(I)==3 && sz(3) ~= 3,...
                       'vision:dims:imageNot2DorRGB');

coder.internal.errorIf(any(sz(1:2) < 3),...
                       'vision:extractHOGFeatures:imageDimsLT3x3');

% -------------------------------------------------------------------------
% Input parameter parsing for codegen
% -------------------------------------------------------------------------
function [results, usingDefaultBlockOverlap] = codegenParseInputs(varargin)
pvPairs = struct( ...
    'CellSize',     uint32(0), ...
    'BlockSize',    uint32(0), ...
    'BlockOverlap', uint32(0),...
    'NumBins',      uint32(0),...
    'UseSignedOrientation', uint32(0));

popt = struct( ...
    'CaseSensitivity', false, ...
    'StructExpand'   , true, ...
    'PartialMatching', true);

defaults = getParamDefaults();

optarg = eml_parse_parameter_inputs(pvPairs, popt, varargin{:});

usingDefaultBlockOverlap = ~optarg.BlockOverlap;

results.CellSize  = eml_get_parameter_value(optarg.CellSize, ...
    defaults.CellSize, varargin{:});

results.BlockSize = eml_get_parameter_value(optarg.BlockSize, ...
    defaults.BlockSize, varargin{:});

results.BlockOverlap = eml_get_parameter_value(optarg.BlockOverlap, ...
    defaults.BlockOverlap, varargin{:});

results.NumBins = eml_get_parameter_value(optarg.NumBins, ...
    defaults.NumBins, varargin{:});

results.UseSignedOrientation  = eml_get_parameter_value(...
    optarg.UseSignedOrientation, ...
    defaults.UseSignedOrientation, varargin{:});

% -------------------------------------------------------------------------
% Set block overlap based on block size
% -------------------------------------------------------------------------
function autoBlockSize = getAutoBlockOverlap(blockSize)
szGTOne = blockSize > 1;
autoBlockSize = zeros(size(blockSize), 'like', blockSize);
autoBlockSize(szGTOne) = cast(ceil(double(blockSize(szGTOne))./2), 'like', ...
    blockSize);

% -------------------------------------------------------------------------
% Default parameter values
% -------------------------------------------------------------------------
function defaults = getParamDefaults()
intClass = 'int32';
defaults = struct('CellSize'    , cast([8 8],intClass),...
                  'BlockSize'   , cast([2 2],intClass), ...
                  'BlockOverlap', cast([1 1],intClass), ...
                  'NumBins'     , cast( 9   ,intClass), ...
                  'UseSignedOrientation', false,...
                  'ImageSize' , cast([1 1],intClass),...
                  'WindowSize', cast([1 1],intClass));
              
% -------------------------------------------------------------------------
function params = setParams(userInput,sz)
params.CellSize     = reshape(int32(userInput.CellSize), 1, 2);
params.BlockSize    = reshape(int32(userInput.BlockSize), 1 , 2);
params.BlockOverlap = reshape(int32(userInput.BlockOverlap), 1, 2);
params.NumBins      = int32(userInput.NumBins);
params.UseSignedOrientation = logical(userInput.UseSignedOrientation);
params.ImageSize  = int32(sz(1:2));
params.WindowSize = int32([1 1]);

% -------------------------------------------------------------------------
% Input parameter validation 
% -------------------------------------------------------------------------
function validate(params)

checkSize(params.CellSize,  'CellSize');

checkSize(params.BlockSize, 'BlockSize');

checkOverlap(params.BlockOverlap);

checkNumBins(params.NumBins);

checkUsedSigned(params.UseSignedOrientation);

% -------------------------------------------------------------------------
% Cross validation of input values
% -------------------------------------------------------------------------
function crossValidateParams(params)
% Cross validate parameters
                   
coder.internal.errorIf(any(params.BlockOverlap(:) >= params.BlockSize(:)), ...
    'vision:extractHOGFeatures:blockOverlapGEBlockSize');

% -------------------------------------------------------------------------
function parser = getInputParser()
persistent p;
if isempty(p)
    
    defaults = getParamDefaults();
    p = inputParser();
    
    addOptional(p, 'Points', []);    
    addParameter(p, 'CellSize',     defaults.CellSize);
    addParameter(p, 'BlockSize',    defaults.BlockSize);
    addParameter(p, 'BlockOverlap', defaults.BlockOverlap);
    addParameter(p, 'NumBins',      defaults.NumBins);
    addParameter(p, 'UseSignedOrientation', defaults.UseSignedOrientation);    
    
    parser = p;
else
    parser = p;
end

% -------------------------------------------------------------------------
function checkPoints(pts)

if vision.internal.inputValidation.isValidPointObj(pts);    
    vision.internal.inputValidation.checkPoints(pts, mfilename, 'POINTS');   
else
    validateattributes(pts, ...
        {'int16', 'uint16', 'int32', 'uint32', 'single', 'double'}, ...
        {'2d', 'nonsparse', 'real', 'size', [NaN 2]},...
        mfilename, 'POINTS');
end

% -------------------------------------------------------------------------
function checkSize(sz,name)

vision.internal.errorIfNotFixedSize(sz, name);
validateattributes(sz, {'numeric'}, ...
                   {'real','finite','positive','nonsparse','numel',2,'integer'},...
                   'extractHOGFeatures',name); 

% -------------------------------------------------------------------------
function checkOverlap(sz)

vision.internal.errorIfNotFixedSize(sz, 'BlockOverlap');
validateattributes(sz, {'numeric'}, ...
                   {'real','finite','nonnegative','nonsparse','numel',2,'integer'},...
                   'extractHOGFeatures','BlockOverlap');

% -------------------------------------------------------------------------
function checkNumBins(x)

vision.internal.errorIfNotFixedSize(x, 'NumBins');
validateattributes(x, {'numeric'}, ...
                   {'real','positive','scalar','finite','nonsparse','integer'},...
                   'extractHOGFeatures','NumBins');

% -------------------------------------------------------------------------
function checkUsedSigned(isSigned)

vision.internal.errorIfNotFixedSize(isSigned, 'UseSignedOrientation');
validateattributes(isSigned, {'logical','numeric'},...
    {'nonnan', 'scalar', 'real','nonsparse'},...
    'extractHOGFeatures','UseSignedOrientation');

% -------------------------------------------------------------------------
function checkNumOutputsForCodegen(numOut, maxargs)

if ~isempty(coder.target)
    % Do not allow HOG visualization if generating code
    coder.internal.errorIf(numOut > maxargs-1,...
        'vision:extractHOGFeatures:hogVisualizationNotSupported');    
end
