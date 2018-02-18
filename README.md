# Optical Character Recognition 
### Approach using _k_-Nearest Neighbors and Support Vector Machines

Semester project in the NTNU course TDT4173 Machine Learning and CBR. 

This project is presenting two approaches to Optical Character Recognition (OCR). The approaches applied are _k_-Nearest Neighbor Classifier and Linear Support Vector Classifier. A general overview of the implementation is shown in the figure below

![](https://github.com/kulia/optical-character-recognition/blob/master/report_src/figures/model.png)

The data is loaded from the Char74k-Lite dataset and diveded, at random, into two databases. 80 % is selected as the training set, and 20 % is the test set. The data is then preprocessed. A model is trained on the training set, before the model is passed to the classifier. The system output is the classifier error.

We implemented the Linear Support Vector Classifier in both Python and Matlab. The Python version was used to optimize detection parameters with grid search and calculate the overall error, while the Matlab version uses inheret high level Matlab features for displaying data, and its superb matrix handling.

The Python program mainly used two packages: Scikit-image and Scikit-learn. Scikit-image is a opensource image processing library, that was manly used for preprocessing. Scikit-learn is a library that can be used for classification, regression, clustering, dimensional reduction, model selection, and also preprocessing. It was heavely applied during this project. 
