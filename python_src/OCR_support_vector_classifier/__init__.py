import sklearn.svm as svm
import numpy as np
from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

print('OCR_svc reloaded!')

class HogSpecs(BaseEstimator):
	def __init__(self, ImageData, orientations=10, pixels_per_cell=(5,5), cells_per_block=(2,2), visualise=True):
		super(HogSpecs, self).__init__()
		self.cells_per_block = cells_per_block
		self.orientations = orientations
		self.pixels_per_cell = pixels_per_cell
		self.visualise = visualise
		
		self.ImageData = ImageData
	
	def fit(self, X, y=None):
		return self
	
	def transform(self, train_data):
		# print('ransform: Before reshape:', )
		train_data = train_data.reshape((train_data.shape[0], 20, 20))
		result = []
		for image in train_data:
			features = hog(
				image,
				orientations=self.orientations,
				pixels_per_cell=self.pixels_per_cell,
				cells_per_block=self.cells_per_block,
			)
			result.append(features)
		return np.array(result)
		
def optimize_svc(ImageData):
	model, parameters = set_models(ImageData)['hog_lsvc']
		
	gsSVC = GridSearchCV(model, param_grid=parameters, n_jobs=-1, verbose=4, cv=2)
	gsSVC.fit(ImageData.train_data, ImageData.train_target)
	
	print('Best score for train data:', np.round(100 * gsSVC.best_score_), '%')
	return gsSVC
	
def set_models(ImageData):
	models = {
		'lsvc' : (
			svm.SVC(),
			[{'C': np.arange(1., 5, 0.30), 'kernel': ['linear']}]
		),
		
		'hog_lsvc' : (
			Pipeline([('hog' , HogSpecs(ImageData)) , ('clf', svm.SVC()) ]),
			{
				'hog__orientations' : [6],
				'hog__pixels_per_cell' : [(4, 4)],
				'hog__cells_per_block' : [(2, 2)],
				'clf__C' : np.arange(1, 2, 1),
				'clf__kernel':['linear'],
			}
		)
	}
	
	return models
