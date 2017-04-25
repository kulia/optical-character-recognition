import sklearn.svm as svm
import numpy as np

from sklearn.base import BaseEstimator
from skimage.feature import hog
from skimage import color
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from sklearn.base import BaseEstimator

from sklearn.metrics import accuracy_score

print('OCR_svc reloaded!')

class HogSpecs(BaseEstimator):
	def __init__(self, ImageData, orientations=2, pixels_per_cell=(2,2), cells_per_block=(2,2), visualise=True):
		super(HogSpecs, self).__init__()
		self.orientations = orientations
		self.pixels_per_cell = pixels_per_cell
		self.cells_per_block = cells_per_block
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
		
	gsSVC = GridSearchCV(model, param_grid=parameters, n_jobs=-1, verbose=4)
	gsSVC.fit(ImageData.train_data, ImageData.train_target)
	
	print('Best score for train data:', np.round(100 * gsSVC.best_score_), '%')
	
def set_models(ImageData):
	models = {
		'lsvc' : (
			svm.SVC(),
			[{'C': np.arange(0.01, 2, 0.10), 'kernel': ['linear']}]
		),
		
		'hog_lsvc' : (
			Pipeline([('hog' , HogSpecs(ImageData)) , ('clf', svm.SVC()) ]),
			{
				'hog__orientations' : [2, 4, 5, 10],
				'hog__pixels_per_cell' : [(2, 2), (4, 4), (5, 5)],
				'hog__cells_per_block' : [(2, 2), (4, 4), (5, 5)],
				'clf__C' : [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 5, 10],
			}
		)
	}
	
	return models