import sklearn.svm as svm
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

def optimize_svc(ImageData):
	parameter_candidates = [
		{'C': np.arange(0.01, 2, 0.01), 'kernel': ['linear']}
	]
	
	clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1, verbose=4)
	clf.fit(ImageData.train_data, ImageData.train_target)
	
	print('Best score for train data:', np.round(100 * clf.best_score_), '%')