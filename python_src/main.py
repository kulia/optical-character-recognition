import time

import matplotlib.pyplot as plt
import numpy as np

import OCR_support_vector_classifier as ocr_svc
import OCR_nearest_neighbors as orc_nn
import helpers
import helpers.file_handler as file_handler
import preprocessing as pp

from character_detection import character_detection

from sklearn.decomposition import PCA

# from importlib import reload

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def select_samples(train, target, number_of_samples):
	indeces = np.random.permutation(len(train))[:number_of_samples]
	return train[indeces], target[indeces], indeces


def randomize_data_indices(X_data, Y_data):
	return select_samples(X_data, Y_data, len(X_data))


def data_to_train_and_test(X_data, Y_data, ratio=0.2):
	data_X, data_Y, _ = randomize_data_indices(X_data, Y_data)
	
	test_data = data_X[:int(ratio * len(data_X))]
	test_target = data_Y[:int(ratio * len(data_Y))]
	
	train_data = data_X[int(ratio * len(data_X)):]
	train_target = data_Y[int(ratio * len(data_Y)):]
	
	return train_data, train_target, test_data, test_target
	
	
class ImageData():
	def __init__(self, preprocessing=True, show_fig=False, debug=True):
		# Private variables
		self.case_list_ = ['agument_data', 'original', 'pick_and_save_subset', 'load_subset']
		self.case_list_ = self.case_list_[1]

		# Public variables
		self.path = file_handler.Path()
		self.train_data, self.train_target, self.test_data, self.test_target = self.get_data()
		
		self.n_samples = len(self.train_data)
		self.n_pixeles = len(self.train_data[0])
		
		self.show_fig = show_fig
		
		if preprocessing:
			print('ImageData: Start preprocessing')
			self.preprocess_data()
		
	def preprocess_data(self):
		self.train_data = pp.standardized_augmentation(self.train_data, display=False)
		self.test_data  = pp.standardized_augmentation(self.test_data,  display=False)
		
	def get_data(self):
		path_tmp = file_handler.Path('../database/tmp/')
		
		n_sne = 6000

		# For working while dev
		if 'agument_data' in self.case_list_:
			file_handler.generate_chars74k_csv_database()
		if 'original' in self.case_list_ or 'pick_and_save_subset' in self.case_list_:
			data_Y = file_handler.load_target_to_array(self.path.char74k_augmented + 'target.csv')
			data_Y = data_Y[:-1]
			data_X = file_handler.load_image_array_from_csv(self.path.char74k_augmented + 'train.csv')
		
		if 'pick_and_save_subset' in self.case_list_:
			data_X, data_Y, _ = select_samples(data_X, data_Y, n_sne)
			
			file_handler.delete_file(path_tmp.char74k_augmented + 'target.csv')
			file_handler.delete_file(path_tmp.char74k_augmented + 'train.csv')
			
			file_handler.save_array_to_csv(target, path_tmp.char74k_augmented + 'target')
			for line in train:
				file_handler.save_array_to_csv(line, path_tmp.char74k_augmented + 'train')
		if 'load_subset' in self.case_list_:
			data_Y = file_handler.load_target_to_array(path_tmp.char74k_augmented + 'target.csv')
			data_X = file_handler.load_image_array_from_csv(path_tmp.char74k_augmented + 'train.csv')
		
		train_data, train_target, test_data, test_target = data_to_train_and_test(data_X, data_Y, ratio=0.2)
		return train_data, train_target, test_data, test_target


def estimate_error_knn(ImageData, loop=False):
	c = Colors()
	
	error_min = 1
	n_pca_best = 0
	n_neighbors_best = 0
	if loop:
		for n_pca in range(1, 65, 1):
			for n_neighbors in range(1, 10, 1):
				error = orc_nn.classify(ImageData.train_data, ImageData.train_target, ImageData.test_data, ImageData.test_target, n_pca=n_pca,
				                        n_neighbors=n_neighbors)
				
				if error < error_min:
					print(c.OKBLUE, 'Error:', round(100 * error, ndigits=2), '%. n_pca = ', n_pca, 'and n_neighbors = ',
					      n_neighbors, c.ENDC)
					error_min = error
					n_pca_best = n_pca
					n_neighbors_best = n_neighbors
				else:
					print('', 'Error:', round(100 * error, ndigits=2), '%. n_pca = ', n_pca, 'and n_neighbors = ',
					      n_neighbors)
	else:
		n_pca = 60
		n_neighbors = 5
		
		error = orc_nn.classify(ImageData.train_data, ImageData.train_target, ImageData.test_data,
		                        ImageData.test_target, n_pca=n_pca, n_neighbors=n_neighbors)
		print('', 'Error:', round(100 * error, ndigits=2), '%. n_pca = ', n_pca, 'and n_neighbors = ',
		      n_neighbors)
	
	helpers.write_variable_to_latex(round(100 * error_min, ndigits=2), 'error')
	helpers.write_variable_to_latex(round(n_pca_best), 'n_pca')
	helpers.write_variable_to_latex(round(n_neighbors_best), 'n_neighbors')


def main():
	t0 = time.time()
	image_data = ImageData(preprocessing=False)
	print('Loading time: ', time.time() - t0, 's')

	estimate_error_knn(image_data)
	
	# ocr_svc.optimize_svc(image_data)
	#
	# print('Prediction: ', time.time() - t0, 's')
	#
	# pca = PCA(n_components=1)
	# pca_model = pca.fit(image_data.train_data)
	# # character_detection(pca_model)
	plt.show()
				
if __name__ == '__main__':
	main()