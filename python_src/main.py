import numpy as np
import time
import helpers

import matplotlib.pyplot as plt
import helpers.file_handler as file_handler
import OCR_nearest_neighbors as orc_nn
import OCR_nearest_neighbors.preprocessing as pp
import helpers.visualize as image_helpers

from random import random

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
	

def main():
	c = Colors
	path = file_handler.Path()
	path_tmp = file_handler.Path('../database/tmp/')
	
	case_list = ['agument_data', 'original', 'pick_and_save_subset', 'load_subset']
	# case_list = case_list[-1]
	case_list = case_list[1]
	
	n_sne = 6000
	
	t0 = time.time()
	
	# For working while dev
	if 'agument_data' in case_list:
		file_handler.generate_chars74k_csv_database()
	if 'original' in case_list or 'pick_and_save_subset' in case_list:
		data_Y = file_handler.load_target_to_array(path.char74k_augmented + 'target.csv')
		data_Y = data_Y[:-1]
		data_X = file_handler.load_image_array_from_csv(path.char74k_augmented + 'train.csv')
		
	if 'pick_and_save_subset' in case_list:
		data_X, data_Y, _ = select_samples(data_X, data_Y, n_sne)
		
		file_handler.delete_file(path_tmp.char74k_augmented + 'target.csv')
		file_handler.delete_file(path_tmp.char74k_augmented + 'train.csv')
		
		file_handler.save_array_to_csv(target, path_tmp.char74k_augmented + 'target')
		for line in train:
			file_handler.save_array_to_csv(line, path_tmp.char74k_augmented + 'train')
	if 'load_subset' in case_list:
		data_Y = file_handler.load_target_to_array(path_tmp.char74k_augmented + 'target.csv')
		data_X = file_handler.load_image_array_from_csv(path_tmp.char74k_augmented + 'train.csv')
	
	print('Loading time: ', time.time() - t0, 's')
	
	t0 = time.time()
	
	train_data, train_target, test_data, test_target = data_to_train_and_test(data_X, data_Y, ratio=0.2)
	
	error_min = 1
	n_pca_best = 0
	n_neighbors_best = 0
	
	pp.standardized_augmentation(train_data[:2], display=True)
	
	train_data = pp.standardized_augmentation(train_data)
	test_data = pp.standardized_augmentation(test_data)
	
	# for _ in range(5):
	# 	index = int(np.round( len(train_data) * random() ))
	# 	sample_image = train_data[index]
	# 	sample_image = pp.standardized_augmentation(np.array([sample_image]))
	# 	sample_image = sample_image.reshape((20, 20))
	# 	plt.figure()
	# 	image_helpers.show_image(sample_image, colorbar=True)
	# 	plt.title(train_target[index])
	
	for n_pca in range(30, 70, 1):
		for n_neighbors in range(1, 10, 1):
			error = orc_nn.classify(train_data, train_target, test_data, test_target, n_pca=n_pca, n_neighbors=n_neighbors)

			if error < error_min:
				print(c.OKBLUE,'Error ', round(100 * error, ndigits=2), '% when n_pca = ', n_pca, 'and n_neighbors = ', n_neighbors, c.ENDC)
				error_min = error
				n_pca_best = n_pca
				n_neighbors_best = n_neighbors_best
			else:
				print('', 'Error ', round(100 * error, ndigits=2), '% when n_pca = ', n_pca, 'and n_neighbors = ', n_neighbors)
				
	print('Prediction: ', time.time() - t0, 's')
	
	helpers.write_variable_to_latex(round(100*error_min, ndigits=2), 'error')
	helpers.write_variable_to_latex(round(n_pca_best), 'n_pca')
	helpers.write_variable_to_latex(round(n_neighbors_best), 'n_neighbors')
	
	plt.show()
				
if __name__ == '__main__':
	main()