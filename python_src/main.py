import numpy as np
import time

import helpers.file_handler as file_handler
import OCR_nearest_neighbors as orc_nn

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
	path = file_handler.Path()
	path_tmp = file_handler.Path('../database/tmp/')
	
	case_list = ['agument_data', 'original', 'pick_and_save_subset', 'load_subset']
	# case_list = case_list[-1]
	case_list = case_list[1:2]
	
	n_sne = 1500
	
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

	for n_pca in range(1, 100, 5):
		for n_neighbors in range(1, 10):
			error = orc_nn.classify(train_data, train_target, test_data, test_target, n_pca=n_pca, n_neighbors=n_neighbors)

			if error < error_min:
				print('Error ', int(100 * error), '% when n_pca = ', n_pca, 'and n_neighbors = ', n_neighbors)
				error_min = error
				
	print('Prediction: ', time.time() - t0, 's')
	
if __name__ == '__main__':
	main()