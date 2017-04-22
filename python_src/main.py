import numpy as np

import helpers.visualize as image_helpers
import preprocessing.data_augmentation as data_augmentation
import feature_engineering.separation as fe
import matplotlib.pyplot as plt
import helpers.file_handler as file_handler
from sklearn.decomposition import PCA

path_to_image = '../database/chars74k-lite/{}/{}_{}.jpg'
path_to_database = '../database/chars74k-lite-augmented/'

from sklearn.manifold import TSNE

import time

if __name__ == '__main__':
	path = file_handler.Path()
	path_tmp = file_handler.Path('../database/tmp/')
	
	n_sne = 1000
	format_database = True
	
	case_list = ['agument_data', 'original', 'pick_and_save_subset', 'load_subset']
	case_list = case_list[-1]
	
	# For working while dev
	if 'agument_data' in case_list:
		data_augmentation.generate_chars74k_csv_database()
	if 'original' in case_list or 'pick_and_save_subset' in case_list:
		target = file_handler.load_target_to_array(path.char74k_augmented + 'target.csv')
		train = file_handler.load_image_array_from_csv(path.char74k_augmented + 'train.csv')
	if 'pick_and_save_subset' in case_list:
		train, target, _ = data_augmentation.select_samples(train, target, n_sne)
		
		file_handler.delete_file(path_tmp.char74k_augmented + 'target.csv')
		file_handler.delete_file(path_tmp.char74k_augmented + 'train.csv')
		
		file_handler.save_array_to_csv(target, path_tmp.char74k_augmented + 'target')
		for line in train:
			file_handler.save_array_to_csv(line, path_tmp.char74k_augmented + 'train')
	if 'load_subset' in case_list:
		target = file_handler.load_target_to_array(path_tmp.char74k_augmented + 'target.csv')
		train = file_handler.load_image_array_from_csv(path_tmp.char74k_augmented + 'train.csv')
	
	t0 = time.time()
	
	test = train[0:200]
	test_target = target[0:200]
	
	train = train[200:]
	target = target[200:]
	
	for j in range(1, 300):
		pca = PCA(n_components=20)
		pca_model = pca.fit(train)
		pca_train = pca_model.transform(train)
		pca_test = pca_model.transform(test)
		
		knn_model = fe.find_nearest_neighbor(pca_train, target, n_neighbors=j)
		prediction = fe.predict(pca_test, knn_model)
		
		error_counter = 0
		for i in range(len(prediction)):
			if prediction[i] != test_target[i]:
				error_counter += 1
		
		print(j, 'Error: ', 100 * error_counter / len(test_target), '%')
		
	print('time: ', time.time() - t0, 's')