import numpy as np

import helpers.visualize as image_helpers
import preprocessing.data_augmentation as data_augmentation
import feature_engineering.separation as fe
import matplotlib.pyplot as plt
import helpers.file_handler as file_handler

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
	# case_list = case_list[0]
	
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
	print('--- iteration {} ---'.format(i))
	pca, pca_results = fe.pca_analysis(train[1:], n_components=50)
	print('time: ', time.time() - t0)
	
	print('Finished')
	plt.show()