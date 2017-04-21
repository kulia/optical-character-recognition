import numpy as np

import helpers.visualize as image_helpers
import preprocessing.data_augmentation as data_augmentation
import feature_engineering.separation as fe
import matplotlib.pyplot as plt
import helpers.file_handler as file_handler

path_to_image = '../database/chars74k-lite/{}/{}_{}.jpg'
path_to_database = '../database/chars74k-lite-augmented/'

def select_samples(train, target, number_of_samples):
	indeces = np.random.permutation(2000)[:number_of_samples]
	return train[indeces], target[indeces], indeces

if __name__ == '__main__':
	path = file_handler.Path()
	
	data_augmentation.augment_chars74k_database()
	
	target = file_handler.load_target_to_array(path.char74k_augmented + 'target.csv')
	train = file_handler.load_image_array_from_csv(path.char74k_augmented + 'train.csv')
	
	train, target, _ = select_samples(train, target, 100)
	
	path_tmp = file_handler.Path('../database/tmp/')
	file_handler.delete_file(path_tmp.char74k_augmented + 'target.csv')
	file_handler.delete_file(path_tmp.char74k_augmented + 'train.csv')
	file_handler.save_array_to_csv(target, path_tmp.char74k_augmented + 'target')
	for line in train:
		# print(line)
		file_handler.save_array_to_csv(line, path_tmp.char74k_augmented + 'train')

	

	pca, pca_results = fe.pca_analysis(train)

	print(pca.explained_variance_ratio_)
	
	# plt.figure()
	# image_helpers.plot_pca(pca_results, target)
	# plt.show()