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
	
	n_sne = 5000
	format_database = True
	
	if format_database:
		# data_augmentation.augment_chars74k_database()

		target = file_handler.load_target_to_array(path.char74k_augmented + 'target.csv')
		train = file_handler.load_image_array_from_csv(path.char74k_augmented + 'train.csv')

		# train, target, _ = data_augmentation.select_samples(train, target, 4000)

	# 	file_handler.delete_file(path_tmp.char74k_augmented + 'target.csv')
	# 	file_handler.delete_file(path_tmp.char74k_augmented + 'train.csv')
	# 	file_handler.save_array_to_csv(target, path_tmp.char74k_augmented + 'target')
	# 	for line in train:
	# 		file_handler.save_array_to_csv(line, path_tmp.char74k_augmented + 'train')
	#
	# target = file_handler.load_target_to_array(path_tmp.char74k_augmented + 'target.csv')
	# train = file_handler.load_image_array_from_csv(path_tmp.char74k_augmented + 'train.csv')

	print('Number of samples: ', len(train))
	
	for i in range(1):
		i = 9
		t0 = time.time()
		print('--- iteration {} ---'.format(i))
		pca, pca_results = fe.pca_analysis(train[1:], n_components=50)
		
		tsne = TSNE(n_components=2, verbose=2, perplexity=2, n_iter=1000, learning_rate=1000, init='pca')
		tsne_results = tsne.fit_transform(pca_results)
		
		plt.figure()
		image_helpers.plot_pca(tsne_results, target)
		plt.draw()
		plt.title('t-SNE: {}'.format(100*(i+1)))
		print('Timelapce: ', time.time()-t0)

	print('Finished')
	plt.show()