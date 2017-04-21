import numpy as np

import helpers.image as image_helpers
import preprocessing.data_augmentation as data_augmentation
import matplotlib.pyplot as plt

import helpers.file_handler as file_handler

from sklearn.decomposition import PCA

from string import ascii_lowercase as alphabet

path_to_image = '../database/chars74k-lite/{}/{}_{}.jpg'
path_to_database = '../database/chars74k-lite-augmented/'

def pca_analysis(image, n_components=3):
	image_pca = PCA(n_components=n_components)
	image_pca.fit_transform(image)
	return image_pca.explained_variance_ratio_
	
if __name__ == '__main__':
	path = file_handler.Path()
	
	
	# data_augmentation.augment_chars74k_database()
	
	# a = file_handler.load_image_array_from_csv(path.char74k_augmented + 'train.csv')
	target = file_handler.load_target_to_array(path.char74k_augmented + 'target.csv')
	
	print(target[-10:])
	
	# print(a[0:5])
	
	#
	# alphabet_array = np.array([])
	# target_array = ''
	# for letter in alphabet:
	# 	if 0 == alphabet_array.size:
	# 		alphabet_array = file_handler.load_all_characters(path_to_database, letter)
	# 	else:
	# 		alphabet_array = np.append(alphabet_array, file_handler.load_all_characters(path_to_database, letter))
	#
	# 	target_array += letter
	#
	# print('f', alphabet_array.shape)
	
	
	# print('f', alphabet_array[-1])

	# plt.figure()
	#
	# image_helpers.show_image(alphabet_array[0])
	#
	#
	# plt.figure()
	# image_helpers.show_image(alphabet_array[-1])
			
		# print(alphabet_array)
	
		
	
	# a_array = file_handler.load_all_characters('../database/chars74k-lite-augmented/', 'a')
	# z_array = file_handler.load_all_characters('../database/chars74k-lite-augmented/', 'c')
	#
	# fig = plt.figure()
	# i = 0
	# for a in a_array:
	# 	# print('a: ', pca_analysis(a).shape)
	# 	# plt.scatter(pca_analysis(a)[0], pca_analysis(a)[1], c='r')
	# 	plt.scatter(i, np.amax(pca_analysis(a)), c='r')
	# 	i+=1
	#
	# i=0
	# # fig = plt.figure()
	# for z in z_array:
	# 	# print('z: ', pca_analysis(z))
	# 	# image_helpers.show_image(pca_analysis(z))
	# 	# plt.scatter(pca_analysis(z)[0], pca_analysis(z)[1], c='b')
	# 	plt.scatter(i, np.amax(pca_analysis(z)), c='b')
	# 	# print(np.amax(pca_analysis(z)))
	# 	i += 1
	
	plt.show()