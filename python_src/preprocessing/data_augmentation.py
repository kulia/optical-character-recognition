import numpy as np
import scipy.signal
 
import helpers.file_handler as file_handler
import helpers.visualize as image_helpers

from string import ascii_lowercase as alphabet
from os import listdir

import os

def augment_chars74k_database():
	# path_to_database = '../database/'
	# path_to_image_folder = path_to_database + 'chars74k-lite/{}/'
	#
	train_database_name = 'train'
	target_database_name = 'target'
	
	
	path = file_handler.Path()
	
	file_handler.delete_file(path.char74k_augmented + train_database_name + '.csv')
	file_handler.delete_file(path.char74k_augmented + target_database_name + '.csv')
	
	for letter in alphabet:
		print('Letter: ', letter)
		for index in np.arange(len(listdir(path.char74k + '{}/'.format(letter)))):
			letter_filename = '{}/{}_{}.jpg'.format(letter, letter, index)
			path_to_image = path.char74k + letter_filename
			
			image = file_handler.load_image(path_to_image)
			image = standardized_augmentation(image)
			
			image_array = image_helpers.image_matrix_to_array(image)
			
			file_handler.save_array_to_csv(image_array, path.char74k_augmented + train_database_name)
			file_handler.save_target_to_csv(path.char74k_augmented + target_database_name, letter)
			
def standardized_augmentation(image):
	image = normalize(image)
	image = image_helpers.convert_to_sensor_values(image)
	image = image_to_bool(image)
	return image
	
	
def normalize(image):
	image_min = np.nanmin(image)
	image_max = np.nanmax(image)
	
	return (image - image_min) / (image_max - image_min)


def sigmoid(image_normalized):
	image_normalized = normalize_abs_1(image_normalized)
	image_normalized = np.divide(1.0, 1.0 + np.exp(-image_normalized))
	return normalize(image_normalized)


def tanh(image_normalized):
	image_normalized = normalize_abs_1(image_normalized)
	image_normalized = np.tanh(image_normalized)
	return  normalize(image_normalized)


def normalize_abs_1(image_normalized):
	return 2*image_normalized - 1


def median_filter(image_normalized, filter_width=3):
	return scipy.signal.medfilt(image_normalized, filter_width)


def image_to_bool(image_normalized, threshold=0.5):
	return (image_normalized >= threshold).astype(int)


def select_samples(train, target, number_of_samples):
	indeces = np.random.permutation(len(train))[:number_of_samples]
	return train[indeces], target[indeces], indeces