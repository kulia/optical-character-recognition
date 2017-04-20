import numpy as np
import scipy.signal
import helpers.image as image_helpers
import helpers.file_handler as file_handler

from string import ascii_lowercase
from os import listdir

def augment_chars74k_database():
	path_to_database = '../database/'
	path_to_image_folder = path_to_database + 'chars74k-lite/{}/'
	
	path_to_destination_database = path_to_database + 'chars74k-lite-augmented/'
	
	file_handler.create_dir(path_to_destination_database)
	
	for letter in ascii_lowercase:
		print('Letter: ', letter)
		for index in np.arange(len(listdir(path_to_image_folder.format(letter)))):
			letter_filename = '{}_{}.jpg'.format(letter, index)
			path_to_image = path_to_image_folder.format(letter) + letter_filename
			
			image_array = file_handler.load_image(path_to_image)
			image_array = standardized_augmentation(image_array)
			
			path_to_destination_folder = path_to_destination_database + '{}/'.format(letter)
			path_to_destination_image = path_to_destination_folder + '{}_{}'.format(letter, index)
			
			file_handler.create_dir(path_to_destination_folder)
			file_handler.save_image_to_csv(image_array, path_to_destination_image)
			
			
def standardized_augmentation(image):
	image = normalize(image)
	image = image_helpers.convert_to_sensor_values(image)
	image = tanh(image)
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
	return image_normalized >= threshold