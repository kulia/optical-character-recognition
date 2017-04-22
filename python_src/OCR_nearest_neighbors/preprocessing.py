import numpy as np
import scipy.signal
from skimage.feature import hog

import helpers.visualize as image_helpers
							
def standardized_augmentation(data):
	# data = image_helpers.convert_to_sensor_values(data)
	# data = histogram_of_oriented_gradients(data)
	return data
	
def histogram_of_oriented_gradients(data_images):
	images = np.array([])
	for data_image in data_images:
		if len(data_image) == 400:
			data_image = data_image.reshape((20, 20))
		else:
			print('Wrong size', data_image.shape)
		
		fd, data_image = hog(data_image, orientations=8, pixels_per_cell=(16, 16),
		                    cells_per_block=(1, 1), visualise=True)
		
		images = np.append(images, data_image)
		
	return np.array(images).flatten()
	
	
def normalize(data):
	image_min = np.nanmin(data)
	image_max = np.nanmax(data)
	
	return (data - image_min) / (image_max - image_min)


def sigmoid(data_normalized):
	data_normalized = normalize_abs_1(data_normalized)
	data_normalized = np.divide(1.0, 1.0 + np.exp(-data_normalized))
	return normalize(data_normalized)


def tanh(data_normalized):
	data_normalized = normalize_abs_1(data_normalized)
	data_normalized = np.tanh(data_normalized)
	return  normalize(data_normalized)


def normalize_abs_1(data_normalized):
	return 2 * data_normalized - 1


def median_filter(data_normalized, filter_width=3):
	return scipy.signal.medfilt(data_normalized, filter_width)


def image_to_bool(data_normalized, threshold=0.2):
	return (data_normalized >= threshold).astype(int)