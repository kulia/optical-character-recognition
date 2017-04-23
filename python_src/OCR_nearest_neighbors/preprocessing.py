import numpy as np
import scipy.signal
from skimage.feature import hog

from skimage import restoration
from skimage.filters import threshold_otsu
import skimage.morphology

import helpers.visualize as image_helpers
							
debug = True
							
def standardized_augmentation(data):
	data = denoise(data)
	data = image_to_bool(data)
	data = closing(data)
	data = image_helpers.convert_to_sensor_values(data)
	data = histogram_of_oriented_gradients(data)
	return data
	
	
def closing(data):
	data_out = np.array([])
	for data_sample in data:
		
		data_sample = data_sample.reshape((20, 20))
		sample_bool = skimage.morphology.closing(data_sample, skimage.morphology.square(2))
		sample_bool = np.array(sample_bool).flatten()
		
		if len(data_out) == 0:
			data_out = sample_bool
		else:
			data_out = np.vstack((data_out, sample_bool))
	
	return np.array(data_out)


def image_to_bool(data_normalized):
	data_bool = np.array([])
	for data_sample in data_normalized:
		
		data_sample = data_sample.reshape((20, 20))
		sample_bool = data_sample <= threshold_otsu(data_sample)
		sample_bool = np.array(sample_bool).flatten()
		
		if len(data_bool) == 0:
			data_bool = sample_bool
		else:
			print(data_bool.shape, sample_bool.shape)
			data_bool = np.vstack((data_bool, sample_bool))
	
	return np.array(data_bool)
	
	
def denoise(data):
	denoised_data = np.array([])
	for data_sample in data:
		data_sample = data_sample.reshape((20, 20))
		denoised_sample = restoration.denoise_tv_chambolle(data_sample)
		denoised_sample = denoised_sample.flatten()
		
		if len(denoised_data) == 0:
			denoised_data = denoised_sample
		else:
			denoised_data = np.vstack((denoised_data, denoised_sample))
			
	return denoised_data
	
	
def histogram_of_oriented_gradients(data_images):
	images = np.array([])
	index = 0
	for data_image in data_images:
		if len(data_image) == 400:
			data_image = data_image.reshape((20, 20))
		else:
			print('Wrong size', data_image.shape)
		
		fd, data_image = hog(data_image, orientations=10, pixels_per_cell=(5, 5),
		                    cells_per_block=(2, 2), visualise=True)
		
		data_image = np.array(data_image).flatten()
		
		if len(images) == 0:
			images = data_image
		else:
			images = np.vstack((images, data_image))
			
		index += 1
		if debug and not index%(len(data_images)/10):
			print('HOG is', 100 * index/len(data_images), '% finished.')
		
	return np.array(images)
	
	
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
