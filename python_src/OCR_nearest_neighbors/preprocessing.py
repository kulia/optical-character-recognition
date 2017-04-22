import numpy as np
import scipy.signal

import helpers.visualize as image_helpers
			
def standardized_augmentation(data):
	data = image_helpers.convert_to_sensor_values(data)
	data = image_to_bool(data)
	return data
	
	
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