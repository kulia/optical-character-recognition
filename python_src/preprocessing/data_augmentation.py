import numpy as np
import scipy.signal

import helpers.visualize as image_helpers

from string import ascii_lowercase as alphabet
from os import listdir

import os
			
def standardized_augmentation(image):
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


def image_to_bool(image_normalized, threshold=0.2):
	return (image_normalized >= threshold).astype(int)


def select_samples(train, target, number_of_samples):
	indeces = np.random.permutation(len(train))[:number_of_samples]
	return train[indeces], target[indeces], indeces