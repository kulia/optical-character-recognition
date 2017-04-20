from PIL import Image
import numpy as np
import pandas as pd

import os

def load_image(path_to_image):
	return np.array(Image.open(path_to_image))


def save_image(image, path_to_image):
	image = Image.fromarray(image)
	image.save(path_to_image)


def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)


def save_image_to_csv(image, path_to_image):
	np.savetxt(path_to_image + '.csv', image, delimiter=',')
	

def load_image_from_csv(path_to_image):
	image_array = pd.read_csv(path_to_image, sep=',', header=None)
	return np.array(image_array)