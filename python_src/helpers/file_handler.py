from PIL import Image
import numpy as np
import pandas as pd

from os import listdir
import os

debug = True


class Path:
	def __init__(self, database='../database/'):
		self.database = database
		self.char74k = database + 'chars74k-lite/'
		self.char74k_augmented = database + 'char74k/'
		create_dir(self.char74k_augmented)
		

def load_image(path_to_image):
	return np.array(Image.open(path_to_image))

def save_image(image, path_to_image):
	image = Image.fromarray(image)
	image.save(path_to_image)


def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)
		
		
def delete_file(path):
	if os.path.isfile(path):
		os.remove(path)


def save_image_to_csv(image_array, path_to_image):
	# print(type(image_array))
	with open(path_to_image + '.csv', 'a+') as f:
		f.write(format_array_to_csv(image_array))
		f.write('\n')


def format_array_to_csv(array):
	return str(array)[1:-1].replace(' ', '')


def load_image_array_from_csv(path_to_image_array):
	image_array = np.array([])
	with open(path_to_image_array, 'r') as f:
		index = 0
		for line in f.readlines():
			image_array = np.append(image_array, [float(x) for x in line.strip('\n').split(',')])
			
			if debug and not index%1000:
				print('Lines of database loaded: ', index)
			index += 1
			
	return image_array