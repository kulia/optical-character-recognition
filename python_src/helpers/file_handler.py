import os

import numpy as np
import helpers.visualize as image_helpers

from string import ascii_lowercase as alphabet
from PIL import Image
from os import listdir
from time import time

debug = True


class Path:
	def __init__(self, database='../database/'):
		self.database = database
		self.char74k = database + 'chars74k-lite/'
		self.char74k_augmented = database + 'char74k-augmented/'
		self.figure = '../report_src/figures/'
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


def save_array_to_csv(array, path):
	with open(path + '.csv', 'a+') as f:
		f.write(format_array_to_csv(array))
		f.write('\n')


def load_image_array_from_csv(path_to_image_array):
	image_array = np.array([])
	t_end = time()
	
	with open(path_to_image_array, 'r') as f:
		index = 0
		for line in f.readlines():
			t0 = time()
			if len(image_array) == 0:
				image_array = np.array([float(x) for x in line.strip('\n').strip('\'').split(',')])
			else:
				image_array = np.vstack((image_array, [float(x) for x in line.strip('\n').strip('\'').split(',')]))
			
			t1 = time()
			
			if debug and not index%1000:
				print('Loading database: ', int(100 * index/7112), ' %', 1000*(t1-t0), 1000*(t0-t_end))
				
			index += 1
			
			t_end = time()
			
	return image_array


def save_target_to_csv(path_to_target, target):
	with open(path_to_target + '.csv', 'a+') as f:
		f.write(target + ',')


def load_target_to_array(path_to_target):
	target = np.array([])
	with open(path_to_target, 'r+') as f:
		for line in f.readlines():
			target = np.append(target, [x for x in line.strip('\n').split(',')])
			if target[-1] == ',':
				target = target[:-1]
	
	return target


def format_array_to_csv(array):
	array = np.array(array)
	return str(array.tolist())[1:-1].replace(' ', '')


def generate_chars74k_csv_database():
	train_database_name = 'train'
	target_database_name = 'target'
	
	path = Path()
	
	delete_file(path.char74k_augmented + train_database_name + '.csv')
	delete_file(path.char74k_augmented + target_database_name + '.csv')
	
	for letter in alphabet:
		print('Letter: ', letter)
		for index in np.arange(len(listdir(path.char74k + '{}/'.format(letter)))):
			letter_filename = '{}/{}_{}.jpg'.format(letter, letter, index)
			path_to_image = path.char74k + letter_filename
			
			image = load_image(path_to_image)
			image = normalize(image)
			
			image_array = image_helpers.image_matrix_to_array(image)
			
			save_array_to_csv(image_array, path.char74k_augmented + train_database_name)
			save_target_to_csv(path.char74k_augmented + target_database_name, letter)


def normalize(data):
	image_min = np.nanmin(data)
	image_max = np.nanmax(data)
	
	return (data - image_min) / (image_max - image_min)