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
		self.char74k_augmented = database + 'char74k-augmented/'
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
	with open(path_to_image_array, 'r') as f:
		index = 0
		for line in f.readlines():
			if len(image_array) == 0:
				image_array = np.array([float(x) for x in line.strip('\n').split(',')])
			else:
				image_array = np.vstack((image_array, [float(x) for x in line.strip('\n').split(',')]))
			
			if debug and not index%1000:
				print('Lines of database loaded: ', index)
				if index == 2000:break
				
			index += 1
			
	return image_array


def save_target_to_csv(path_to_target, target):
	with open(path_to_target + '.csv', 'a+') as f:
		f.write(target + ',')


def load_target_to_array(path_to_target):
	target = np.array([])
	with open(path_to_target, 'r+') as f:
		for line in f.readlines():
			target = np.append(target, [x for x in line.strip('\n').split(',')[:-1]])
	
	return target


def format_array_to_csv(array):
	# print(str(array)[1:-1].replace(' ', ''))
	array = np.array(array)
	return str(array.tolist())[1:-1].replace(' ', '')