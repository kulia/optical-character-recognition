from PIL import Image
import numpy as np

import os

import pandas as pd
# df=pd.read_csv('myfile.csv', sep=',',header=None)
# df.values
# array([[ 1. ,  2. ,  3. ],
#        [ 4. ,  5.5,  6. ]])

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