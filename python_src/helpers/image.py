from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_image(path_to_image):
	return np.array(Image.open(path_to_image))


def save_image(image, path_to_image):
	image = Image.fromarray(image)
	image.save(path_to_image)
	
	
def show_image(image, colorbar=True):
	cax = plt.imshow(image, cmap=plt.get_cmap('BrBG'))
	if colorbar:
		plt.colorbar(cax)
		
		
def convert_to_sensor_values(image):
	return image**2