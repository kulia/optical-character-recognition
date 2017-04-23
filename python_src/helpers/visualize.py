import matplotlib.pyplot as plt
import numpy as np
	
def show_image(image, colorbar=False):
	image = image.reshape((20, 20))
	cax = plt.imshow(image, cmap=plt.get_cmap('BrBG'))
	
	plt.xticks([])
	plt.yticks([])
	
	if colorbar:
		plt.colorbar(cax)
		
		
def convert_to_sensor_values(image):
	return image**2


def image_matrix_to_array(image):
	return image.flatten().tolist()


def plot_pca(pca_results, target):
	number_of_colors = 26
	colors = [(x * 1.0 / number_of_colors, 0.5*(np.cos(2*np.pi*x/number_of_colors)+1), 0.5*(np.sin(2*np.pi*x/number_of_colors)+1)) for x in range(number_of_colors)]

	i = 0

	for elements in pca_results:
		plt.scatter(elements[0], elements[1], color=colors[chr_to_index(target[i])], marker="$"+target[i].strip('\'')+"$")
		i += 1
		
def str_to_int(string):
	string = string.strip('\'')
	string = ord(string)
	return string

def chr_to_index(char):
	char = str_to_int(char)
	return char-97