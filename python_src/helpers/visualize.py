import matplotlib.pyplot as plt
import numpy as np
	
def show_image(image, colorbar=True):
	cax = plt.imshow(image, cmap=plt.get_cmap('BrBG'))
	if colorbar:
		plt.colorbar(cax)
		
		
def convert_to_sensor_values(image):
	return image**2


def image_matrix_to_array(image):
	return image.flatten().tolist()


def plot_pca(pca_results, target):
	number_of_colors = 26
	colors = [(x * 1.0 / number_of_colors, 0.5, 0.5) for x in range(number_of_colors)]
	
	i = 0
	
	for elements in pca_results:
		plt.scatter(elements[0], elements[1], color=colors[ord(str(target[i])) - 97])
		i += 1
	
	for i in range(len(colors)):
		plt.scatter(0, 0, label=chr(i + 97), c=colors[i], alpha=1)
		plt.legend()