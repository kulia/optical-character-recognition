import helpers.file_handler as file_handler
import helpers.visualize as visualize
import matplotlib.pyplot as plt
import numpy as np


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    
def character_detection(pca_model):
	c = Colors()
	
	path_detect_1 = '../database/detection-images/detection-1.jpg'
	path_detect_2 = '../database/detection-images/detection-2.jpg'
	
	image_1 = file_handler.load_image(path_detect_1)
	image_1 = file_handler.normalize(image_1)
	
	image_1_bool = image_1 < 0.99999
	
	
	window_size = 20
	non_overlap = 2
	
	print('Shape of image: ', image_1.shape)
	
	plt.figure()

	visualize.show_image(image_1, colorbar=True)
	
	for i in np.arange(0, len(image_1_bool)-window_size):
		for j in np.arange(0, len(image_1_bool[i])-window_size):
			window = image_1_bool[i:i+window_size, j:j+window_size]
			window_sum = np.sum(window)
			
			pca_val = pca_model.transform(np.array(window).flatten())
			
			if pca_val > -5:
				plt.scatter(j + 10, i + 10, c='r')
				print('Window num ', i, j, '- Sum of cell: ', window_sum)

			if window_sum > 190:
				print('Window num ', i, j, '- Sum of cell: ', window_sum)
				print(c.OKBLUE, 'PCA: ', pca_model.transform(np.array(window).flatten()), c.ENDC)
			else:
				print(c.ENDC, 'PCA: ', pca_model.transform(np.array(window).flatten()))