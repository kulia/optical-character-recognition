import matplotlib.pyplot as plt

	
def show_image(image, colorbar=True):
	cax = plt.imshow(image, cmap=plt.get_cmap('BrBG'))
	if colorbar:
		plt.colorbar(cax)
		
		
def convert_to_sensor_values(image):
	return image**2