from sklearn.decomposition import PCA
import OCR_nearest_neighbors.classification as kNN
import helpers.file_handler as file_handler
import preprocessing as pp
import numpy as np
import helpers.visualize as image_helpers
import matplotlib.pyplot as plt
# from matplotlib import rc

def classify(train_data, train_target, test_data, test_target, n_pca=49, n_neighbors=4, display=False):
	test_data_original = test_data
	
	train_data = pp.standardized_augmentation(train_data, display=False)
	test_data = pp.standardized_augmentation(test_data, display=False)
	train_data = pp.histogram_of_oriented_gradients(train_data)

	if display:
		plt.figure()
		image_helpers.show_image(data[0])
		plt.savefig(path.figure + 'pp/hog.pdf', format='pdf', dpi=1000)
		plt.draw()
	
	pca = PCA(n_components=n_pca)
	pca_model = pca.fit(train_data)
	pca_train = pca_model.transform(train_data)
	pca_test = pca_model.transform(test_data)
	
	knn_model = kNN.find_nearest_neighbor(pca_train, train_target, n_neighbors=n_neighbors)
	prediction = kNN.predict(pca_test, knn_model)
	
	error = calculate_error(prediction, test_target)
	
	five_Examples(test_data_original, prediction, test_target)
	
	return error

def calculate_error(prediction, target):
	error_counter = 0
	for i in range(len(prediction)):
		if prediction[i] != target[i]:
			# print(prediction[i], target[i])
			error_counter += 1
	
	return error_counter / len(target)


def five_Examples(test_data, prediction, target, path=file_handler.Path()):
	count = 0
	font_size = 30
	
	for i in np.arange(len(test_data)):
		if prediction[i] == target[i]:
			# rc('text', usetex=True)
			image_helpers.show_image(test_data[i])
			plt.xlabel('Prediction: '+prediction[i], fontsize=font_size)
			plt.savefig(path.figure + 'ex/p_{}.pdf'.format(count), format='pdf', dpi=1000)
			plt.draw()
			count += 1
		if count > 4:
			break
	
	count = 0
	for i in np.arange(len(test_data)):
		if prediction[i] != target[i]:
			# rc('text', usetex=True)
			image_helpers.show_image(test_data[i])
			plt.xlabel('Prediction: ' + prediction[i] + '\n Target: ' + target[i] , fontsize=font_size)
			plt.savefig(path.figure + 'ex/n_{}.pdf'.format(count), format='pdf', dpi=1000)
			plt.draw()
			count += 1
		if count > 4:
			break