from sklearn.decomposition import PCA
import OCR_nearest_neighbors.classification as kNN

def classify(train_data, train_target, test_data, test_target, n_pca=49, n_neighbors=4):
	pca = PCA(n_components=n_pca)
	pca_model = pca.fit(train_data)
	pca_train = pca_model.transform(train_data)
	pca_test = pca_model.transform(test_data)
	
	knn_model = kNN.find_nearest_neighbor(pca_train, train_target, n_neighbors=n_neighbors)
	prediction = kNN.predict(pca_test, knn_model)
	
	error = calculate_error(prediction, test_target)
	
	return error
	
def calculate_error(prediction, target):
	error_counter = 0
	for i in range(len(prediction)):
		if prediction[i] != target[i]:
			# print(prediction[i], target[i])
			error_counter += 1
	
	return error_counter / len(target)