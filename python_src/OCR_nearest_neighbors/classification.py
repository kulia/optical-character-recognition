from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def pca_analysis(data, n_components=3):
	pca = PCA(n_components=n_components, svd_solver='full')
	pca_model = pca.fit(data)
	pca_results = pca_model.transform(data)
	
	return pca, pca_results, pca_model

def tsne(pca_results):
	tsne = TSNE(n_components=2, verbose=2, perplexity=2, n_iter=1000, learning_rate=1000, init='pca')
	return tsne.fit_transform(pca_results)


def find_nearest_neighbor(data, target, n_neighbors=2):
	k_nearest_neighbor_model = KNeighborsClassifier(n_neighbors=n_neighbors)
	# print(data.shape, target.shape)
	return k_nearest_neighbor_model.fit(data, target)


def predict(unknown, k_nearest_neighbor_model):
	return k_nearest_neighbor_model.predict(unknown)
