import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def pca_analysis(data, n_components=3):
	pca = PCA(n_components=n_components, svd_solver='full')
	pca_results = pca.fit_transform(data)
	return pca, pca_results

def tsne(pca_results):
	tsne = TSNE(n_components=2, verbose=2, perplexity=2, n_iter=1000, learning_rate=1000, init='pca')
	return tsne.fit_transform(pca_results)