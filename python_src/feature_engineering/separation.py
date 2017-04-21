import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def pca_analysis(data, n_components=3):
	pca = PCA(n_components=n_components)
	pca_results = pca.fit_transform(data)
	return pca, pca_results