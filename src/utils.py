import numpy as np
from scipy.linalg import sqrtm

def is_psd(x):
	return np.all(np.linalg.eigvals(x) > 0)

def laplacian(adjacencyMat):
	"""
	generates a graph Laplacian
	""" 
	L = np.diag(np.sum(adjacencyMat, axis=1)) - adjacencyMat
	return L

def signalAnalyzer(f, M):
	"""
	compute soothness quotient
	"""
	L = laplacian(M)
	f = np.expand_dims(f, axis=1)
	smoothnessQuotient = f.T @ L @ f / (f.T @ f).squeeze()
	frequencyResponse = np.real(f.T @ sqrtm(L)).squeeze()

	return smoothnessQuotient[0], frequencyResponse
