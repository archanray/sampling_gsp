import numpy as np
from scipy.linalg import sqrtm

def is_psd(x):
	return np.all(np.linalg.eigvals(x) > 0)

def laplacian(adjacencyMat):
	"""
	generates a graph Laplacian
	""" 
	L = np.diag(np.sum(adjacencyMat, axis=1)) - adjacencyMat
	# add noise to the diagonal to ensure PSD
	L = L + np.diag(1e-14*np.ones(len(L)))
	return L

def matSqrt(M):
	"""
	compute matrix square root using numpy
	"""
	e, v = np.linalg.eig(M)
	e = np.real(e)
	sqrt_M = v @ np.diag(np.sqrt(e)) @ np.linalg.inv(v)
	return sqrt_M


def signalAnalyzer(f, M):
	"""
	compute soothness quotient
	"""
	L = laplacian(M)
	f = np.expand_dims(f, axis=1)
	smoothnessQuotient = (f.T @ L @ f / (f.T @ f)).squeeze()
	frequencyResponse = np.real(f.T @ sqrtm(L)).squeeze()
	# frequencyResponse = np.real(f.T @ matSqrt(L)).squeeze()

	return smoothnessQuotient, frequencyResponse

