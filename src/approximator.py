import numpy as np

def sampler(samplingModes, dataMatrix, signals, samples=100):
	"""
	add details later
	"""
	n = len(dataMatrix)
	list_of_available_indices = list(range(n))

	if samplingModes == "uniform":
		probs = np.ones(n) / float(n)

	if samplingModes == "nnz":
		probs = np.count_nonzero(dataMatrix, axis=1, keepdims=False) / \
					np.count_nonzero(dataMatrix, keepdims=False)

	sample_indices = np.sort(np.random.choice(list_of_available_indices,\
							size=samples, replace=True, p=probs))

	sampled_signals = signals[sample_indices]

	return dataMatrix[sample_indices][:, sample_indices], sampled_signals