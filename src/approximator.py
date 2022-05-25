import numpy as np

def sampler(samplingModes, dataMatrix, signals, samples=100):
	"""
	add details later
	"""
	n = len(dataMatrix)
	list_of_available_indices = list(range(n))

	if samplingModes == "uniform":
		probs = np.ones(n) / float(n)

	if samplingModes == "nnz" or "sparsity sampler" in samplingModes:
		probs = np.count_nonzero(dataMatrix, axis=1, keepdims=False) / \
					np.count_nonzero(dataMatrix, keepdims=False)

	if samplingModes == "row norm":
		probs = np.linalg.norm(dataMatrix, axis=1)**2 / np.linalg.norm(dataMatrix)**2

	sample_indices = np.sort(np.random.choice(list_of_available_indices,\
							size=samples, replace=True, p=probs))

	chosen_p = probs[sample_indices]
	sampled_mat = dataMatrix[sample_indices][:, sample_indices]
	sampled_signals = signals[sample_indices]
	sqrt_chosen_p = np.sqrt(chosen_p*samples)
	D = np.diag(1 / sqrt_chosen_p)
	sampled_mat = D @ sampled_mat @ D

	if "sparsity sampler" in samplingModes:
		multiplier = float(samplingModes.split("_")[1])
		nnzA = np.count_nonzero(dataMatrix)
		original_nnzs = np.count_nonzero(sampled_mat)
		sampled_mat = sampled_mat - np.diag(np.diag(sampled_mat))
		pipj = np.outer(chosen_p, chosen_p)
		mask = (pipj >= 1/(samples*multiplier*nnzA)).astype(int) 
		sampled_mat = sampled_mat*mask
		nnz_sampled_mat = np.count_nonzero(sampled_mat)
		try:
			sampled_mat = sampled_mat / float(original_nnzs)
		except:
			sampled_mat = 0

	return sampled_mat, sampled_signals