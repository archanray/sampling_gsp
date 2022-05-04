import numpy as np
from src.dataset import get_data
from src.viz import seeGraph as sg
from src.viz import plot_FR as pfr
from src.approximator import sampler
from src.utils import signalAnalyzer as sa
from src.viz import seeError as SE
from tqdm import tqdm

#======================================================================================#
# hyperparameters
dataset = "grid"
# options for the following line
#"nnz", "uniform" 
sampling_mode = "nnz"
# options for the following line
#"smoothed gaussian", "uniform" 
signal_distro = "uniform" 
min_samples = 10
max_samples = 500
steps = 10
num_nodes = 40
trials = 20
p = 0.01

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
## generate the graph and get required parameters
data_matrix, dataset_size, min_samples_r, max_samples_r, node_signals = \
											get_data(dataset, size=num_nodes, \
												p=p, signal=signal_distro)
print("created data matrix")
# get the frequency response
ogSQ, ogFR = sa(node_signals, data_matrix)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# approximation of smoothness quotient
# logger
SQ_error_mean = []
SQ_error_p1 = []
SQ_error_p2 = []
for samples in tqdm(range(min_samples, max_samples, steps)):
	local_SQ_error = []
	for tr in range(trials):
		sampled_adjacency, sampled_signals = \
			sampler(sampling_mode, data_matrix, node_signals, samples=samples)
		SQ, _ = sa(sampled_signals, sampled_adjacency)
		# print(SQ)
		error = np.abs(ogSQ - SQ)
		local_SQ_error.append(error)
	SQ_error_mean.append(np.mean(local_SQ_error))
	SQ_error_p1.append(np.percentile(local_SQ_error, 20))
	SQ_error_p2.append(np.percentile(local_SQ_error, 80))

# visualize the approximation result
if dataset == "grid":
	dataset_size = num_nodes**2
else:
	dataset_size = num_nodes
SE(SQ_error_mean, SQ_error_p1, SQ_error_p2, \
	dataset, sampling_mode, signal_distro, min_samples, \
	max_samples, steps, dataset_size, ogSQ)