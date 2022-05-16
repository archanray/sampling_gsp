import numpy as np
from src.dataset import get_data
from src.viz import seeGraph as sg
from src.viz import plot_FR as pf
from src.viz import plot_FR_v_eigs as pfe
from src.approximator import sampler
from src.utils import signalAnalyzer as sa

#======================================================================================#
# hyperparameters
dataset = "grid"
sampling_mode = "uniform"
samples = 200
num_nodes = 40
p = 0.01

#======================================================================================#
# OG matrix
data_matrix, dataset_size, min_samples, max_samples, node_signals = \
											get_data(dataset, size=num_nodes, \
												p=p, signal = "smoothed gaussian")
# see the graph
sg(data_matrix, node_signals, dataset, sampling_mode, name="ogGraph_visualize", plot_node_size=10)
# get the frequency response
SQ, FR = sa(node_signals, data_matrix)
# plot frequency response
# pf(np.sort(FR), SQ, dataset, sampling_mode, name="ogFR_visualize")
pf(FR, SQ, dataset, sampling_mode, name="ogFR_visualize")
eigvals = np.real(np.linalg.eigvalsh(data_matrix))
pfe(FR, eigvals, SQ, dataset, sampling_mode, name="ogFRE_visualize")

#============== ========================================================================#
# sampled matrix
sampled_adjacency, sampled_signals = \
			sampler(sampling_mode, data_matrix, node_signals, samples=samples)
# see the graph
sg(sampled_adjacency, sampled_signals, dataset, sampling_mode, \
	name="sampledGraph_visualize", plot_node_size=10)
# get the frequency response
SQ, FR = sa(sampled_signals, sampled_adjacency)
# plot frequency response
# pf(np.sort(FR), SQ, dataset, sampling_mode, name="sampledFR_visualize")
pf(FR, SQ, dataset, sampling_mode, name="sampledFR_visualize")
eigvals = np.real(np.linalg.eigvalsh(sampled_adjacency))
pfe(FR, eigvals, SQ, dataset, sampling_mode, name="sampledFRE_visualize")
