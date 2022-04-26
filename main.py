import numpy as np
from src.dataset import get_data
from src.viz import seeGraph as sg
from src.viz import plot_FR as pfr
from src.approximator import sampler
from src.utils import signalAnalyzer as sa

#======================================================================================#
# hyperparameters
dataset = "barabasi"
sampling_mode = "uniform"
samples = 200
num_nodes = 1000
p = 0.01

#======================================================================================#
# OG matrix
data_matrix, dataset_size, min_samples, max_samples, node_signals = \
											get_data(dataset, size=num_nodes, p=p)
# see the graph
sg(data_matrix, node_signals, dataset, sampling_mode, name="ogGraph_visualize", plot_node_size=10)
# get the frequency response
SQ, FR = sa(node_signals, data_matrix)
# plot frequency response
pfr(np.sort(FR), SQ, dataset, sampling_mode, name="ogFR_visualize")

#======================================================================================#
# sampled matrix
sampled_adjacency, sampled_signals = \
			sampler(sampling_mode, data_matrix, node_signals, samples=samples)
# see the graph
sg(sampled_adjacency, sampled_signals, dataset, sampling_mode, \
	name="sampledGraph_visualize", plot_node_size=10)
# get the frequency response
SQ, FR = sa(sampled_signals, sampled_adjacency)
# plot frequency response
pfr(np.sort(FR), SQ, dataset, sampling_mode, name="sampledFR_visualize")