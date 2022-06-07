import torch
import numpy as np
from src.utils import AdjacencyFromLinearArray as AFLA
from src import utils
from src.viz import seeCombinedError as sCE
from src.approximator import sampler
from tqdm import tqdm
from src.utils import signalAnalyzer as sa
import matplotlib.pyplot as plt
from src.utils import generate_2D_gaussian as G2DG
from copy import copy
import networkx as nx

#################################################################################
"""
following lines are executed to avoid errors due to 
multiple installations
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#################################################################################

############################# compute  smoothness ###############################
def computeSmoothness(y, L):
	if len(y.shape) == 1:
		y = np.expand_dims(y, axis=1)
	return np.squeeze((y.T @ L @ y) / np.linalg.norm(y)**2)
#################################################################################

########################## compute quadratic forms ##############################
def computeQF(y, L):
	if len(y.shape) == 1:
		y = np.expand_dims(y, axis=1)
	return np.squeeze(y.T @ L @ y)
#################################################################################

#################################################################################
# accepatble sets of values for automated code
# random
dataset = "random"
parameters = [["uniform", "gaussian", "bi_gaussian", "ones", "lin_comb"], \
		["QF", "smooth"],\
		["A"]]

# # facebook
# dataset = "facebook"
# parameters = [["uniform", "gaussian", "ones", "lin_comb"], \
# 		["QF", "smooth"],\
# 		["L", "A", "L_via_A"]]

# # erdos and grid
# dataset = "erdos"
# dataset = "grid"
# parameters = [["uniform", "gaussian", "bi_gaussian", "ones", "lin_comb"], \
# 		["QF", "smooth"],\
# 		["L", "A", "L_via_A"]]
#################################################################################

#################################################################################
# # "random", "facebook", "erdos", "grid"
# dataset = "facebook"
# # "uniform", "gaussian", "bi_gaussian", "ones", "lin_comb"
# signal_distro = "uniform"
# # "QF", "smooth"
# compare_as = "QF"
# # "L", "A", "L_via_A"
# compare_for = "L"
#################################################################################

for signal_distro in parameters[0]:
	for compare_as in parameters[1]:
		for compare_for in parameters[2]:
			print(signal_distro, compare_as, compare_for)
			#################################################################################
			# set up
			if signal_distro == "lin_comb":
				k = 5
			# set up torch device
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

			if dataset == "random" and compare_for == "L":
				raise Exception("random matrices can't have Laplacian")

			if dataset == "grid" or dataset == "erdos" or dataset == "random":
				# graph parameters
				num_nodes = 2500
			#################################################################################

			############################### get the graphs ##################################
			if dataset == "grid":
				from networkx import grid_graph
				size = int(np.sqrt(num_nodes))
				g = grid_graph([size, size])

			if dataset == "facebook":
				data_file = "./data/facebook_combined.txt"
				g = nx.read_edgelist(data_file,create_using=nx.DiGraph(), nodetype = int)
				num_nodes = g.number_of_nodes()

			if dataset == "erdos":
				from networkx.generators.random_graphs import erdos_renyi_graph
				g = erdos_renyi_graph(num_nodes, p=0.1)

			if dataset == "random":
				A = np.random.random((num_nodes, num_nodes))
				A = A+A.T # making the matrix symmetric but indefinite
				A = -1*A
				np.fill_diagonal(A, 0)
				A = np.exp(A) # making the matrix PSD more like a RBF 
			 ###################################################################################

			########################### dense adjacency matrix ################################
			if dataset != "random":
				A = nx.adjacency_matrix(g)
				if dataset == "facebook":
					A = A+A.T
				A = A.todense()
				A = np.asarray(A)

			###################################################################################


			############################# construct laplacian #################################
			# eigenvalues are in decreasing order!
			if (dataset != "random" and compare_for == "L") or \
					(dataset != "random" and compare_for == "L_via_A"):
				L = utils.constructLaplacian(A)
				L = L.astype(np.float64)
				L = torch.tensor(L).to(device)
				# eigvals, eigvecs = np.linalg.eig(L)
				eigvals, eigvecs = torch.eig(L, eigenvectors=True)
				eigvals = eigvals[:,0].cpu().detach().numpy()
				eigvecs = eigvecs.cpu().detach().numpy()

			if dataset == "random" or compare_for == "A":
				E = np.real(np.linalg.eigvals(A))
				A = A.astype(np.float64)
				A = torch.tensor(A).to(device)
				eigvals, eigvecs = torch.eig(A, eigenvectors=True)
				eigvals = eigvals[:,0].cpu().detach().numpy()
				eigvecs = eigvecs.cpu().detach().numpy()

			####################################################################################

			####################################################################################
			# construct smooth signal(Heat Kernel)
			if signal_distro == "uniform":
				y = np.random.random_sample(num_nodes)
			if signal_distro == "gaussian":
				mu = 0.0
				sigma = 1.0
				y = np.random.normal(mu, sigma, num_nodes)
			if signal_distro == "bi_gaussian":
				mu = [0.0, 2.0]
				cov = [[1.0, 0.0], [0.0, 2.0]]
				y = G2DG(num_nodes, mu, cov, dataset)
			if signal_distro == "ones":
				y = np.ones(num_nodes)
			if signal_distro == "lin_comb":
				rand_vals = np.random.random(k)
				rand_vals = rand_vals / np.sum(rand_vals)
				y = np.sum(rand_vals*eigvecs[:,0:k],axis=1)
			###################################################################################

			if compare_as == "QF":
				comparator = computeQF
			if compare_as == "smooth":
				comparator = computeSmoothness

			# compute original value
			if dataset != "random" and compare_for == "L":
				original_value = comparator(y, L.cpu().detach().numpy())
			if compare_for == "A":
				original_value = comparator(y, A.cpu().detach().numpy())
			if dataset != "random" and compare_for == "L_via_A":
				original_value = comparator(y, L.cpu().detach().numpy())
			print("original measure:", original_value)

			# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
			"""
			this is the main approximation center! :(
			"""
			# "nnz", "uniform", "row norm", "sparsity sampler_.1f"
			sampling_modes = ["nnz", "uniform"]
			name_adder = "_uniform_nnz_"+compare_as+"_"+compare_for+"_"+signal_distro
			# # name_adder = "_sparse"
			min_samples = 100
			max_samples = 600
			steps=10
			trials = 10
			if compare_for == "A":
				data_matrix = A.cpu().detach().numpy()
			if compare_for == "L":
				data_matrix = L.cpu().detach().numpy()
			if compare_for == "L_via_A":
				data_matrix = A

			node_signals = copy(y)

			ogSQ = original_value

			# # approximation of whatever measure we use here
			# # logger
			sQ_comnined_mean = {}
			sQ_comnined_p1 = {}
			sQ_comnined_p2 = {}
			for sampling_mode in sampling_modes:
				print(sampling_mode)
				SQ_error_mean = []
				SQ_error_p1 = []
				SQ_error_p2 = []
				for samples in tqdm(range(min_samples, max_samples, steps)):
					local_SQ_error = []
					for tr in range(trials):
						sampled_adjacency, sampled_signals = \
							sampler(sampling_mode, data_matrix, node_signals, samples=samples)
						# SQ = sa(sampled_signals, sampled_adjacency)
						if compare_for != "L_via_A":
							SQ = comparator(sampled_signals, sampled_adjacency)
						else:
							sampledL = utils.constructLaplacian(sampled_adjacency)
							SQ = comparator(sampled_signals, sampledL)
						# print(SQ)
						# error = np.abs((ogSQ - SQ) / ogSQ)
						error = np.abs((ogSQ - SQ) / len(data_matrix))
						local_SQ_error.append(error)
					SQ_error_mean.append(np.mean(local_SQ_error))
					SQ_error_p1.append(np.percentile(local_SQ_error, 20))
					SQ_error_p2.append(np.percentile(local_SQ_error, 80))

				sQ_comnined_mean[sampling_mode] = SQ_error_mean
				sQ_comnined_p1[sampling_mode] = SQ_error_p1
				sQ_comnined_p2[sampling_mode] = SQ_error_p2

			# visualize the approximation result
			sCE(sQ_comnined_mean, sQ_comnined_p1, sQ_comnined_p2, \
				dataset, min_samples, \
				max_samples, steps, num_nodes, ogSQ, name_adder)
