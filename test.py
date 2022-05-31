# import numpy as np
# from timeit import default_timer as timer
# from src.utils import scipyMatMul as sMM
# from src.utils import computeSquaredDistance as cSD
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal
# from src.utils import multivariate_gaussian, sample_positions
# import networkx as nx

#################################################################################
"""
following lines are executed to avoid errors due to 
multiple installations
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#################################################################################

# loops = 100

# y = np.random.random_sample(5000)
# y = np.expand_dims(y, axis=1)
# start = timer()
# for i in tqdm(range(loops)):
# 	dists = sMM(y)
# end = timer()

# print("scipy time:", (end-start)/loops)

# start = timer()
# for i in tqdm(range(loops)):
# 	dists = cSD(y,y)
# end = timer()

# print("my time:", (end-start)/loops)

###############################################################
# # 3D gaussian signals
# nodes = 1600
# N = int(np.sqrt(nodes))
# # get the parameters
# mu = np.array([0., 1.])
# sigma = np.array([[ 1. , 0], [0,  1.]])

# # get the positions
# pos = sample_positions(N, mu, sigma)

# # get the signals
# signals = multivariate_gaussian(pos, mu, sigma)

# # plt.plot(signals)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(pos[:,:,0], pos[:,:,1], signals, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# plt.show()

##############################################################
# # the following lines are good, but not quite :/ 
# # as it only samples 2D points and not signals
# mu = [0,0]
# cov = [[1,0], [0,1]]
# x = np.random.multivariate_normal(mu, cov, 500)
# plt.plot(x[:,0], x[:,1], 'x')
# plt.axis('equal')
# plt.show()


##########################################################################
# # test proportion of sparsity zeroed out

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

def computeSmoothness(y, L):
	return np.squeeze((y.T @ L @ y) / np.linalg.norm(y)**2)

or_sp = []
new_sp = []
for i in tqdm(range(20)):
	signal_distro = "bi_gaussian"
	dataset = "optimal"
	# set up torch device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# graph parameters
	num_nodes = 1600
	num_edges = int(0.5*(num_nodes-1)*num_nodes)


	# weight initialization (uniform random sample in range [0,1) )
	edgeWgt = np.random.random_sample(num_edges)
	# edgeWgt = torch.rand(num_edges)

	# construct Laplacian
	A = AFLA(edgeWgt, num_nodes)
	L = utils.constructLaplacian(A)
	L = torch.tensor(L).to(device)
	# eigvals, eigvecs = np.linalg.eig(L)
	eigvals, eigvecs = torch.eig(L, eigenvectors=True)
	eigvals = eigvals[:,0].cpu().detach().numpy()
	eigvecs = eigvecs.cpu().detach().numpy()

	# construct smooth signal(Heat Kernel)
	if signal_distro == "uniform":
		x = np.random.random_sample(num_nodes)
	if signal_distro == "gaussian":
		mu = 0.0
		sigma = 1.0
		x = np.random.normal(mu, sigma, num_nodes)
	if signal_distro == "bi_gaussian":
		mu = [0.0, 2.0]
		cov = [[1.0, 0.0], [0.0, 2.0]]
		x = G2DG(num_nodes, mu, cov, dataset)

	Vgs = np.exp(-2*np.diag(eigvals))
	# y is the signal
	y = (eigvecs @ Vgs @ eigvecs.T) @ np.expand_dims(x, axis=1)
	# the above line should produce a (num_nodes, 1) shaped vector
	# see the signal
	if signal_distro != "bi_gaussian":
		plt.plot(np.squeeze(y))
		directory = "figures/"+dataset
		plt.savefig(directory+"/signal_"+signal_distro+".pdf")

	# compute smoothness
	original_smoothness = computeSmoothness(y, L.cpu().detach().numpy())
	# print("original smoothness:", original_smoothness)

	# print("sparsity of original matrix:", np.count_nonzero(A) / (len(A)**2))
	or_sp.append(np.count_nonzero(A) / (len(A)**2))

	#################################################################################
	# eq: 
	# F(wij) = wij*(yi-yj)**2 + beta*wij**2 - alpha**T * (log(Di)+log(Dj)) + (wij>=0)
	# the last term is modified to: gamma * 1/(1+exp(-wij))
	alpha = 0.7
	beta = 0.11
	gamma = 0.5
	LR = 0.01
	LR_decay = 0.01
	maxIter = 5000
	tolerance = 1e-8

	Z = utils.scipyMatMul(y,y)
	Ld = np.diag(L.cpu().detach().numpy())
	iLd = 1/Ld
	S = np.ones((len(Ld), len(Ld))) * iLd 
	# all elements in i^th column of S is same as ith element of iLd
	S = S+S.T

	edgeWgtNew = utils.gradientDescentS(edgeWgt, Z, S, alpha, \
						beta, gamma, LR, LR_decay, maxIter, tolerance, num_nodes, device)


	# construct new laplacian
	AF = AFLA(edgeWgtNew, num_nodes)
	LF = utils.constructLaplacian(AF)
	# compute new smoothness
	new_smoothness = computeSmoothness(y, LF)

	# print("optimal smoothness:", new_smoothness)

	# print("***Achieved graph with smoothness minimized for the given signal!***")

	# print("sparsity of optimized matrix:", np.count_nonzero(AF) / (len(AF)**2))
	new_sp.append(np.count_nonzero(AF) / (len(AF)**2))

print("original:", np.mean(or_sp), np.std(or_sp))
print("optimal:", np.mean(new_sp), np.std(new_sp))

#################################################################################
# check number of edges that needs to be altered after sampling to achieve small
# smootheness quotient

# import torch
# import numpy as np
# from src.utils import AdjacencyFromLinearArray as AFLA
# from src import utils
# from src.viz import seeCombinedError as sCE
# from src.approximator import sampler
# from tqdm import tqdm
# from src.utils import signalAnalyzer as sa
# import matplotlib.pyplot as plt
# from src.utils import generate_2D_gaussian as G2DG
# from src.utils import get_triu_entries as GTE

# #################################################################################
# """
# following lines are executed to avoid errors due to 
# multiple installations
# """
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# #################################################################################

# def computeSmoothness(y, L):
# 	return np.squeeze((y.T @ L @ y) / np.linalg.norm(y)**2)

# signal_distro = "bi_gaussian"
# dataset = "optimal"
# # set up torch device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # graph parameters
# num_nodes = 1600
# num_edges = int(0.5*(num_nodes-1)*num_nodes)


# # weight initialization (uniform random sample in range [0,1) )
# edgeWgt = np.random.random_sample(num_edges)
# # edgeWgt = torch.rand(num_edges)

# # construct Laplacian
# A = AFLA(edgeWgt, num_nodes)
# L = utils.constructLaplacian(A)
# L = torch.tensor(L).to(device)
# # eigvals, eigvecs = np.linalg.eig(L)
# eigvals, eigvecs = torch.eig(L, eigenvectors=True)
# eigvals = eigvals[:,0].cpu().detach().numpy()
# eigvecs = eigvecs.cpu().detach().numpy()

# # construct smooth signal(Heat Kernel)
# if signal_distro == "uniform":
# 	x = np.random.random_sample(num_nodes)
# if signal_distro == "gaussian":
# 	mu = 0.0
# 	sigma = 1.0
# 	x = np.random.normal(mu, sigma, num_nodes)
# if signal_distro == "bi_gaussian":
# 	mu = [0.0, 2.0]
# 	cov = [[1.0, 0.0], [0.0, 2.0]]
# 	x = G2DG(num_nodes, mu, cov)

# Vgs = np.exp(-2*np.diag(eigvals))           #<----- suss?
# # y is the signal
# y = (eigvecs @ Vgs @ eigvecs.T) @ np.expand_dims(x, axis=1)
# # the above line should produce a (num_nodes, 1) shaped vector
# # see the signal
# plt.plot(np.squeeze(y))
# directory = "figures/"+dataset
# plt.savefig(directory+"/signal_"+signal_distro+".pdf")

# # compute smoothness
# original_smoothness = computeSmoothness(y, L.cpu().detach().numpy())
# print("original smoothness:", original_smoothness)

# print("sparsity of original matrix:", np.count_nonzero(A) / (len(A)**2))

# #################################################################################
# # eq: 
# # F(wij) = wij*(yi-yj)**2 + beta*wij**2 - alpha**T * (log(Di)+log(Dj)) + (wij>=0)
# # the last term is modified to: gamma * 1/(1+exp(-wij))
# alpha = 0.7
# beta = 0.11
# gamma = 0.5
# LR = 0.01
# LR_decay = 0.01
# maxIter = 5000
# tolerance = 1e-8

# Z = utils.scipyMatMul(y,y)
# Ld = np.diag(L.cpu().detach().numpy())
# iLd = 1/Ld
# S = np.ones((len(Ld), len(Ld))) * iLd 
# # all elements in i^th column of S is same as ith element of iLd
# S = S+S.T

# edgeWgtNew = utils.gradientDescentS(edgeWgt, Z, S, alpha, \
# 					beta, gamma, LR, LR_decay, maxIter, tolerance, num_nodes, device)


# # construct new laplacian
# AF = AFLA(edgeWgtNew, num_nodes)
# LF = utils.constructLaplacian(AF)
# # compute new smoothness
# new_smoothness = computeSmoothness(y, LF)

# print("optimal smoothness:", new_smoothness)

# print("***Achieved graph with smoothness minimized for the given signal!***")

# print("sparsity of optimized matrix:", np.count_nonzero(AF) / (len(AF)**2))

# # #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# """
# this is the main approximation center! :(
# """
# # "nnz", "uniform", "row norm", "sparsity sampler_.1f"
# sampling_modes = ["nnz", "sparsity sampler_0.1", "uniform"]
# name_adder = "_uniform_nnz_sparse"
# samples = 200
# steps=10
# trials = 50
# data_matrix = AF
# node_signals = np.squeeze(y)
# ogSQ = new_smoothness


# # parameters for local optimization
# alphaS = 0.8
# betaS = 0.05
# gammaS = 0.2
# LRS = 0.001
# LR_decayS = 0.001
# maxIterS = 5000
# toleranceS = 1e-2

# # approximation of smoothness quotient
# # logger
# SQ_error_mean = {}
# SQ_error_p1 = {}
# SQ_error_p2 = {}
# change_mean = {}
# change_p1 = {}
# change_p2 = {}
# for sampling_mode in sampling_modes:
# 	print(sampling_mode)
# 	local_SQ_error = []
# 	local_change = []
# 	for tr in tqdm(range(trials)):
# 		sampled_adjacency, sampled_signals = \
# 			sampler(sampling_mode, data_matrix, node_signals, samples=samples)
# 		# given sampled signal, first smooth them using the grad descent
# 		edgeWgtS = GTE(sampled_adjacency)
# 		# get laplacian from sampled_adjanceny
# 		LdS = np.diag(np.sum(sampled_adjacency, axis=1))
# 		iLdS = 1/LdS
# 		SS = np.ones((len(LdS), len(LdS))) * iLdS
# 		# all elements in i^th column of S is same as ith element of iLd
# 		SS = SS+SS.T
# 		ZS = utils.scipyMatMul(sampled_signals, sampled_signals)

# 		edgeWgtSop = utils.gradientDescentS(edgeWgtS, ZS, SS, alphaS, \
# 						betaS, gammaS, LRS, LR_decayS, maxIterS, toleranceS, samples, device)
# 		ASop = AFLA(edgeWgtSop, samples)
		
# 		newSQ = sa(sampled_signals, ASop)
# 		change = np.count_nonzero(sampled_adjacency - ASop) / (samples**2)

# 		error = np.abs(ogSQ - newSQ) / ogSQ

# 		local_SQ_error.append(error)
# 		local_change.append(change)

# 	SQ_error_mean[sampling_mode] = np.mean(local_SQ_error)
# 	SQ_error_p1[sampling_mode] = np.percentile(local_SQ_error, 20)
# 	SQ_error_p2[sampling_mode] = np.percentile(local_SQ_error, 80)
# 	change_mean[sampling_mode] = np.mean(local_change)
# 	change_p1[sampling_mode] = np.percentile(local_change, 20)
# 	change_p2[sampling_mode] = np.percentile(local_change, 80)

# print("Analysis for", samples, "samples")
# print("error mean")
# print(SQ_error_mean)
# print("error 20th percentile")
# print(SQ_error_p1)
# print("error 80th percentile")
# print(SQ_error_p2)
# print("mean entries that were altered in the optimization step")
# print(change_mean)
# print("20th percentile of entries that were altered in the optimization step")
# print(change_p1)
# print("80th percentile of entries that were altered in the optimization step")
# print(change_p2)