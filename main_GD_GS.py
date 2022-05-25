import torch
import numpy as np
from src.utils import AdjacencyFromLinearArray as AFLA
from src import utils
from src.viz import seeCombinedError as sCE
from src.approximator import sampler
from tqdm import tqdm
from src.utils import signalAnalyzer as sa
import matplotlib.pyplot as plt

#################################################################################
"""
following lines are executed to avoid errors due to 
multiple installations
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#################################################################################

def computeSmoothness(y, L):
	return np.squeeze((y.T @ L @ y) / np.linalg.norm(y)**2)

signal_distro = "gaussian"
dataset = "optimal"
# set up torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# graph parameters
num_nodes = 1500
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

Vgs = np.exp(-2*np.diag(eigvals))
# y is the signal
y = (eigvecs @ Vgs @ eigvecs.T) @ np.expand_dims(x, axis=1)
# the above line should produce a (num_nodes, 1) shaped vector
# see the signal
plt.plot(np.squeeze(y))
directory = "figures/"+dataset
plt.savefig(directory+"/signal_"+signal_distro+".pdf")

# compute smoothness
original_smoothness = computeSmoothness(y, L.cpu().detach().numpy())
print("original smoothness:", original_smoothness)

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

print("optimal smoothness:", new_smoothness)

print("***Achieved graph with smoothness minimized for the given signal!***")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
"""
this is the main approximation center! :(
"""
# "nnz", "uniform", "row norm", "sparsity sampler_.1f"
sampling_modes = ["nnz", "sparsity sampler_0.1", "uniform"]
name_adder = "_uniform_nnz_sparse"
min_samples = 100
max_samples = 600
steps=10
trials = 50
data_matrix = AF
node_signals = np.squeeze(y)
ogSQ = new_smoothness

# approximation of smoothness quotient
# logger
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
			SQ = sa(sampled_signals, sampled_adjacency)
			# print(SQ)
			error = np.abs(ogSQ - SQ)
			local_SQ_error.append(error)
		SQ_error_mean.append(np.mean(local_SQ_error))
		SQ_error_p1.append(np.percentile(local_SQ_error, 20))
		SQ_error_p2.append(np.percentile(local_SQ_error, 80))

	sQ_comnined_mean[sampling_mode] = SQ_error_mean
	sQ_comnined_p1[sampling_mode] = SQ_error_p1
	sQ_comnined_p2[sampling_mode] = SQ_error_p2

# visualize the approximation result
dataset_size = num_nodes
sCE(sQ_comnined_mean, sQ_comnined_p1, sQ_comnined_p2, \
	dataset, signal_distro, min_samples, \
	max_samples, steps, dataset_size, ogSQ, name_adder)

# n = len(data_matrix)
# sQ_mean = np.zeros(len(range(min_samples, max_samples,steps)))
# sQ_std = np.zeros(len(range(min_samples, max_samples,steps)))
# list_of_available_indices = list(range(n))
# probs = np.ones(n) / float(n)


# count = 0
# for numSamples in tqdm(range(min_samples, max_samples,steps)):
# 	local_sQ = []
# 	for tr in range(trials):
# 		sample_indices = np.sort(np.random.choice(list_of_available_indices,\
# 							size=numSamples, replace=True, p=probs))

# 		sampledAM = data_matrix[sample_indices][:, sample_indices]
# 		sampledY = node_signals[sample_indices]

# 		# compute local smoothness quotient
# 		LS = np.diag(np.sum(sampledAM, axis=1)) - sampledAM
# 		LS = LS / np.linalg.norm(LS) # why this?
# 		q = np.expand_dims(sampledY, axis=1)
# 		local_sQ.append(sampledY.T @ LS @ sampledY / (np.linalg.norm(sampledY)**2))

# 	sQ_mean[count] = np.mean(local_sQ)
# 	sQ_std[count] = np.std(local_sQ)
# 	count+=1

# import matplotlib.pyplot as plt

# plt.plot(sQ_mean, label="mean")
# plt.plot(sQ_mean-sQ_std, label="mean-std")
# plt.plot(sQ_mean+sQ_std, label="mean+std")
# plt.legend()
# plt.show()


