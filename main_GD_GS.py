import torch
import numpy as np
from src.utils import AdjacencyFromLinearArray as AFLA
from src import utils
from src.viz import seeError as SE
from src.approximator import sampler
from tqdm import tqdm
from src.utils import signalAnalyzer as sa

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

# set up torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# graph parameters
num_nodes = 500
num_edges = int(0.5*(num_nodes-1)*num_nodes)

# weight initialization
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
x = np.random.random_sample(num_nodes)
Vgs = np.exp(-2*np.diag(eigvals))
# y is the signal
y = (eigvecs @ Vgs @ eigvecs.T) @ np.expand_dims(x, axis=1)
# the above line should produce a (num_nodes, 1) shaped vector

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
there is a bug here! Ugh! :(
"""
sampling_mode = "uniform"
dataset = "optimal"
signal_distro = "uniform"
min_samples = 10
max_samples = num_nodes
steps=10
trials = 50
data_matrix = AF
node_signals = np.squeeze(y)
ogSQ = new_smoothness
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
dataset_size = num_nodes
SE(SQ_error_mean, SQ_error_p1, SQ_error_p2, \
	dataset, sampling_mode, signal_distro, min_samples, \
	max_samples, steps, dataset_size, ogSQ)