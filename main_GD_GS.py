import torch
import numpy as np
from src.utils import AdjacencyFromLinearArray as AFLA

# graph parameters
num_nodes = 50
num_edges = int(0.5*(num_nodes-1)*num_nodes)

# weight initialization
edgeWgt = np.random.random_sample(num_edges)

# construct Laplacian
A = AFLA(edgeWgt, num_nodes)
D = np.diag(np.sum(A, axis=1))
L = D-A
L = L / np.linalg.norm(L) # why this?
eigs, eigvecs = np.linalg.eig(L)
print(eigs.shape, eigvecs.shape)

# construct smooth signal(Heat Kernel)
