import numpy as np
from scipy.linalg import sqrtm
import torch
from copy import copy
from tqdm import tqdm

def is_psd(x):
    return np.all(np.linalg.eigvals(x) > 0)

def laplacian(adjacencyMat):
    """
    generates a graph Laplacian
    """ 
    L = np.diag(np.sum(adjacencyMat, axis=1)) - adjacencyMat
    L = L / np.linalg.norm(L) #max(np.linalg.norm(L),1)
    # add noise to the diagonal to ensure PSD
    # L = L + np.diag(1e-14*np.ones(len(L)))
    return L

def matSqrt(M):
    """
    compute matrix square root using numpy
    """
    e, v = np.linalg.eig(M)
    e = np.real(e)
    sqrt_M = v @ np.diag(np.sqrt(e)) @ np.linalg.inv(v)
    return sqrt_M

def signalAnalyzer(f, M, FR=False):
    """
    compute soothness quotient
    """
    L = laplacian(M)
    f = np.expand_dims(f, axis=1)
    smoothnessQuotient = (f.T @ L @ f / (np.linalg.norm(f)**2)).squeeze()
    if FR:
        frequencyResponse = np.real(f.T @ sqrtm(L)).squeeze()
        # frequencyResponse = np.real(f.T @ matSqrt(L)).squeeze()

        return smoothnessQuotient, frequencyResponse
    else:
        return smoothnessQuotient

def AdjacencyFromLinearArray(edgeWgt, num_nodes):
    A = np.ones((num_nodes, num_nodes))
    A = np.triu(A,1)
    A[A == 1] = edgeWgt
    A = A+A.T
    return A
    
def constructLaplacian(A):
    D = np.diag(np.sum(A, axis=1))
    L = D-A
    L = L / np.linalg.norm(L)
    return L

def scipyMatMul(x,y):
    from scipy.spatial.distance import cdist
    dists = cdist(x,y, metric="euclidean")
    return dists*dists

def computeSquaredDistance(X, Y):
    """
    fast squared euclidean distance computation:
    X shape is n x features
    """
    num_test = X.shape[0]
    num_train = Y.shape[0]
    dists = np.zeros((num_test, num_train))
    sum1 = np.sum(np.power(X,2), axis=1)
    sum2 = np.sum(np.power(Y,2), axis=1)
    sum3 = 2*np.dot(X, Y.T)
    dists = sum1.reshape(-1,1) + sum2
    dists = np.sqrt(dists - sum3)
    dists = dists / np.max(dists)
    return dists*dists

def getTriuVectorFromMatrix(A):
    indices = np.triu_indices_from(A, 1)
    A = np.asarray(A[indices])
    return A

def y(eW, Z, beta, alpha, S):
    return Z - 2*beta*eW + alpha*S

def gradientDescentS(edgeWgt, Z, S, alpha, \
                    beta, gamma, LR, LR_decay, maxIter, tol, num_nodes, device):
    """
    gradient descent
    """
    # edgeWgt = torch.tensor(edgeWgt, requires_grad=True).to(device)
    Z = getTriuVectorFromMatrix(Z)
    # Z = torch.tensor(Z).to(device)
    # S = torch.tensor(S, requires_grad=True).to(device)
    for k in tqdm(range(maxIter)):
        S = getTriuVectorFromMatrix(S)

        holdEdge = copy(edgeWgt)
        edgeWgt = edgeWgt - LR*y(edgeWgt, Z, beta, alpha, S)
        edgeWgt = np.maximum(0, edgeWgt)
        A = AdjacencyFromLinearArray(edgeWgt, num_nodes)
        Ld = np.sum(A, axis=1)
        Ld[Ld == 0] = 1e+8
        iLd = 1/Ld
        S = np.ones((len(Ld), len(Ld))) * iLd 
        S = S+S.T

        if np.linalg.norm(edgeWgt-holdEdge) / np.linalg.norm(edgeWgt) < tol:
            break
        if np.remainder(k, 100) == 0:
            LR *= LR_decay

    return edgeWgt