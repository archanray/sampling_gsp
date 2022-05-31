import numpy as np
from scipy.linalg import sqrtm
import torch
from copy import copy
from tqdm import tqdm

def is_psd(x):
    return np.all(np.linalg.eigvals(x) > 0)


def get_triu_entries(M):
    B = np.ones_like(M)
    B = np.triu(B,1)
    q = M[B==1]
    return q

def laplacian(adjacencyMat):
    """
    generates a graph Laplacian
    """ 
    L = np.diag(np.sum(adjacencyMat, axis=1)) - adjacencyMat
    # L = L / np.linalg.norm(L) #max(np.linalg.norm(L),1)
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
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)
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


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N


def sample_positions(N, mu, sigma):
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 4, N)
    # sample the points
    X, Y = np.meshgrid(X, Y)

    
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    return pos


def generate_2D_gaussian(nodes, mu, sigma, dataset):
    if int(np.sqrt(nodes)) - np.sqrt(nodes) != 0:
        raise Exception("number of nodes should be a square of an integer")

    N = int(np.sqrt(nodes))
    # get the parameters
    # mu = np.array([0., 1.])
    # sigma = np.array([[ 1. , 0], [0,  1.]])
    mu = np.array(mu)
    sigma = np.array(sigma)

    # get the positions
    pos = sample_positions(N, mu, sigma)

    # get the signals
    signals = multivariate_gaussian(pos, mu, sigma)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(pos[:,:,0], pos[:,:,1], signals, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    directory = "figures/"+dataset
    plt.savefig(directory+"/signal_bi_gaussian.pdf")

    # row wise flatten and return
    return signals.flatten()