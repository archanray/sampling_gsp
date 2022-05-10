import numpy as np
from scipy.stats import multivariate_normal

def add_synthetic_signals(adjacencyMat):
    """
    input adjacency matrix
    output synthetic signals on the nodes
    """
    num_nodes = len(adjacencyMat)
    node_signals = np.random.random(num_nodes)
    node_signals = 2*node_signals - 1.0
    return node_signals

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

def add_gaussian_signal(A, mode="laplacian", pow_=-1):
    """
    creates a synthetic gaussian signal on a graph
    """
    # gaussian_vector = np.random.normal(size=len(A))
    # Mean vector and covariance matrix
    N = int(np.sqrt(len(A)))
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 4, N)
    X, Y = np.meshgrid(X, Y)
    mu = np.array([0., 1.])
    Sigma = np.array([[ 1. , -0.5], [-0.5,  1.]])
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    Z = multivariate_gaussian(pos, mu, Sigma)
    ## uncomment the following to use scipy's multivariate normal distro
    # F = multivariate_normal(mu, Sigma)
    # Z = F.pdf(pos)
    gaussian_vector = Z.flatten() # flats by going through each row 
    if mode == "true":
        f = gaussian_vector
    if mode == "laplacian":
        f = (np.linalg.matrix_power(\
                (np.diag(np.sum(A, axis=1)) - A), \
                    pow_\
                ) @ \
                np.expand_dims(gaussian_vector, axis=1)\
            )
    # change this to add 2D gaussian signal and not a 1D gaussian signal
    return f.squeeze()

def get_data(name="erdos", size=10, p=0.2, signal="smoothed gaussian", pow_=-1):
    if name == "arxiv" or name == "facebook" or name == "erdos" or name == "barabasi" or name == "grid":
        """
        dataset arxiv: https://snap.stanford.edu/data/ca-CondMat.html
        dataset facebook: https://snap.stanford.edu/data/ego-Facebook.html
        """
        if name == "arxiv":
            data_file = "./data/CA-CondMat.txt"
        if name == "facebook":
            data_file = "./data/facebook_combined.txt"
        import networkx as nx
        if name == "erdos":
            from networkx.generators.random_graphs import erdos_renyi_graph
            g = erdos_renyi_graph(size, p=p)
        if name == "barabasi":
            from networkx.generators.random_graphs import barabasi_albert_graph
            g = barabasi_albert_graph(size, int(size*p))
        if name == "grid":
            from networkx import grid_graph
            g = grid_graph([size, size])
        if name == "arxiv" or name == "facebook":
            g = nx.read_edgelist(data_file,create_using=nx.DiGraph(), nodetype = int)
        A = nx.adjacency_matrix(g)
        A = A.todense()
        A = np.asarray(A)
        if name == "facebook":
            A = A+A.T # symmetrizing as the original dataset is directed

        dataset_size = len(A)
        
        min_sample_size = int(dataset_size * 0.01)
        max_sample_size = int(dataset_size * 0.2)
        
        if signal == "uniform":
            return A, dataset_size, min_sample_size, max_sample_size, add_synthetic_signals(A)
        if signal == "smoothed gaussian":
            return A, dataset_size, min_sample_size, max_sample_size, add_gaussian_signal(A, "laplacian", pow_)
