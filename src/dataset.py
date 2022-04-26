import numpy as np

def add_synthetic_signals(adjacencyMat):
    """
    input adjacency matrix
    output synthetic signals on the nodes
    """
    num_nodes = len(adjacencyMat)
    node_signals = np.random.random(num_nodes)
    node_signals = 2*node_signals - 1.0
    return node_signals


def get_data(name="erdos", size=10, p=0.2):
    if name == "arxiv" or name == "facebook" or name == "erdos" or name == "barabasi":
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
        
        return A, dataset_size, min_sample_size, max_sample_size, add_synthetic_signals(A)
