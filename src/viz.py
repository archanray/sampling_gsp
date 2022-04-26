import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as m
import os

def title_corrector(dataset):
	if dataset == "erdos":
		corrected_title = "Erdos-Renyi Graph"
	if dataset == "barabasi":
		corrected_title = "Barabasi-Albert Graph"
	return corrected_title

def generateGraph(A, signals):
	G = nx.from_numpy_matrix(A)
	for i in range(len(signals)):
		G.nodes[i]["weight"] = signals[i]
	return G

def seeGraph(A, signals, dataset, sampling_mode, name="graph_visualize", plot_node_size=10):
	# clear canvas
	plt.gcf().clear()
	vmin = np.floor(np.min(signals))
	vmax = np.ceil(np.max(signals))
	# directory to store the visualization once up
	save_dir = os.path.join("./figures", dataset)
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	# filename
	filename = name+"_"+sampling_mode+".pdf"
	total_path = os.path.join(save_dir, filename)

	# generate the graph
	G = generateGraph(A, signals)

	# create colorbar
	cdict = [(1,0,0), (0.8,0.8,0.8) ,(0,0,1)]
	cm = m.colors.LinearSegmentedColormap.from_list("my_colormap", cdict, 1024)

	fig, ax = plt.subplots()
	nx.draw_networkx(G, cmap=cm, with_labels=False, node_color=signals, 
		node_size=plot_node_size, vmin=vmin, vmax=vmax)
	cb = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin = vmin, vmax=vmax))
	cb._A = []
	plt.colorbar(cb)
	plt.title(title_corrector(dataset))
	plt.savefig(total_path)
	return None

def plot_FR(FR, SQ, dataset, sampling_mode, name="FR_visualize"):
	# clear canvas
	plt.gcf().clear()
	# directory to store the visualization once up
	save_dir = os.path.join("./figures", dataset)
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	# filename
	filename = name+"_"+sampling_mode+".pdf"
	total_path = os.path.join(save_dir, filename)

	# plot the frequency response
	plt.plot(range(len(FR)), FR)
	plt.xlabel("Nodes")
	plt.ylabel("frequency response")
	plt.title(title_corrector(dataset)+"- "+"smoothness quotient: "+\
		           str(float("{:.3f}".format(SQ))))
	plt.savefig(total_path)
	return None