# code to visualize a true signal and a sampled signal on a graph

import numpy as np
import matplotlib.pyplot as plt
from src.viz import seeGraph as sg
from src.dataset import get_data
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot_signals_grid(dataset, signal_type, A, signals):
	"""
	visualize the signal
	"""
	N = int(np.sqrt(len(A)))
	X = np.linspace(-3, 3, N)
	Y = np.linspace(-3, 4, N)
	X, Y = np.meshgrid(X, Y)
	mu = np.array([0., 0.])
	Sigma = np.array([[ 1. , -0.5], [-0.5,  1.]])
	# Pack X and Y into a single 3-dimensional array
	pos = np.empty(X.shape + (2,))
	pos[:, :, 0] = X
	pos[:, :, 1] = Y
	Z = np.reshape(signals, (-1, N))

	# Create a surface plot and projected filled contour plot under it.
	plt.gcf().clear()
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
					cmap=cm.viridis)

	# cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

	# Adjust the limits, ticks and view angle
	# ax.set_zlim(-0.15,0.2)
	# ax.set_zticks(np.linspace(0,0.2,5))
	# ax.view_init(27, -21)

	plt.savefig("figures/"+dataset+"/"+"original_signal.pdf")
	return None

def main():
	# hyper parameters
	dataset = "grid"
	sampling_mode = "uniform"
	signal_distro = "smoothed gaussian" 
	samples = 200
	num_nodes = 40
	p = 0.01

	#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
	## generate the graph and get required parameters
	data_matrix, dataset_size, _, _, node_signals = \
								get_data(dataset, size=num_nodes, \
										p=p, signal=signal_distro)
	#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
	# see the graph
	sg(data_matrix, node_signals, dataset, \
		sampling_mode, name="ogGraph_visualize", plot_node_size=10)

	# see the signal
	plot_signals_grid(dataset, signal_distro, data_matrix, node_signals)

	return None

if __name__ == "__main__":
	main()
