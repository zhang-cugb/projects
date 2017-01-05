import numpy as np
from matplotlib import  pyplot as plt
from scipy import linalg



def pod(snapshots, energy, maxbasis):
	'''Return the reduced basis and the singular values.
	Arguments: 
		snapshots(np.array): snapshot matrix
		energy(float): cumulative energy of the reduced basis
		maxbasis(int):max. number of basis vectors in the reduced basis matrix
	Return:
		U (np.array): reduced basis matrix
		S (np.array): vector with the corresponding singular values
	'''
	#Singular value decomposition
	U,S,V = linalg.svd(snapshots)
	# Select vectors
	cen = np.array(S**2).cumsum()/np.array(S**2).sum()*100
	# Find index 
	indices = np.where(cen >= energy)
	i = indices[0][0]
	#Select
	i = max((i,maxbasis))

	Ur = U[:,:i]
	Sr = S[:i,]

	return Ur,Sr


def deim():
	'''Discrete Empirical Interpolation Method
	'''
	return


def projection(A, x, b,  phi):
	''' Project matrix A with basis phi
	'''
	Ar = phi * A
	return Ar, xr, br


def plot_basis(cell_values, nrows, ncols, grid, filename):
	'''Plot and save a figure with the basis functions 
	in the basis matrix corresponding to a single layer.
	Arguments:
		cell_values(np.array) : basis matrix
		nrows, ncols (int) : rows and columns in the figure
		filename (string) : path to save the figure
	Returns:
	'''
	layer_num = cell_values.shape[1]
	DPI = 100
	fig, axes = plt.subplots(nrows=int(nrows), ncols=int(ncols), dpi=DPI)
	colormap = "plasma"
	vmin = cell_values.min()
	vmax = cell_values.max()
	title = "Bases"

	for layer in range(0, layer_num):
	    Z = cell_values[:, layer]
	    Z = Z.reshape([grid.dim[0], grid.dim[1]], order='F')
	    ax = axes.flat[layer]
	    im = ax.pcolor(Z, cmap=colormap, vmin=vmin, vmax=vmax)
	    ax.set_title("Basis vector {}".format(layer))
	    # Limits for the axes
	    ax.set_ylim([0, grid.dim[0]])
	    ax.set_xlim([0, grid.dim[1]])
	    # Erase the ticks
	    ax.get_xaxis().set_ticks([])
	    ax.get_yaxis().set_ticks([])
	    ax.set_aspect('equal', adjustable='box-forced')
	    ax.invert_yaxis()

	    # Hide the empty axes
	    for i in range(int(layer_num), int(ncols * nrows)):
	        axes.flat[i].axis('off')
	    # Color bar
	    cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])
	    fig.colorbar(im, cax=cbar_ax)
	    #plt.suptitle(title)
	    plt.tight_layout()

	    plt.savefig('{}.png'.format(filename), bbox_inches='tight', dpi=DPI)

	return



def plot_energy(S, filename):
	
	cen = np.array(S**2).cumsum()/np.array(S**2).sum()*100
	DPI = 100
	fig = plt.figure()

	# Energy subplot
	ax1 = fig.add_subplot(2,1,1)
	line = ax1.plot(S**2, "o-",linewidth=1)
	ax1.set_yscale("log")
	plt.title("Basis vector vs Energy")
	plt.xlabel("Basis vector number")
	plt.ylabel("Energy");
	plt.axis([0, None, None, None])

	# Cumulative energy subplot
	ax2 = fig.add_subplot(2,1,2)
	line = ax2.plot(cen, "o-",linewidth=1)
	ax2.axis([0, None, 90, 101])
	plt.title("Cummulative Energy")
	plt.xlabel("Basis vector number")
	plt.ylabel("Cummulative Energy");

	plt.tight_layout()
	plt.savefig('{}.png'.format(filename), bbox_inches='tight', dpi=DPI)
	
	return




