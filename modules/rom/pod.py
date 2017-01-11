'''

This module has functions and solvers to implement
Reduced Order Modelling in the simulator.

'''
import sys
from scipy.sparse.linalg import spsolve
from numpy import array as np_array
from matplotlib import pyplot as plt


def linear_solver(A, b, basis):
    # Project right and left hand side of equation
    Ar = (basis.T.dot(A)).dot(basis)
    br = basis.T.dot(b)
    # Solve reduced system.
    xr = spsolve(Ar, br)
    # Project back
    x = basis.dot(xr)
    return x.reshape([-1, ])


def plot_energy(S, filename):

    cen = np_array(S**2).cumsum() / np_array(S**2).sum() * 100
    DPI = 100
    fig = plt.figure()

    # Energy subplot
    ax1 = fig.add_subplot(2, 1, 1)
    line = ax1.plot(S**2, "o-", linewidth=1)
    ax1.set_yscale("log")
    plt.title("Basis vector vs Energy")
    plt.xlabel("Basis vector number")
    plt.ylabel("Energy")
    plt.axis([0, None, None, None])
    plt.grid(True)

    # Cumulative energy subplot
    ax2 = fig.add_subplot(2, 1, 2)
    line = ax2.plot(cen, "o-", linewidth=1)
    ax2.axis([0, None, 90, 101])
    plt.title("Cumulative Energy")
    plt.xlabel("Basis vector number")
    plt.ylabel("Cumulative Energy")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('{}.png'.format(filename), bbox_inches='tight', dpi=DPI)

    return


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
    colormap = "inferno"
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
        # plt.suptitle(title)
        plt.tight_layout()

        plt.savefig('{}.png'.format(filename), bbox_inches='tight', dpi=DPI)

    return


def main():
    ''' It is run if this module is run. '''

    print(' POD functions.')


if __name__ == "__main__":
    main()
    sys.exit()
