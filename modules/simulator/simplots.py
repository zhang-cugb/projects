'''
This module contains the functions to plot
the data generated with the simulator
'''

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from .units import UnitRegistry

# dpi value for all the plots
DPI = 150
u = UnitRegistry()


def plotCellValues3D(grid, cell_values, colormap):
    '''Plot the cell values in a 3D plot.
        For example: permeability, porosity,
        pressure, saturation
    Arguments:
        grid
        cell_values : an array with values for each cell
        colormap:
        name:
    '''
    X = np.arange(0, grid.dim[1], 1)
    Y = np.arange(0, grid.dim[0], 1)
    X, Y = np.meshgrid(X, Y)
    cell_values = cell_values.reshape(
        [grid.dim[0], grid.dim[1], grid.dim[2]], order='F')
    # number of layers in reservoir
    layer_num = grid.dim[2]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ofst = 0.1
    for layer in range(0, layer_num):
        Z = cell_values[:, :, layer].reshape(
            grid.dim[0], grid.dim[1], order='F')
        # normalize Z to [0..1]
        Z = Z - Z.min()
        Z = Z / Z.max()
        ax.contourf(X, Y, Z, 100, zdir='z', offset=ofst, cmap=colormap)
        ofst = ofst + 0.2
    plt.show()
    return


def plotCellValues2D(grid, cell_values, colormap, vmin, vmax, title='None',
                     filename='None'):
    '''Plot the cell values in a 2D plot.
        Each layer of the reservoir becomes a subplot.
        For example: permeability, porosity,
        pressure, saturation
    Arguments:
        grid: cartesianGrid
        cell_values : an array with values for each cell
        colormap (string) : colormap name
        vmin, vmax (float) :  min and max value of the colormap
        title (string) : title of the plot
        filename (string): path to save the figure
    '''
    plt.close('all')

    X = np.arange(0, grid.dim[1], 1)
    Y = np.arange(0, grid.dim[0], 1)
    X, Y = np.meshgrid(X, Y)

    cell_values = cell_values.reshape(
        [grid.dim[0], grid.dim[1], grid.dim[2]], order='F')

    # number of layers in reservoir
    layer_num = grid.dim[2]
    if layer_num > 1:
        # Maximum number of plots in a row
        maxrow = 4
        if layer_num < maxrow:
            ncols = layer_num
        else:
            ncols = maxrow

        nrows = np.ceil(layer_num / ncols)

        fig, axes = plt.subplots(nrows=int(nrows), ncols=int(ncols))
        for layer in range(0, layer_num):
            Z = cell_values[:, :, layer]
            ax = axes.flat[layer]
            im = ax.pcolor(Z, cmap=colormap, vmin=vmin, vmax=vmax)
            ax.set_title("Layer {}".format(layer))
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
        plt.suptitle(title)
        plt.tight_layout()

    else:

        fig, ax = plt.subplots(nrows=1, ncols=1)
        Z = cell_values[:, :, 0]
        im = plt.pcolor(Z, cmap=colormap, vmin=vmin, vmax=vmax)
        ax.set_title("Layer {}".format(0))
        # Limits for the axes
        ax.set_ylim([0, grid.dim[0]])
        ax.set_xlim([0, grid.dim[1]])
        # Erase the ticks
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_aspect('equal')
        ax.invert_yaxis()
        # Color bar
        fig.colorbar(im)
        plt.suptitle(title, fontsize=12, y=1.01)

    plt.savefig('{}.png'.format(filename), bbox_inches='tight', dpi=DPI)
    #plt.show()
    return


def plotFluidProperties(fluid, pmin, pmax, filename='None'):
    '''Plot the properties of the fluid.
    Arguments:
    fluid
    pmin, pmax (float) : min and max boundaries for the pressure in pascals
    filename (string): path to save the figure
    '''
    pressure = np.linspace(pmin, pmax, 100)
    plt.subplot(2, 2, 1)
    plt.title('Density (kg/m3)')
    plt.plot(pressure / u.psi, fluid.rho(pressure))
    plt.xlim([pmin / u.psi, pmax / u.psi])
    plt.xlabel('Pressure (psi)')

    plt.subplot(2, 2, 2)
    plt.title('Viscosity (centi-poise)')
    plt.plot(pressure / u.psi, fluid.miu(pressure) / u.centi / u.poise)
    plt.xlim([pmin / u.psi, pmax / u.psi])
    plt.xlabel('Pressure (psi)')

    plt.subplot(2, 2, 3)
    plt.title('Formation volume factor')
    plt.plot(pressure / u.psi, fluid.fvf(pressure))
    plt.xlim([pmin / u.psi, pmax / u.psi])
    plt.xlabel('Pressure (psi)')

    plt.tight_layout()

    plt.savefig('{}.png'.format(filename), dpi=DPI)
    return


def plotRate(wells, sch, path):
    '''
    Plot rate vs time , and acummulative production for each well.
    The units are barrels.
    Arguments:
    wells: well object
    sch: schedule
    path (string) : path to folder to save figures
    '''
    plt.close('all')

    time = np.hstack((0, sch.timesteps.cumsum())) / u.day
    # Rate plot
    for wi in range(0, len(wells.wells)):
        rate = np.array(wells.wells[wi]['rate_sol']) / u.barrel * u.day
        plt.plot(time, rate, 'o-', label=wells.wells[wi]['name'])
    plt.legend(loc=1)
    plt.title('Rate vs time')
    plt.xlabel('Time (days)')
    plt.ylabel('Rate (barrels/day)')
    plt.savefig('{}\well_rate.png'.format(path), dpi=DPI)

    plt.close('all')
    # Cumulative production plot
    for wi in range(0, len(wells.wells)):
        rate = np.array(wells.wells[wi]['rate_sol']) / u.barrel * u.day
        production = rate * (np.hstack((0, sch.timesteps)) / u.day)
        cumproduction = production.cumsum()
        plt.plot(time, cumproduction, 'o-', label=wells.wells[wi]['name'])
    plt.legend(loc=4)
    plt.title('Cumulative production vs time')
    plt.xlabel('Time (days)')
    plt.ylabel('Production (barrels)')
    plt.savefig('{}\production.png'.format(path), dpi=DPI)


def plotRateTwoPhase(wells, sch, path):
    '''
    Plot rate vs time , and acummulative production for each well.
    The units are barrels.
    Arguments:
    wells: well object
    sch: schedule
    path (string) : path to folder to save figures
    '''
    plt.close('all')
    plt.figure()

    time = np.hstack((0, sch.timesteps.cumsum())) / u.day
    # Rate plot
    for wi in range(0, len(wells.wells)):
        oil_rate = np.array(wells.wells[wi]['rate_sol'][1]) / u.barrel * u.day
        water_rate = np.array(wells.wells[wi]['rate_sol'][0]) / u.barrel * u.day

        name1 = wells.wells[wi]['name'] + " : " + "oil"
        name2 = wells.wells[wi]['name'] + " : " + "water"
        plt.plot(time, oil_rate, '-', label = name1, linewidth=3)
        #plt.plot(time, water_rate, '-', label = name2, linewidth=3)

    plt.legend(loc=1)
    plt.title('Oil rate vs time')
    plt.xlabel('Time (days)')
    plt.ylabel('Rate (barrels/day)')
    plt.savefig('{}\well_rate.png'.format(path), dpi=DPI)

    plt.close('all')
    # Cumulative production plot
    for wi in range(0, len(wells.wells)):
        oil_rate = np.array(wells.wells[wi]['rate_sol'][1]) / u.barrel * u.day
        water_rate = np.array(wells.wells[wi]['rate_sol'][0]) / u.barrel * u.day

        production = oil_rate * (np.hstack((0, sch.timesteps)) / u.day)
        cumproduction = production.cumsum()
        plt.plot(time, cumproduction, '-', label=wells.wells[wi]['name'], linewidth=3)
    plt.legend(loc=4)
    plt.title('Cumulative oil production vs time')
    plt.xlabel('Time (days)')
    plt.ylabel('Production (barrels)')
    plt.savefig('{}\production.png'.format(path), dpi=DPI)

    plt.close('all')
    # Water cut
    for wi in range(0, len(wells.wells)):
        oil_rate = np.array(wells.wells[wi]['rate_sol'][1])
        water_rate = np.array(wells.wells[wi]['rate_sol'][0])

        cut = water_rate / (oil_rate + water_rate)
        plt.plot(time, cut, '-', label=wells.wells[wi]['name'], linewidth=3)
    x1, x2, y1, y2 = plt.axis()
    plt.axis([x1, x2, 0, 1])

    plt.legend(loc=2)
    plt.title('Water cut vs time')
    plt.xlabel('Time (days)')
    plt.ylabel('Water cut (fraction)')
    plt.savefig('{}\water_cut.png'.format(path), dpi=DPI)


def plotMatrix(A):
    ''' Spy plot of matrix A.
    Arguments:
        A: (numpy.array, numpy.matrix, sparse.matrix)
    '''
    plt.spy(A, markersize=5)    
    plt.show()
    return
