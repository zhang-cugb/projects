'''
Model of a single phase flow reservoir.
It simulates water flowing through canals in a single layer of rocks
'''

from modules.simulator.simulator import *
from modules.simulator.units import UnitRegistry
from modules.simulator.simplots import *
import numpy as np


# Units
u = UnitRegistry()

# ----------------------------
#        RESERVOIR MODEL
# ----------------------------

# GRID
Nx, Ny, Nz = np.array([50, 100, 1])
Sx, Sy, Sz = np.array([1, 1, 0.5]) * u.meter

depth = np.hstack((Sz * 0 * np.ones([Nx * Ny, ])))
grid = uniformCartesianGrid(Nx, Ny, Nz, Sx, Sy, Sz, depth)

# ROCK
# Permeability
# Low and high permeability values
LPERM = 1 * u.milli * u.darcy
HPERM = 1000 * u.milli *u.darcy
# Fill rock with low permeability 
perm = LPERM * np.ones([grid.cellnumber, ]) 
perm = np.random.rand(grid.cellnumber,) * perm
# Create canals with the sin function
A = grid.dim[0]/5
b = 0.03 /grid.dim[1]
shift = lambda x,h,v: A * np.sin(b*2*np.pi*(x-h)) + v
# Select cells to make canals 
horizontal = np.arange(grid.dim[0]//3,grid.cellnumber,grid.dim[0])
# First canal
k = shift(horizontal, 0, 0).astype('int')
perm[horizontal + k ] =     HPERM 
perm[horizontal + k + 1 ] = HPERM 
perm[horizontal + k + 2 ] = HPERM 
perm[horizontal + k - 1 ] = HPERM 
# Second canal
k = shift(horizontal, 2*np.pi/b, 21).astype('int')
perm[horizontal + k ] =     HPERM 
perm[horizontal + k + 1 ] = HPERM 
perm[horizontal + k - 1 ] = HPERM 
# Porosity
poro = 0.20 * np.ones([grid.cellnumber, ])
# Rock compressibility equal for all cells
cr = 1E-12 / u.psi
porofunc = lambda p: poro * np.exp(cr * (p - 2800 * u.psi))
rock = Rock(perm, poro, cr, porofunc)


# FLUID
cf = 5E-5 / u.psi
miu = lambda p:  1.0 * np.exp(5E-5 * (p/u.psi -2800)) * u.centi * u.poise
rho = lambda p: 1000 * np.exp(cf * (p - 15 * u.psi) )  * u.kilogram/ u.meter**3
fvf = lambda p: 1.0 * np.exp(- cf * (p - 15 * u.psi) )   # adimensional
fluid = singleFluid(miu, rho, fvf, cf)


# SCHEDULE
timesteps = 0.01 * np.ones(200) *u.day
sch = Schedule(timesteps)

# SOURCE TERM
source_empty = np.zeros([grid.cellnumber, 1])

# BOUNDARY
boundary = Boundary()

boundary.set_boundary_condition('W', 'constant-pressure', 3000 * u.psi)
boundary.set_boundary_condition('E', 'constant-pressure', 2800 * u.psi)

# Set initial conditions
p_init = 2800 * u.psi * np.ones([grid.cellnumber, ])

# WELLS
# Empty wells term
wells_empty = Wells(grid, rock, fluid)

# Initilize solver
LCM = LaggingCoefficients(
    grid, rock, fluid, wells_empty, source_empty, p_init, boundary, gravity=True)

# # Run simulation
# r, well_solution, sch = LCM.solve(sch, max_inner_iter = 2, tol = 1E-6, ATS = False)

# # Transform units to psi
# p = r/u.psi

# # We will  save a plot for each timestep and then use 
# # ffmpeg to create a gif
# acum_sum = np.cumsum(sch.timesteps) / u.day
# # Add "0" for the initial conditions
# acum_sum = np.hstack((0, acum_sum))

# for k in np.arange(0,acum_sum.size, 1): 
#     plotCellValues2D(grid, p[:, k], 'inferno', np.min(p), np.max(p),
#                      title='Pressure (psi). {:.3f} days'.format(acum_sum[k]), 
#                      filename='{}\\ex1_{}'.format(imgpath, k))