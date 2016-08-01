from modules.simulator.simulator import *
from modules.simulator.two_phase_flow import *
from modules.simulator.simplots import *
from modules.simulator.units import UnitRegistry
import numpy as np

# Set image folder path
imgpath = ".\\images"
# Units
u = UnitRegistry()

# Reservoir Model

# GRID
Nx, Ny, Nz = np.array([10, 10, 2])
Sx, Sy, Sz = np.array([30, 30, 30]) * u.feet

depth = np.hstack((Sz * 0 * np.ones([Nx * Ny, ]), 
                  Sz * 1 * np.ones([Nx * Ny, ])))
                
grid = uniformCartesianGrid(Nx, Ny, Nz, Sx, Sy, Sz, depth)

# ROCK
perm = 100 * np.ones([grid.cellnumber, ]) * u.milli * u.darcy
# Low permeability cells
perm[[3,4,14,15,25,26, 36, 37, 47, 48,58,59]] = 1 *u.milli *u.darcy
poro = 0.15 * np.ones([grid.cellnumber, ])
# Rock compressibility equal for all cells
cr = 3E-6 / u.psi
porofunc = lambda p: poro * np.exp(cr * (p - 2800 * u.psi))
rock = Rock(perm, poro, cr, porofunc)

#  FLUID
# Oil phase
cf_o = 1E-5 / u.psi
miu_o = lambda p: ( -6E-5 * (p / u.psi) + 1.1791 ) * u.centi * u.poise
rho_o = lambda p: 45 * np.exp(cf_o * (p - 2800 * u.psi)) * u.pound / u.feet**3
fvf_o = lambda p : -1E-5 * (p / u.psi) + 1.018
# Water phase
cf_w = 3E-6 / u.psi
miu_w = lambda p: 1 * np.exp(5E-5 * (p / u.psi - 2800)) * u.centi * u.poise
#miu_w = lambda p: (1 +  5E-5 * (p / u.psi - 2800)) * u.centi * u.poise
rho_w = lambda p: 62.4 * np.exp(cf_w * (p - 2800 * u.psi)) * u.pound / u.feet**3
fvf_w = lambda p: 1 * np.exp(-3E-6 * (p / u.psi - 2800 ))   # adimensional

# Relative permeability functions
Swr = 0.25
Sor = 0.30
Swmin = Swr
Somin = Sor
Korw = 0.08
Koro = 0.7
Swmax = 1 - Somin
Somax = 1 - Swmin

kr_o = lambda So: Koro * ((So - Sor) / (1 - Sor - Swr)) ** 3
kr_w = lambda Sw: Korw * ((Sw - Swr) / (1 - Sor - Swr)) ** 2


dkr_o = lambda So: 3 * Koro  * (So - Sor) ** 2 / (1 - Sor - Swr) ** 3
dkr_w = lambda Sw: 2 * Korw * (Sw - Swr) / (1 - Sor - Swr) ** 2

def krwfunc(Sw):
    ''' Water relative permeambility
    Argument: Sw (np.array)'''
    kr = np.zeros_like(Sw)
    imin = Sw <= Swmin
    imax = Sw >= Swmax
    i = np.logical_not(np.logical_or(imin, imax))
    # The array is already filled with zeros    
    # kr[imin] = 0 
    kr[i] = kr_w(Sw[i])
    kr[imax] = Korw
    return kr

def krofunc(Sw):
    ''' Oil relative permeambility
    Argument: Sw (np.array)'''
    So = 1 - Sw
    kr = np.zeros_like(So)
    imin = So <= Somin
    imax = So >= Somax
    i = np.logical_not(np.logical_or(imin, imax))
    # The array is already filled with zeros
    # kr[imin] = 0
    kr[i] = kr_o(So[i])
    kr[imax] = Koro
    return kr


def d_krwfunc(Sw):
    ''' Water relative permeambility
    Argument: Sw (np.array)'''
    kr = np.zeros_like(Sw)
    imin = Sw <= Swmin
    imax = Sw >= Swmax
    i = np.logical_not(np.logical_or(imin, imax))
    # The array is already filled with zeros    
    # kr[imin] = 0 
    kr[i] = dkr_w(Sw[i])
    
    return kr

def d_krofunc(Sw):
    ''' Oil relative permeambility
    Argument: Sw (np.array)'''
    So = 1 - Sw
    kr = np.zeros_like(So)
    imin = So <= Somin
    imax = So >= Somax
    i = np.logical_not(np.logical_or(imin, imax))
    # The array is already filled with zeros
    # kr[imin] = 0
    kr[i] = dkr_o(So[i])
    
    return kr

# For this model, there is zero capillary pressure
pc = lambda Sw: 0 * Sw

fluid = blackOil((miu_o, miu_w), (rho_o, rho_w),
                 (fvf_o, fvf_w), (cf_o, cf_w), (krofunc, krwfunc), pc)

fluid.oilphase.d_kr = d_krofunc
fluid.waterphase.d_kr = d_krwfunc

# SCHEDULE
t1 = 0.1 * np.ones(100) * u.day
t2 = 0.5 * np.ones(100) * u.day
t3 = 1 * np.ones(120) * u.day

timesteps = np.hstack((t1,t2,t3))
sch = Schedule(timesteps[:])

# BOUNDARY'S
boundary = Boundary()

# INITIAL CONDITIONS
p_init = 3000 * u.psi * np.ones([grid.cellnumber, ])
sw_init = 0.2 * np.ones([grid.cellnumber, ])



# WELLS
# Producer well
wells = Wells(grid, rock, fluid)
ci = Nx * (Ny-1) - 5
wells.add_vertical_well(0.35 * u.feet, ci, 2900 * u.psi, 0, 'Producer')

# Water injection well
source = np.zeros([grid.cellnumber * fluid.phasenumber, 1])
# Two water injectors  in cell 18 and cell 11.
# The water term is on (cellnumber * 2)
# The oil term is on (cellnumber * 2 + 1)
source[22] = -150 * u.barrel / u.day
source[36] = -170 * u.barrel / u.day

