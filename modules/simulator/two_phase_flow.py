'''

This module adds two phase (oil - water) flow capabilities  to the simulator.

'''

from .simulator import *


class blackOil:
    ''' Two phase, immiscible, fluid.
    The first phase is oil and the second one is water.

    Attributes:
        miu (function, function): viscosity
        rho (function, function): density
        fvf (function, function): formation volume factor
        cf (float, float): compressibility
        kr (function, function): relative permeability. kr = func(Sw).
        pc (function): capillary pressure. pc = func(Sw).
        self.waterphase (singleFluid)
        self.oilphase (singleFluid)
        phasenumber (int) : number of phases
    '''

    def __init__(self, miu, rho, fvf, cf, kr, pc):
        '''
        Arguments:
            miu (func, func): viscosity
            rho (func, func): density
            fvf (func, func): formation volume factor
            cf (float, float): compressibility
            kr (function): relative permeability
            pc (function): capillary pressure. pc(Sw)
        '''
        self.miu = miu
        self.rho = rho
        self.fvf = fvf
        self.cf = cf
        self.kr = kr
        self.pc = pc
        self.phasenumber = 2

        self.oilphase = singleFluid(miu[0], rho[0], fvf[0], cf[0], kr[0])
        self.waterphase = singleFluid(miu[1], rho[1], fvf[1], cf[1], kr[1])
        

def fluidTransTwoPhase(grid, fluid, pressure, saturation):
    '''Computes the fluid part of the transmissibility for each cell.
        The transmissibility is upwinded to improve the stability of the solution.
    Arguments:
        grid (object) : cartesianGrid
        fluid (object) : singleFluid
        pressure (np.array) : vector with the oil pressure of each cell.
        saturation (np.array) : vector with the water saturation of each cell.
    Returns:
        Tx, Ty, Tz (np.array) : transmissibility matrices for each dimension.

    '''
    nrows, ncols, nlayers = grid.dim
    # The pressure in matrix form
    p = pressure.astype('float32').reshape(grid.dim, order='F')
    sw = saturation.astype('float32').reshape(grid.dim, order='F')
    # Fluid properties

    miu = fluid.miu(p)
    fvf = fluid.fvf(p)
    kr = fluid.kr(sw) 
    # Fluid  transmissibility for each cell
    T = kr / (miu * fvf)
    # ===========================
    # Upwind the transmissibility
    # ===========================
    # Pressure head
    rho = fluid.rho(p)
    head = potential(grid, p , rho)

    Tx = T.swapaxes(0, 1)
    headx = head.swapaxes(0, 1)
    Tx = upwind(headx[:, :-1, :], Tx[:, :-1, :], headx[:, 1:, :], Tx[:, 1:, :])

    Ty = T
    heady = head
    Ty = upwind(heady[:, :-1, :], Ty[:, :-1, :], heady[:, 1:, :], Ty[:, 1:, :])

    Tz = T.swapaxes(1, 2)
    headz = head.swapaxes(1, 2)
    Tz = upwind(headz[:, :-1, :], Tz[:, :-1, :], headz[:, 1:, :], Tz[:, 1:, :])

    # Complete the transmissibilities with zeros
    Tx = np.hstack((Tx, np.zeros([ncols, 1, nlayers]))).swapaxes(0, 1)
    Ty = np.hstack((Ty, np.zeros([nrows, 1, nlayers])))
    Tz = np.hstack((Tz, np.zeros([nrows, 1, ncols]))).swapaxes(1, 2)

    return Tx, Ty, Tz


def derivative_fluidTransTwoPhase(grid, fluid, pressure, saturation):
    '''Computes the derivative of the fluid part of the transmissibility for each cell.
        The derivative takes into account the upwinding of the transmissibility.
    Arguments:
        grid (object) : cartesianGrid
        fluid (object) : singleFluid
        pressure (np.array) : vector with the oil pressure of each cell.
        saturation (np.array) : vector with the water saturation of each cell.
    Returns:
        dtx_dp, dty_dp, dtz_dp (np.array) : derivative with respect to oil pressure
        dtx_dsw, dty_dsw, dtz_dsw (np.array) : derivative with respect to water saturation
    '''
    nrows, ncols, nlayers = grid.dim
    # The pressure in matrix form
    p = pressure.reshape(grid.dim, order='F')
    sw = saturation.reshape(grid.dim, order='F')
    # Fluid properties
    miu = fluid.miu(p)
    fvf = fluid.fvf(p)
    kr = fluid.kr(sw)

    # Derivatives of the fluid properties
    #eps_sw = 1E-6
    eps_p = 10 * u.psi

    # Viscosity
    d_miu = (fluid.miu(p + eps_p) - fluid.miu(p))/ eps_p
    # Formation volume factor
    d_fvf = (fluid.fvf(p + eps_p) - fluid.fvf(p))/ eps_p
    # Relative permeability
    d_kr = fluid.d_kr(sw)
        
    # Derivative of transmissibility with respect to ...
    # OIL PRESSURE
    dt_dp = kr * (-1/(fvf * miu) ** 2) * (miu * d_fvf + fvf * d_miu)
    # WATER SATURATION
    dt_dsw = (1/(fvf * miu)) * d_kr 


    # ======================================================
    # Derivatives and upwinding. Example 1D, with 3 cells.
    # |0|1|2|, flow -->
    # T1/2 = T0, then, dT1/2 / dP0 = nonzero
    # T1/2 = T0, then, dT1/2 / dP1 = 0
    
    # |0|1|2|, <--- flow 
    # T1/2 = T1, then, dT1/2 / dP0 = zero
    # T1/2 = T1, then, dT1/2 / dP1 = nonzero

    # ======================================================
    # DERIVATIVES WITH RESPECT TO NEIGHBOR CELL 
    # ======================================================
    # "Upwind" the derivatives
    # Pressure head
    rho = fluid.rho(p)
    head = potential(grid, p , rho)

    # Derivatives with respect to the OIL PRESSURE
    dtx_dp = dt_dp.swapaxes(0, 1)
    headx = head.swapaxes(0, 1)
    zerox = np.zeros_like(headx)
    dtx_dp = upwind(headx[:, :-1, :], zerox[:, :-1, :], headx[:, 1:, :], dtx_dp[:, 1:, :])

    dty_dp = dt_dp
    heady = head
    zeroy = np.zeros_like(heady)
    dty_dp = upwind(heady[:, :-1, :], zeroy[:, :-1, :], heady[:, 1:, :], dty_dp[:, 1:, :])

    dtz_dp = dt_dp.swapaxes(1, 2)
    headz = head.swapaxes(1, 2)
    zeroz = np.zeros_like(headz)
    dtz_dp = upwind(headz[:, :-1, :], zeroz[:, :-1, :], headz[:, 1:, :], dtz_dp[:, 1:, :])

    # Complete the transmissibilities with zeros
    dtx_dp = np.hstack((dtx_dp, np.zeros([ncols, 1, nlayers]))).swapaxes(0, 1)
    dty_dp = np.hstack((dty_dp, np.zeros([nrows, 1, nlayers])))
    dtz_dp = np.hstack((dtz_dp, np.zeros([nrows, 1, ncols]))).swapaxes(1, 2)

    # Derivatives with respect to the WATER SATURATION
    dtx_dsw = dt_dsw.swapaxes(0, 1)
    dtx_dsw = upwind(headx[:, :-1, :], zerox[:, :-1, :], headx[:, 1:, :], dtx_dsw[:, 1:, :])

    dty_dsw = dt_dsw
    dty_dsw = upwind(heady[:, :-1, :], zeroy[:, :-1, :], heady[:, 1:, :], dty_dsw[:, 1:, :])

    dtz_dsw = dt_dsw.swapaxes(1, 2)
    dtz_dsw = upwind(headz[:, :-1, :], zeroz[:, :-1, :], headz[:, 1:, :], dtz_dsw[:, 1:, :])

    # Complete the transmissibilities with zeros
    dtx_dsw = np.hstack((dtx_dsw, np.zeros([ncols, 1, nlayers]))).swapaxes(0, 1)
    dty_dsw = np.hstack((dty_dsw, np.zeros([nrows, 1, nlayers])))
    dtz_dsw = np.hstack((dtz_dsw, np.zeros([nrows, 1, ncols]))).swapaxes(1, 2)

    # ======================================================
    # DERIVATIVES WITH RESPECT TO  CELL "i" 
    # ======================================================
    # "Upwind" the derivatives
    # Derivatives with respect to the OIL PRESSURE
    dtx_dpi = dt_dp.swapaxes(0, 1)
    dtx_dpi = upwind(headx[:, :-1, :], dtx_dpi[:, :-1, :], headx[:, 1:, :], zerox[:, 1:, :])

    dty_dpi = dt_dp
    dty_dpi = upwind(heady[:, :-1, :], dty_dpi[:, :-1, :], heady[:, 1:, :], zeroy[:, 1:, :])

    dtz_dpi = dt_dp.swapaxes(1, 2)
    dtz_dpi = upwind(headz[:, :-1, :], dtz_dpi[:, :-1, :], headz[:, 1:, :], zeroz[:, 1:, :])

    # Complete the transmissibilities with zeros
    dtx_dpi = np.hstack((dtx_dpi, np.zeros([ncols, 1, nlayers]))).swapaxes(0, 1)
    dty_dpi = np.hstack((dty_dpi, np.zeros([nrows, 1, nlayers])))
    dtz_dpi = np.hstack((dtz_dpi, np.zeros([nrows, 1, ncols]))).swapaxes(1, 2)

    # Derivatives with respect to the WATER SATURATION
    dtx_dswi = dt_dsw.swapaxes(0, 1)
    dtx_dswi = upwind(headx[:, :-1, :], dtx_dswi[:, :-1, :], headx[:, 1:, :], zerox[:, 1:, :])

    dty_dswi = dt_dsw
    dty_dswi = upwind(heady[:, :-1, :], dty_dswi[:, :-1, :], heady[:, 1:, :], zeroy[:, 1:, :])

    dtz_dswi = dt_dsw.swapaxes(1, 2)
    dtz_dswi = upwind(headz[:, :-1, :], dtz_dswi[:, :-1, :], headz[:, 1:, :], zeroz[:, 1:, :])

    # Complete the transmissibilities with zeros
    dtx_dswi = np.hstack((dtx_dswi, np.zeros([ncols, 1, nlayers]))).swapaxes(0, 1)
    dty_dswi = np.hstack((dty_dswi, np.zeros([nrows, 1, nlayers])))
    dtz_dswi = np.hstack((dtz_dswi, np.zeros([nrows, 1, ncols]))).swapaxes(1, 2)

    # Organize the arrays in lists
    DT_DPnext = [dtx_dp, dty_dp, dtz_dp]
    DT_DSWnext = [dtx_dsw, dty_dsw, dtz_dsw]
    DT_DPi = [dtx_dpi, dty_dpi, dtz_dpi]
    DT_DSWi = [dtx_dswi, dty_dswi, dtz_dswi]

    dft = {'dTdPnext': DT_DPnext, 'dTdSwnext':  DT_DSWnext, 'dTdPi':DT_DPi,'dTdSwi': DT_DSWi}

    return dft


def harmonic_density(grid, fluid, pressure):
    '''
    Returns the harmonic average of the density.
    Arguments:
        grid (object) : cartesianGrid
        fluid (object) : singleFluid
        pressure (np.array) : vector with the pressure for each cell.
    Returns:
        rhox, rhoy, rhoz (np.array) : density matrices for each dimension.
    '''
    # Averaging constant
    w = 0.5
    nrows, ncols, nlayers = grid.dim
    # The pressures in matrix form
    p = pressure.reshape(grid.dim, order='F')
    # Pressure average
    # in x
    px = np.add(
        w * p.swapaxes(0, 1)[:, 0:-1, :], (1 - w) * p.swapaxes(0, 1)[:, 1:, :])
    # in y
    py = np.add(w * p[:, 0:-1, :], (1 - w) * p[:, 1:, :])
    # in z
    pz = np.add(
        w * p.swapaxes(1, 2)[:, 0:-1, :], (1 - w) * p.swapaxes(1, 2)[:, 1:, :])
    # Density
    rhox = fluid.rho(px)
    rhoy = fluid.rho(py)
    rhoz = fluid.rho(pz)
    # Complete density matrix with zeros
    rhox = np.hstack((rhox, np.zeros([ncols, 1, nlayers]))).swapaxes(0, 1)
    rhoy = np.hstack((rhoy, np.zeros([nrows, 1, nlayers])))
    rhoz = np.hstack((rhoz, np.zeros([nrows, 1, ncols]))).swapaxes(1, 2)
    
    return rhox, rhoy, rhoz


def dPc(grid, fluid, sw):
    '''
    Derivative of capillaty pressure
    Arguments:
        grid: cartesianGrid
        fluid: blackOil or another fluid with a capillary pressure attribute.
        saturation (np.array): water saturation
    Returns:
        dPcx, dPcy, dPcz (np.array)
    '''
    nrows, ncols, nlayers = grid.dim
    eps_sw = 0.001
    # Capillary pressure of each cell in matrix form
    dpc = (fluid.pc(sw + eps_sw) - fluid.pc(sw))/eps_sw
    dpc = dpc.reshape(grid.dim, order ='F')
    # Capillary pressure in each dimension
    dpcx = dpc.swapaxes(0,1)[:, :-1, :]
    dpcy = dpc[:, :-1, :]
    dpcz = dpc.swapaxes(1,2)[:, :-1, :]
    # Complete the transmissibilities with zeros
    dpcx = np.hstack((dpcx, np.zeros([ncols, 1, nlayers]))).swapaxes(0, 1)
    dpcy = np.hstack((dpcy, np.zeros([nrows, 1, nlayers])))
    dpcz = np.hstack((dpcz, np.zeros([nrows, 1, ncols]))).swapaxes(1, 2)
    # This is incomplete. We have to upwind the capillary pressure as well.
    return dpcx, dpcy, dpcz


def derivatives(rock, fluid, p_old, p_new, sw_old, sw_new):
    '''Calculates the numerical derivatives of porosity,
        capillary pressure, and of the reciprocal of
        the formation volume factor.
    Arguments:
        rock: Rock
        fluid: blackOil
        p_old, p_new (np.array) : previous and current timestep oil pressure values.
        sw_old, sw_new (np.array) : the same as for pressure but for water saturation.
    Returns:
        d_poro: derivative of porosity
        d_pc: derivative of capillary pressure
        d_bo: derivative of bo
        d_bw: derivative of bw
    '''
    eps_sw = 1E-6
    eps_p = 10 * u.psi

    pressure = p_new
    saturation = sw_new
    d_bw = (1 / fluid.waterphase.fvf(pressure + eps_p) - 1 / fluid.waterphase.fvf(pressure)) / eps_p
    d_bo = (1 / fluid.oilphase.fvf(pressure + eps_p) - 1 / fluid.oilphase.fvf(pressure)) / eps_p
    d_poro = (rock.porofunc(pressure + eps_p) - rock.porofunc(pressure) ) / eps_p
    d_pc = (fluid.pc(saturation + eps_sw) - fluid.pc(saturation)) / eps_sw
    return d_bw, d_bo, d_poro, d_pc


def accumTermTwoPhase(grid, rock , fluid, p_old, p_new, sw_old, sw_new ):
    '''
    Arguments:
        grid: cartesianGrid
        rock: rock
        fluid : blackOil
        p_old, p_new (np.array) : previous and current timestep oil pressure values.
        sw_old, sw_new (np.array) : the same but for water saturation.
    Returns:
        D (sparse) : block diagonal matrix, each block is 2 x 2.
    '''
    d_bw, d_bo, d_poro, d_pc = derivatives(rock, fluid, p_old, p_new, sw_old, sw_new)
    
    poro_new = rock.porofunc(p_new)
    fvfo_new = fluid.oilphase.fvf(p_new)
    fvfw_new = fluid.waterphase.fvf(p_new)

    poro_old = rock.porofunc(p_old)
    fvfo_old = fluid.oilphase.fvf(p_old)
    fvfw_old = fluid.waterphase.fvf(p_old)

    # Format of the blocks
    #[d11   d12     0       0   ...]
    #[d21   d22     0       0   ...]
    #[0     0       d11     d12 ...]
    #[0     0       d21     d22 ...]
 
    d11 = sw_old * (1 / fvfw_old * d_poro + poro_new * d_bw) 
    d12 = poro_new / fvfw_new - (sw_old * poro_old * d_bw * d_pc)
    d21 = (1 - sw_old) * (1/fvfo_old * d_poro + poro_new * d_bo)
    d22 = (-1) * poro_new/fvfo_new

    d11 = d11.reshape([-1,])
    d12 = d12.reshape([-1,])
    d21 = d21.reshape([-1,])
    d22 = d22.reshape([-1,])
    
    # Insert the blocks and multiply by cell volume.
    D = insertMainDiagonals(grid, d11, d12, d21, d22) * grid.cellvolume
        
    return D


def derivative_accumTermTwoPhase(grid, rock , fluid, p_old, p_new, sw_old, sw_new ):
    '''
    Derivative of the accumulation term 
    Arguments:
        grid: cartesianGrid
        rock: rock
        fluid : blackOil
        p_old, p_new (np.array) : previous and current timestep oil pressure values.
        sw_old, sw_new (np.array) : the same but for water saturation.
    Returns:
        D (sparse) : block diagonal matrix, each block is 2 x 2.
    '''
    d_bw, d_bo, d_poro, d_pc = derivatives(rock, fluid, p_old, p_new, sw_old, sw_new)

    dp = (p_new - p_old)

    d11 = 2 * sw_new * d_poro/d_bw 
    d12 = np.zeros_like(d11)
    d21 = 2 * (1 - sw_new) * d_poro / d_bo
    d22 = np.zeros_like(d21)

    d11 = d11.reshape([-1,])
    d12 = d12.reshape([-1,])
    d21 = d21.reshape([-1,])
    d22 = d22.reshape([-1,])
    
    # Insert the blocks and multiply by cell volume.
    dD = insertMainDiagonals(grid, d11, d12, d21, d22) * grid.cellvolume
        
    return dD


def insertMainDiagonals(grid, a11,a12,a21,a22):
    nrows, ncols, nlayers = grid.dim 
    # Number of liquid phases
    n = 2
    k = grid.cellnumber * n
    # Main diagonal
    ix0 = np.arange(0, k, n) #arows
    iy0 = np.arange(1, k, n) #acols
    #T[arows, arows] = a11
    #T[arows, acols] = a12
    #T[acols, arows] = a21
    #T[acols, acols] = a22
    xcoor = np.concatenate((ix0, ix0, iy0, iy0))
    ycoor = np.concatenate((ix0, iy0, ix0, iy0))
    data = np.concatenate((a11, a12, a21, a22))
    T = sparse.coo_matrix((data, (xcoor, ycoor)), (k, k))

    return T.tocsr()


def insertBlockDiagonals(grid, A, B, C, D):
    '''
    Arguments:
        A, B, C, D (scipy.sparse): diagonal matrices with the format given in 
        the function "insertDiagonals".
    Returns:
        T(scipy.sparse): block matrix with the same number of diagonals as the input, 
        but with blocks.

        Block format:
        [A B] Each block has an element of each matrix.
        [D C]
    '''
    nrows, ncols, nlayers = grid.dim  
    # Number of liquid phases
    n = 2
    k = grid.cellnumber * n

    # Main diagonals
    a11 = A.diagonal()
    a12 = B.diagonal()
    a21 = C.diagonal()
    a22 = D.diagonal()

    ix0 = np.arange(0, k, n)
    iy0 = np.arange(1, k, n)

    xcoor = np.concatenate((ix0, ix0, iy0, iy0))
    ycoor = np.concatenate((ix0, iy0, ix0, iy0))
    data = np.concatenate((a11, a12, a21, a22))

    # Select the other diagonals to be inserted into the matrix
    if nrows > 1:

        brows = np.arange(1, grid.cellnumber, 1)
        bcols = np.arange(0 , grid.cellnumber - 1 , 1)
        
        b11 = np.array(A[brows, bcols]).reshape([-1,])
        b12 = np.array(B[brows, bcols]).reshape([-1,]) 
        b21 = np.array(C[brows, bcols]).reshape([-1,])
        b22 = np.array(D[brows, bcols]).reshape([-1,])
        
        c11 = np.array(A[bcols, brows]).reshape([-1,])
        c12 = np.array(B[bcols, brows]).reshape([-1,]) 
        c21 = np.array(C[bcols, brows]).reshape([-1,])
        c22 = np.array(D[bcols, brows]).reshape([-1,])
        
        ix1 = np.arange(n, k , n)
        ix2 = np.arange(n + 1, k , n)
        iy1 = np.arange(0 , (grid.cellnumber - 1) * n , n)
        iy2 = np.arange(1 , (grid.cellnumber - 1) * n , n)

        xcoor = np.concatenate((xcoor, ix1, ix1, ix2, ix2, iy1, iy1, iy2, iy2))
        ycoor = np.concatenate((ycoor, iy1, iy2, iy1, iy2, ix1, ix2, ix1, ix2))
        data = np.concatenate((data, b11, b12, b21, b22, c11, c12, c21, c22))

    if ncols > 1:

        drows = np.arange(nrows, grid.cellnumber, 1)
        dcols = np.arange(0 , grid.cellnumber - nrows , 1)

        d11 = np.array(A[drows, dcols]).reshape([-1,])
        d12 = np.array(B[drows, dcols]).reshape([-1,]) 
        d21 = np.array(C[drows, dcols]).reshape([-1,])
        d22 = np.array(D[drows, dcols]).reshape([-1,])

        e11 = np.array(A[dcols, drows]).reshape([-1,])
        e12 = np.array(B[dcols, drows]).reshape([-1,])
        e21 = np.array(C[dcols, drows]).reshape([-1,])
        e22 = np.array(D[dcols, drows]).reshape([-1,])

        ix3 = np.arange(nrows * n, k, n)
        ix4 = np.arange(nrows * n + 1, k, n)
        iy3 = np.arange(0, (grid.cellnumber - nrows) * n , n)
        iy4 = np.arange(1, (grid.cellnumber - nrows) * n , n)

        xcoor = np.concatenate((xcoor, ix3, ix3, ix4, ix4, iy3, iy3, iy4, iy4))
        ycoor = np.concatenate((ycoor, iy3, iy4, iy3, iy4, ix3, ix4, ix3, ix4))
        data = np.concatenate((data, d11, d12, d21, d22, e11, e12, e21, e22))

    if nlayers > 1:

        frows = np.arange( nrows * ncols, grid.cellnumber, 1)
        fcols = np.arange(0 , grid.cellnumber - nrows * ncols, 1)

        f11 = np.array(A[frows, fcols]).reshape([-1,])
        f12 = np.array(B[frows, fcols]).reshape([-1,])
        f21 = np.array(C[frows, fcols]).reshape([-1,])
        f22 = np.array(D[frows, fcols]).reshape([-1,])

        g11 = np.array(A[fcols, frows]).reshape([-1,])
        g12 = np.array(B[fcols, frows]).reshape([-1,])
        g21 = np.array(C[fcols, frows]).reshape([-1,])
        g22 = np.array(D[fcols, frows]).reshape([-1,])

        ix5 = np.arange( nrows * ncols * n, k, n)
        ix6 = np.arange( nrows * ncols * n + 1, k, n)
        iy5 = np.arange(0, (grid.cellnumber - nrows * ncols) * n, n)
        iy6 = np.arange(1, (grid.cellnumber - nrows * ncols) * n, n)

        xcoor = np.concatenate((xcoor, ix5, ix5, ix6, ix6, iy5, iy5, iy6, iy6))
        ycoor = np.concatenate((ycoor, iy5, iy6, iy5, iy6, ix5, ix6, ix5, ix6))
        data = np.concatenate((data, f11, f12, f21, f22, g11, g12, g21, g22))

    T = sparse.coo_matrix((data, (xcoor, ycoor)), (k, k))
         
    return T.tocsr()


def potential(grid, pressure, rho):
    '''Returns the pressure potential of each cell.
    Arguments:
        grid: cartesianGrid
        rho (np.array):  fluid density.
        pressure (np.array) : cell pressure 
    Returns:
        potential (np.array):  array with the same shape as rho and pressure
    potential = pressure - ( rho * g * depth)
    potential = pressure - gravityTerm
    '''
    # check that pressure and rho have same shape
    if pressure.shape == rho.shape:
        pass
    else:
        raise Exception('Shape mismatch')
    # Not sure if the sign should be a (+) or a (-).
    dim = pressure.shape
    return pressure - constants.g * rho * grid.depth.reshape(dim, order='F')


def upwind(potential_i, x_i, potential_next, x_next):
    '''For each cell, it compares the pressure head (p)  with that of its neighbor.
        Then returns the value of "x" from the cell that has the highest "p".
    Arguments:
        potential_i (np.array): pressure head of the cell.
        x_i (np.array): "x" property of the cell.
        potential_next (np.array): pressure head of the neighboring cell.
        x_next (np.array): "x" property of the neighboring cell.
    Returns:
        filtered (np.array): filtered vector with the appropriate value of "x".
    '''
    filtered = np.where(potential_next > potential_i, x_next, x_i)

    return filtered


class LaggingCoefficientsTwoPhase:
    '''Lagging Coefficients Solver.
    Attributes
        grid: cartesianGrid
        rock: Rock
        wells: Wells
        fluid: blackOil
        p_init (np.array) vector with the initial pressure for each cell
        gravity (bool): True, consider gravity effects (default).
            False, neglect gravity effects.
        sources (np.array): source term for each block.
                (-) is injection, (+) is production.
        solve (function) : advances the simulation
    '''

    def __init__(self, grid, rock, fluid, wells, source,
        p_init, sw_init,  boundary, gravity=True):
        ''' Initilize the solver.
        Arguments:
        grid: cartesianGrid
        rock: Rock
        wells: Wells
        fluid: blackOil
        wells
        source
        p_init (np.array) vector with the initial oil pressure for each cell
        sw_init (np.array) vector with the initial water saturation for each cell
        gravity (bool): True, consider gravity effects (default). 
            False, neglect gravity effects.
        boundary : currently not supported. ONLY NO FLOW BOUNDARY. 
        sources (np.array)  vector (cellnumber x 1 ) with the source term for each block.
            (-) is injection, (+) is production. 
        '''
        self.grid = grid
        self.rock = rock
        self.fluid = fluid
        self.wells = wells
        self.source = source
        self.p_init = p_init
        self.sw_init = sw_init
        self.gravity = gravity
        self.boundary = boundary

    def solve(self, schedule, max_inner_iter=5, tol=1E-6, 
        linear_solver=None, ATS = False):
        '''Run the solver for the timesteps in the schedule.
        Arguments:
            schedule (object) : schedule
            linear_solver (function) : f(A, b) = x
            max_inner_iter (int) : maximum number of iterations for each time step.
            tol (float) : relative error tolerance. 
            ATS (bool) : adaptive time step. False (default).
        Returns:
            pressure (np.array) : [ cellnumber x (timesteps + 1) ]
                with the pressure solution for each cell for each time step.
            saturation (np.array) : the same as "pressure" but for the water saturation. 
            well_solution (dictionary) : dictionary with the solution for each well.
        '''
        # --- OUTSIDE THE TIME LOOP ---

        # Empty-matrix to store results. One extra column to save initial conditions.
        results = {'p_oil':self.p_init.reshape([-1,1]) ,
                    'sw': self.sw_init.reshape([-1,1])}
        
        # Compute the geometric part of the transmissibility.
        gtx, gty, gtz = geometricTrans(self.grid, self.rock)
        # Compute the difference in depth.
        dzx, dzy, dzz = dh(self.grid, self.grid.depth)
        DZ = dh_matrix(self.grid, dzx, dzy, dzz)

        # Start with the initial conditions
        p_old, sw_old = self.p_init, self.sw_init
        p_guess, sw_guess = p_old, sw_old

        # Time steps
        timesteps = schedule.timesteps
        maxtime = schedule.tottime

        # --- INSIDE THE TIME LOOP ---
        k = 0
        while k < timesteps.size and timesteps[:k-1].sum() < maxtime:
            # Print message to show progress of simulation.
            message = 'Solving timestep {} / {}. \t {:.3f} days'  .format(
                k + 1, timesteps.size, timesteps[:k].sum()/u.day)
            print(message)
            # Select timestep
            dt = timesteps[k]  

            # Inner loop to recompute pressure and properties.
            for innerk in range(0, max_inner_iter):

                # Print message to show progress of simulation.
                message = '\t Inner loop {} / {}'.format(
                    innerk + 1, max_inner_iter)
                print(message)

                # Compute well rate and well transmissibilities.
                self.wells.update_wells(p_guess, sw_guess)
                # Compute the fluid part of the transmissibility.
                # Water phase
                ftx_water, fty_water, ftz_water,  = fluidTransTwoPhase(
                    self.grid, self.fluid.waterphase, p_guess, sw_guess)

                ftx_oil, fty_oil, ftz_oil,  = fluidTransTwoPhase(
                    self.grid, self.fluid.oilphase, p_guess, sw_guess)

        
                rhox_water, rhoy_water, rhoz_water = harmonic_density(self.grid, self.fluid.waterphase, p_guess)
                rhox_oil, rhoy_oil, rhoz_oil = harmonic_density(self.grid, self.fluid.oilphase, p_guess)
                # Multiply the geometric part and the fluid part of the
                # transmissibilities.
                tx_water = gtx * ftx_water
                ty_water = gty * fty_water
                tz_water = gtz * ftz_water

                tx_oil = gtx * ftx_oil
                ty_oil = gty * fty_oil
                tz_oil = gtz * ftz_oil

                # Capillary pressure 
                dpcx, dpcy, dpcz = dPc(self.grid, self.fluid, sw_guess)
                tx_pc = - tx_water * dpcx
                ty_pc = - ty_water * dpcy
                tz_pc = - tz_water * dpcz

                # Assemble the transmissibility matrix.
                # T_inc is the transmissibility matrix without the main diagonal
                T_water, T_inc_water = transTerm(self.grid, tx_water, ty_water, tz_water)
                T_oil, T_inc_oil = transTerm(self.grid, tx_oil, ty_oil, tz_oil)
                T_pc, T_inc_pc = transTerm(self.grid, tx_pc, ty_pc, tz_pc)
                
                empty_matrix = sparse.csr_matrix((self.grid.cellnumber, self.grid.cellnumber))
                T = insertBlockDiagonals(self.grid, T_water, T_pc, T_oil, empty_matrix )
  
                # Compute the accumulation term.
                B = accumTermTwoPhase(self.grid, self.rock , self.fluid, p_old, p_guess, sw_old, sw_guess )
                
                # Compute the gravity term.
                # If gravity == True
                if self.gravity:
                    # Gamma matrix
                    gamma_matrix_oil = gamma(self.grid, rhox_oil, rhoy_oil, rhoz_oil)
                    gamma_matrix_water = gamma(self.grid, rhox_water, rhoy_water, rhoz_water)

                    g_water = gravityTerm(self.grid, DZ, gamma_matrix_water, T_inc_water)
                    g_oil = gravityTerm(self.grid, DZ, gamma_matrix_oil, T_inc_oil)
                    g = np.hstack((g_water, g_oil)).reshape([-1, 1])
                # If gravity == False
                else:
                    g = np.zeros([self.grid.cellnumber * self.fluid.phasenumber, 1])

                # Apply the well conditions only if there are any wells.
                if len(self.wells.wells) > 0:
                    # Update the source term with the well rate.
                    source = self.wells.update_source_term(self.source)
                    # Update the transmissibility term with the wells.
                    # Doesnt work for two phase
                    # T = self.wells.update_trans(self.grid, T_inc)
                else:
                    source = self.source

                # Put the terms in the form A * x = b
                
                psw_old = np.hstack((p_old.reshape([-1, 1]), sw_old.reshape([-1, 1]))).reshape([-1,1])
                psw_guess = np.hstack((p_guess.reshape([-1, 1]), sw_guess.reshape([-1, 1]))).reshape([-1,1])
                # gravity + (sources & sinks) + accumulation
                rhs = g + source - (1 / dt) * B.dot(psw_old.reshape([-1, 1]))
                LHS = T - B / dt

                
                # Apply boundary conditions. They override all the other
                # conditions.
                #A, b = apply_boundary(
                #    self.grid, LHS, rhs, self.boundary, self.wells)
                # SOLVE THE SYSTEM OF EQUATIONS
                # Solve with sparse solver. Is faster, uses less memory.
                psw_new = linalg.spsolve(LHS, rhs)

                # Break loop if relative error is less than the tolerance
                p_new = psw_new.reshape([self.grid.cellnumber, 2])[:, 0]
                
                relative_error = (np.linalg.norm(psw_guess - psw_new))/ np.linalg.norm(psw_guess)
                if relative_error < tol:
                    
                    p_guess = psw_new.reshape([self.grid.cellnumber, 2])[:, 0]
                    sw_guess = psw_new.reshape([self.grid.cellnumber, 2])[:, 1]
                    break
                else:
                    p_guess = psw_new.reshape([self.grid.cellnumber, 2])[:, 0]
                    sw_guess = psw_new.reshape([self.grid.cellnumber, 2])[:, 1]


            # If convergence was not reached, decrease time step
            if relative_error > tol and  ATS:
                head = timesteps[:k]
                # New, smaller time step
                newstep = np.array([dt / 5,])
                tail = timesteps[k:]
                timesteps = np.concatenate((head, newstep, tail))
                # go back to the top
                continue

            # Save results
            p_new = psw_new.reshape([self.grid.cellnumber, 2])[:, 0]
            sw_new = psw_new.reshape([self.grid.cellnumber, 2])[:, 1]

            results['p_oil']= np.hstack((results['p_oil'], p_new.reshape([-1,1])))
            results['sw'] = np.hstack((results['sw'], sw_new.reshape([-1,1])))

            # Save rate for each well
            for wi in range(0, len(self.wells.wells)):
                #Water rate
                self.wells.wells[wi]['rate_sol'][0].append(
                    self.wells.wells[wi]['rate'][0][-1])
                
                # Oil rate
                self.wells.wells[wi]['rate_sol'][1].append(
                    self.wells.wells[wi]['rate'][1][-1])

            # Set old pressure and saturation as the new one
            p_old = p_new
            sw_old = sw_new

            # Advance one time step
            k = k + 1

        # New schedule.with the refined time steps..
        schedule = Schedule(timesteps[:k])

        return results, self.wells.wells, schedule


class ImplicitTwoPhase:
    ''' Fully Implicit Solver, with Analitic Jacobian and residual.
    Attributes
        grid: cartesianGrid
        rock: Rock
        wells: Wells
        fluid: blackOil
        p_init (np.array) vector with the initial pressure for each cell
        gravity (bool): True, consider gravity effects (default).
            False, neglect gravity effects.
        sources (np.array): source term for each block.
                (-) is injection, (+) is production.
        solve (function) : advances the simulation
    '''

    def __init__(self, grid, rock, fluid, wells, source,
        p_init, sw_init,  boundary, gravity=True):
        ''' Initilize the solver.
        Arguments:
        grid: cartesianGrid
        rock: Rock
        wells: Wells
        fluid: blackOil
        wells: Wells
        sources (np.array)  vector (cellnumber x 1 ) with the source term for each block.
            (-) is injection, (+) is production.
        p_init (np.array) vector with the initial oil pressure for each cell
        sw_init (np.array) vector with the initial water saturation for each cell
        gravity (bool): True, consider gravity effects (default). 
            False, neglect gravity effects.
        boundary : currently not supported. ONLY NO FLOW BOUNDARY. 
        '''
        self.grid = grid
        self.rock = rock
        self.fluid = fluid
        self.wells = wells
        self.source = source
        self.p_init = p_init
        self.sw_init = sw_init
        self.gravity = gravity
        self.boundary = boundary

    def solve(self, schedule, max_iter=10, tol=1E-12, 
        linear_solver=None, ATS = False):
        '''Run the solver for the timesteps in the schedule.
        Arguments:
            schedule (object) : schedule
            linear_solver (function) : f(A, b) = x
            max_inner_iter (int) : maximum number of iterations for each time step.
            tol (float) : relative error tolerance. 
            ATS (bool) : adaptive time step. False (default).
        Returns:
            results (dict): {'p_oil': pressure, 'sw': saturation}
                pressure (np.array) : [ cellnumber x (timesteps + 1) ]
                with the pressure solution for each cell for each time step.
                saturation (np.array) : the same as "pressure" but for the water saturation. 
            well_solution (dictionary) : dictionary with the solution for each well.
            schedule (Schedule) :  final schedule after time step refinement.
        '''
        # --- OUTSIDE THE TIME LOOP ---

        # Empty-matrix to store results. One extra column to save initial conditions.
        results = {'p_oil':self.p_init.reshape([-1,1]) ,
                    'sw': self.sw_init.reshape([-1,1])}

        # List to save the information of the linear solver 
        info = []
        
        # Compute the geometric part of the transmissibility.
        gtx, gty, gtz = geometricTrans(self.grid, self.rock)
        # Compute the difference in depth.
        dzx, dzy, dzz = dh(self.grid, self.grid.depth)
        DZ = dh_matrix(self.grid, dzx, dzy, dzz)

        # Start with the initial conditions
        p_old, sw_old = self.p_init, self.sw_init
        psw_old = np.hstack((p_old.reshape([-1, 1]), sw_old.reshape([-1, 1]))).reshape([-1,1])
        psw_new = psw_old

        # Time steps
        timesteps = schedule.timesteps
        maxtime = schedule.tottime

        # --- INSIDE THE TIME LOOP ---
        k = 0
        k_ats = 0
        while k < timesteps.size and timesteps[:k-1].sum() < maxtime:
            # Print message to show progress of simulation.
            message = 'Solving timestep {:<2d} / {:<2d} \t {:.3f} days'.format(
                k+1, timesteps.size, timesteps[:k].sum()/u.day)
            #print(message)
            # Select timestep
            dt = timesteps[k]  

            # We define a function to handle the residual.
            def res_Jacbn(psw_new):
                ''' Returns the residual and Jacobian as a function of pressure.
                Arguments: 
                    psw_new(np.array) : pressure & saturation vector
                Returns:
                    residual (np.array)
                    Jacobian (sparse.matrix)
                ''' 
                # Separate pressure and saturation
                p_new = psw_new.reshape([self.grid.cellnumber, 2])[:, 0]
                sw_new = psw_new.reshape([self.grid.cellnumber, 2])[:, 1]

                #print(np.amax(p_new))
                #print(np.amin(p_new))
                #print(np.amax(sw_new))
                #print(np.amin(sw_new))

                # Compute well rate and well transmissibilities.
                self.wells.update_wells(p_new, sw_new)
                # Compute the fluid part of the transmissibility.
                # Water phase
                ftx_water, fty_water, ftz_water,  = fluidTransTwoPhase(
                    self.grid, self.fluid.waterphase, p_new, sw_new)

                ftx_oil, fty_oil, ftz_oil,  = fluidTransTwoPhase(
                    self.grid, self.fluid.oilphase, p_new, sw_new)

                # Multiply the geometric part and the fluid part of the
                # transmissibilities.
                tx_water = gtx * ftx_water
                ty_water = gty * fty_water
                tz_water = gtz * ftz_water

                tx_oil = gtx * ftx_oil
                ty_oil = gty * fty_oil
                tz_oil = gtz * ftz_oil

                # Capillary pressure 
                # Note this is WRONG!! this should be the derivative of the capillary pressure
                # and it should be upwinded too
                dpcx, dpcy, dpcz = dPc(self.grid, self.fluid, sw_new)
                tx_pc = - tx_water * dpcx
                ty_pc = - ty_water * dpcy
                tz_pc = - tz_water * dpcz

                # Assemble the transmissibility matrix.
                # T_inc is the transmissibility matrix without the main diagonal
                T_water, T_inc_water = transTerm(self.grid, tx_water, ty_water, tz_water)
                T_oil, T_inc_oil = transTerm(self.grid, tx_oil, ty_oil, tz_oil)
                T_pc, T_inc_pc = transTerm(self.grid, tx_pc, ty_pc, tz_pc)
                
                empty_matrix = sparse.csr_matrix((self.grid.cellnumber, self.grid.cellnumber))
                
                T = insertBlockDiagonals(self.grid, T_water, T_pc, T_oil, empty_matrix)
            
                # Compute the accumulation term.
                # For the jacobian
                B = accumTermTwoPhase(self.grid, self.rock , self.fluid, p_old, p_new, sw_old, sw_new )
                dB = derivative_accumTermTwoPhase(self.grid, self.rock , self.fluid, p_old, p_new, sw_old, sw_new )
                db = np.array(dB.dot(psw_new - psw_old)).reshape([-1,])
                dB = sparse.diags(db, 0, format = 'csr')

                # Compute the gravity term.
                # If gravity == True
                if self.gravity:

                    rhox_water, rhoy_water, rhoz_water = harmonic_density(self.grid, self.fluid.waterphase, p_new)
                    rhox_oil, rhoy_oil, rhoz_oil = harmonic_density(self.grid, self.fluid.oilphase, p_new)
                    # Gamma matrix
                    gamma_matrix_oil = gamma(self.grid, rhox_oil, rhoy_oil, rhoz_oil)
                    gamma_matrix_water = gamma(self.grid, rhox_water, rhoy_water, rhoz_water)

                    g_water = gravityTerm(self.grid, DZ, gamma_matrix_water, T_inc_water)
                    g_oil = gravityTerm(self.grid, DZ, gamma_matrix_oil, T_inc_oil)
                    g = np.hstack((g_water, g_oil)).reshape([-1, 1])
                # If gravity == Falsejacobian
                else:
                    g = np.zeros([self.grid.cellnumber * self.fluid.phasenumber, 1])

                # Apply the well conditions only if there are any wells.
                if len(self.wells.wells) > 0:
                    # Update the source term with the well rate.
                    source = self.wells.update_source_term(self.source)
                    # Update the transmissibility term with the wells.
                    # Doesnt work for two phase
                    # T = self.wells.update_trans(self.grid, T_inc)
                else:
                    source = self.source
                               
                #Residual
                r = T.dot(psw_new) - (1/dt) * B.dot(psw_new - psw_old) - source - g                
                # Jacobian of residual
                J = jacobianTwoPhase(self.grid, self.rock, self.fluid, self.wells, B/dt, dB/dt, gtx, gty, gtz, 
                    T_inc_water, T_inc_oil, p_new, sw_new)

                return r, J

            # Solve with Newton-Raphson Method
            psw_new, iter_number, error, solver_info = NewRaph(res_Jacbn, psw_new, tol, max_iter, linear_solver)                        

            # If convergence was not reached, decrease time step
            if error > tol and  ATS:
                if k_ats == 0 :
                    head = timesteps[:k]
                    # New, smaller time step
                    newstep = np.array([dt / 5,])
                    tail = timesteps[k:]
                    timesteps = np.concatenate((head, newstep, tail))
                else:
                    timesteps[k] = dt / 5
                # go back to the top
                continue

            # Save results
            p_new = psw_new.reshape([self.grid.cellnumber, 2])[:, 0]
            sw_new = psw_new.reshape([self.grid.cellnumber, 2])[:, 1]

            results['p_oil']= np.hstack((results['p_oil'], p_new.reshape([-1,1])))
            results['sw'] = np.hstack((results['sw'], sw_new.reshape([-1,1])))

            info.append(solver_info)

            # Save rate for each well
            for wi in range(0, len(self.wells.wells)):
                #Water rate
                self.wells.wells[wi]['rate_sol'][0].append(
                    self.wells.wells[wi]['rate'][0][-1])
                
                # Oil rate
                self.wells.wells[wi]['rate_sol'][1].append(
                    self.wells.wells[wi]['rate'][1][-1])

            # Set old pressure and saturation as the new one
            p_old = p_new
            sw_old = sw_new
            psw_old = np.hstack((p_old.reshape([-1, 1]), sw_old.reshape([-1, 1]))).reshape([-1,1])

            # Advance one time step
            k = k + 1
            # Reset
            k_ats = 0

        # New schedule.with the refined time steps..
        schedule = Schedule(timesteps[:k])

        return results, self.wells.wells, schedule, info


def NewRaph( f, x0,  tol , max_iter, linear_solver):
    '''
    Newton-Raphson method to obtain the root of the function.
    Arguments:
        f (function) :  f(x) = ( residual, jacobian ) 
        x0 (np.array) : vector with first guess for x
        tol (float) : stopping criteria. relative error.
        max_iter (int) : stopping criteria. Maximum number of iterations.
    Returns :
        x (np.array) : vector that solves f(x) = 0
        k (int) : number of iterations
        error (float) : relative error
        info
    '''
    # Algorithm
    # Shift iteration counter by +1 to present results in a more natural way.
    solver_info = []
    info = 0
    for k in range(0, max_iter):
        
        fx0, J = f(x0)
        
        # Select for iterative solver
        #M = sparse.linalg.spilu(J.tocsc())
        #M2 = sparse.linalg.LinearOperator(M.shape,M.solve)
        #dx, info = sparse.linalg.bicgstab(J, fx0, fx0, M = M2)
        # Select for direct solver
        dx = sparse.linalg.spsolve(J, fx0)

        x = x0 -  dx.reshape([-1,1])
        
        #Save solver info
        solver_info.append(info)
        # If relative error is less than tolerance, break loop
        error = np.linalg.norm(x-x0)/np.linalg.norm(x)
        # Print message to show status
        message = ' \t Newton-Raphson solver : {:2d}/{:2d}. Error: {:.2E}'.format(k + 1, max_iter, error)
        #print(message)
        if  error < tol:
            break
        else:
            x0 = x
    return x , k , error, solver_info


def accumVector(grid, rock, fluid, p, sw):
    ''' Accumulation term in vector form.
    For the implicit method.
    Arguments:
        grid: cartesianGrid
        rock: Rock
        fluid: blackOil
        p (np.array): pressure
        sw (np.array): saturation
    Returns:
        C (np.array): [cellnumber * phasenumber, 1]
    '''
    # Each block of the shape
    # [ C_oil  ]
    # [ C_water]
    poro = rock.porofunc(p)
    fvfo = fluid.oilphase.fvf(p)
    fvfw = fluid.waterphase.fvf(p)    
    # cw = grid.cellvolume * poro * sw / fvfw
    # co = grid.cellvolume * poro * (1 - sw) / fvfo
    cw = sw / fvfw * poro
    co = (1 - sw) / fvfo * poro
    C = np.hstack((co.reshape([-1,1]), cw.reshape([-1, 1]))).reshape([-1, 1])

    return C * grid.cellvolume


def jacobianTransTerm(grid, dTdW, gtx, gty, gtz):
    '''
    Returns the jacobian of the transmissibility, with respect
    to some property "W". 
    Arguments:
        dTdW (list) : [dtfx_dw, dtfy_dw, dtfz_dw]
    Returns:
        dTdW (scypy.sparse) : matrix 
    '''
    dtx_dw = gtx * dTdW[0]
    dty_dw = gty * dTdW[1]
    dtz_dw = gtz * dTdW[2]
    # Dimensions
    nrows, ncols, nlayers = grid.dim
    # Reshape the transmissibility to vector form
    dtx_dw = dtx_dw.reshape([grid.cellnumber, ], order='F')   
    dty_dw = dty_dw.reshape([grid.cellnumber, ], order='F')   
    dtz_dw = dtz_dw.reshape([grid.cellnumber, ], order='F')

    b = dtx_dw[:-1]
    c = dtx_dw[:-1]
    d = dty_dw[:-nrows]
    e = dty_dw[:-nrows]
    f = dtz_dw[:-nrows * ncols]
    g = dtz_dw[:-nrows * ncols]
    #Insert diagonals in the matrix. 
    dTdW_inc = insertDiagonals(grid, b, c, d, e, f, g, format='csr')
       
    return dTdW_inc
    

def jacobianTwoPhase(grid, rock, fluid, wells,  B, dB, gtx, gty, gtz, T_inc_water, T_inc_oil, p, sw):
    '''
    Assembles the analytic Jacobian of the residual function. 
    To use with the fully implicit method.
     Arguments:
        grid: cartesianGrid
        fluid : blackOil
        wells: Wells
        B (scipy.sparse) : accumulation matrix
        gtx, gty, gtz : vectors with the geometric part of the transmissibility
        T_inc_water, T_inc_oil (scipy.sparse) : transmissibility matrices for water and oil
        p (np.array): oil pressure vector
        sw (np.array): water saturation vector
    Returns:
        J (sparse.csr) : jacobian
    '''
    # This function was based on the equations in Subsection 9.6 of 
    # "Basic Applied Reservoir Simulation"

    # Delta pressure in each direction
    dpx, dpy, dpz = dh(grid, p)
    DP = dh_matrix(grid, dpx, dpy, dpz)
    # Derivative of the fluid part.
    dtf_water = derivative_fluidTransTwoPhase(grid, fluid.waterphase, p, sw)
    dtf_oil = derivative_fluidTransTwoPhase(grid, fluid.oilphase, p, sw)
    # Arrange all the derivatives in a dictionary, to manage them more easily
    dtf = {'water': dtf_water, 'oil': dtf_oil}
    # =============================================
    # Derivative of transmissibility for neighboring cells
    # =============================================
    # WATER PART
    dTdP_water = jacobianTransTerm(grid, dtf['water']['dTdPnext'], gtx, gty, gtz)
    dTdSw_water = jacobianTransTerm(grid, dtf['water']['dTdSwnext'], gtx, gty, gtz)
    # Complete the residuals
    dRdP_water = T_inc_water + DP.multiply(dTdP_water)
    dRdSw_water = DP.multiply(dTdSw_water)

    # OIL PART
    dTdP_oil = jacobianTransTerm(grid, dtf['oil']['dTdPnext'], gtx, gty, gtz)
    dTdSw_oil = jacobianTransTerm(grid, dtf['oil']['dTdSwnext'], gtx, gty, gtz)
    # Complete the residuals
    dRdP_oil = T_inc_oil + DP.multiply(dTdP_oil)
    dRdSw_oil = DP.multiply(dTdSw_oil)
    
    # Insert the elements of the four matrices in a block (multi)diagonal matrix.
    dR_next = insertBlockDiagonals(grid, dRdP_water, dRdSw_water, dRdP_oil, dRdSw_oil)

    # =============================================
    # Derivative of transmissibility for "i" cells
    # =============================================
    # WATER PART
    dTdPi_water = jacobianTransTerm(grid, dtf['water']['dTdPi'], gtx, gty, gtz)
    dTdSwi_water = jacobianTransTerm(grid, dtf['water']['dTdSwi'], gtx, gty, gtz)
    # Complete the residuals
    dRdPi_water = DP.multiply(dTdPi_water) - T_inc_water 
    dRdSwi_water = DP.multiply(dTdSwi_water)

    # OIL PART
    dTdPi_oil = jacobianTransTerm(grid, dtf['oil']['dTdPi'], gtx, gty, gtz)
    dTdSwi_oil = jacobianTransTerm(grid, dtf['oil']['dTdSwi'], gtx, gty, gtz)
    # Complete the residuals
    dRdPi_oil = DP.multiply(dTdPi_oil) - T_inc_oil 
    dRdSwi_oil = DP.multiply(dTdSwi_oil)

    #plotMatrix(dTdP_water)
    #plotMatrix(dTdPi_water)
    #plotMatrix(dTdSw_water)
    #plotMatrix(dTdSwi_water)

    #plotMatrix(dTdSw_oil)
    #plotMatrix(dTdSwi_oil)
    #plotMatrix(dTdP_oil)
    #plotMatrix(dTdPi_oil)

    d11 = dRdPi_water.sum(axis =1)
    d12 = dRdSwi_water.sum(axis =1)
    d21 = dRdPi_oil.sum(axis = 1)
    d22 = dRdSwi_oil.sum(axis = 1)
    # Transform to arrays and reshape 
    d11 = np.array(d11).reshape([-1,])
    d12 = np.array(d12).reshape([-1,])
    d21 = np.array(d21).reshape([-1,])
    d22 = np.array(d22).reshape([-1,])
    # ============================================
    # Rate and gravity derivatives
    # ============================================
    dQ = insertMainDiagonals(grid,wells.d_rate[0],wells.d_rate[1], wells.d_rate[2], wells.d_rate[3])
    # ===========================================
    # Begin construction of Jacobian 
    # ===========================================
    # Add main diagonal.
    J = insertMainDiagonals(grid, d11, d12, d21, d22)
    # Add the rest of the diagonals.
    J = J + dR_next
    # Add the accumulation , rate and gravity derivative
    J = J - dQ - B - dB
  
    return J.tocsr()


def main():
    ''' It is run if this module is run. '''

    print(' Two phase flow module.')


if __name__ == "__main__":
    main()
    sys.exit()
