'''
POROUS MEDIA FLOW SIMULATOR
Author: Jaime Leal
Class: PETE 656
Instructor: Dr. Eduardo Gildin
Spring 2016

----------------------------------------------------------------------

This module contains the main classes and functions of the simulator.

The idea of how to structure this module was inspired by the 
Matlab Reservoir Simulation Toolbox.

----------------------------------------------------------------------
'''

import sys
import numpy as np
# from scipy.linalg import block_diag
from scipy import sparse
from scipy.sparse import linalg
from .units import *
from .simplots import *
# ---------------------------------------------------
u = UnitRegistry()
constants = Constants()
# ---------------------------------------------------


class Rock:
    '''
    Attributes:
        perm (numpy.array): permeability for each cell.
        poro (numpy.array): porosity for each cell.
        cr (float): rock compressibility.
        porofunc(function) : porosity as a function of pressure
        d_porofunc(function) : derivative of porofunc.
    '''
    
    def __init__(self, perm, poro, cr, porofunc ):
        '''
        Arguments:
            perm (numpy.array): permeability for each cell.
            poro (numpy.array): porosity for each cell.
            cr (float): rock compressibility.
            porofunc(function) : porosity as a function of pressure.
        '''

        if perm.size != poro.size:
            raise Exception(
                'Non-matching sizes of permeability and porosity fields.')
        else: pass

        self.perm = perm
        self.poro = poro
        self.cr = cr
        self.porofunc = porofunc
        self.d_porofunc = None


class singleFluid:
    ''' Fluid class for a single phase incompressible or compressible fluid.

    Attributes:
        miu (function): viscosity
        rho (function): density
        fvf (function): formation volume factor
        cf (float): compressibility
        d_miu (function): d(miu)/ d(p)
        d_rho (function): d(rho)/ d(p)
        d_fvf (function) : d(fvf)/ d(p)
        phasenumber (int) : number of phases
    '''

    def __init__(self, miu, rho, fvf, cf, kr = None):
        '''
        Arguments:
            miu (function): viscosity
            rho (function): density
            fvf (function): formation volume factor
            cf (float): compressibility
            kr (function) : relative permeability (only for use with blackOil)

        For an incompressible fluid the functions should 
        return a constant for each element in "x".
        ie : f(x) = 3 + 0*x.
        
        '''
        self.miu =  miu
        self.rho = rho
        self.fvf = fvf
        self.cf = cf
        self.kr = kr  
        self.d_miu = None
        self.d_rho = None
        self.d_fvf = None
        self.d_kr = None
        self.phasenumber = 1

class uniformCartesianGrid:
    '''  Uniform Cartesian grid with natural ordering of the cells. 
    Attributes:
        cellsize (np.array) : cell size in each dimension [Sx, Sy, Sz]
        cellnumber : number of cells in grid
        cellvolume : volume of an individual cell.
        dim (np.array) : number of cells in each dimension [Nx, Ny, Nz]
        depth (np.array) : depth of each cell
    '''

    def __init__(self, Nx, Ny, Nz, Sx, Sy, Sz, depth=None):
        '''
        Arguments:
            Nx, Ny, Nz (int): number of cells in x, y, and z directions.
            Sx, Sy, Sz (float): size of an individual cell.
            depth (np.array): vector with the depth of each grid block.
                The depth is a positive value.
        '''
        self.cellsize = np.array([Sx, Sy, Sz])
        self.cellnumber = Nx * Ny * Nz
        self.cellvolume = Sx * Sy * Sz
        self.dim = np.array([Nx, Ny, Nz])
        self.depth = depth


class LaggingCoefficients:
    '''Lagging Coefficients Solver.
    Attributes
        grid: cartesianGrid
        rock: Rock
        wells: Wells
        fluid: singleFluid
        p_init (np.array) vector with the initial pressure for each cell
        gravity (bool): True, consider gravity effects (default). 
            False, neglect gravity effects.
        sources (np.array)  vector (cellnumber x 1 ) with the source term for each block.
            (-) is injection, (+) is production. 
        solve (function) : advances the simulation
    '''

    def __init__(self, grid, rock, fluid, wells, source, p_init, boundary,  gravity = True):
        ''' Initialize the solver.
        Arguments:
        grid: cartesianGrid
        rock: Rock
        wells: Wells
        fluid: singleFluid
        p_init (np.array) vector with the initial pressure for each cell
        gravity (bool): True, consider gravity effects (default). 
            False, neglect gravity effects.
        sources (np.array)  vector (cellnumber x 1 ) with the source term for each block.
            (-) is injection, (+) is production. 
        '''
        self.grid = grid
        self.rock = rock
        self.fluid = fluid
        self.wells = wells
        self.source = source
        self.p_init = p_init
        self.gravity = gravity
        self.boundary = boundary
        

    def solve(self, schedule, max_inner_iter = 5 , tol = 1E-6, 
        linear_solver=None,  ATS = False ):
        '''Run the solver for the timesteps in the schedule.
        Arguments:
            schedule (object) : schedule
            max_inner_iter (int) : the maximum number of iterations inside the time loop. 
            tol (float) : relative error tolerance to stop the simulation.
            linear_solver (function) : f(A, b) = x 
            ATS (bool) : adaptive time step. False (default).
        Returns:
            pressure (np.array) : pressure for each time step [ cellnumber x (timesteps + 1) ]
            well_solution (dictionary) : dictionary with the solution for each well.
            schedule (Schedule) : final schedule, after being modified by the ATS. 
        '''
        # --- OUTSIDE THE TIME LOOP ---

        # Empty-matrix to store results. One extra column to save initial pressure.
        results = self.p_init.reshape([self.grid.cellnumber, 1])
        # Compute the geometric part of the transmissibility.
        gtx, gty, gtz = geometricTrans(self.grid, self.rock)
        # Compute the difference in depth for the gravity term.
        dzx, dzy, dzz = dh(self.grid, self.grid.depth)
        DZ = dh_matrix(self.grid, dzx, dzy, dzz)
        
        #Start with the initial pressure
        p_old = self.p_init
        p_guess = p_old

        # Time steps
        timesteps = schedule.timesteps
        maxtime = schedule.tottime

        # --- INSIDE THE TIME LOOP ---
        k = 0
        k_ats = 0
        while k < timesteps.size and timesteps[:k-1].sum() < maxtime:
            # Print message to show progress of simulation.
            message = 'Solving timestep {} / {}'.format(k+1, timesteps.size)
            print(message)
            #Select time step
            dt = timesteps[k]                 

            # Inner loop to recompute pressure and properties.
            for innerk in range(0, max_inner_iter):
                
                # Print message to show progress of simulation.
                message = '\t Inner loop {} / {}'.format(innerk+1, max_inner_iter)
                print(message)
                
                # Compute well rate and well transmissibilities.
                self.wells.update_wells(p_guess)
                # Compute the fluid part of the transmissibility.
                ftx, fty, ftz, rhox, rhoy, rhoz = fluidTrans(self.grid, self.fluid, p_guess)
                # Multiply the geometric part and the fluid part of the transmissibilities.
                tx = gtx * ftx
                ty = gty * fty
                tz = gtz * ftz
                # Assemble the transmissibility matrix. 
                # T_inc is the transmissibility matrix without the main diagonal
                T, T_inc = transTerm(self.grid, tx, ty, tz)
                
                # Compute the accumulation term with the previous pressure solution
                B = accumTerm(self.grid, self.rock , self.fluid, self.p_init, p_old, p_guess)
                
                # Compute the gravity term.
                # If gravity == True
                if self.gravity:
                    # Gamma matrix
                    gamma_matrix = gamma(self.grid, rhox, rhoy, rhoz)
                    g = gravityTerm(self.grid, DZ, gamma_matrix, T_inc)
                # If gravity == False
                else:
                    g = np.zeros([self.grid.cellnumber, 1])
              
                # Apply the well conditions only if there are any wells. 
                if len(self.wells.wells) > 0 :
                    # Update the source term with the well rate.
                    source = self.wells.update_source_term(self.source)
                    # Update the transmissibility term with the wells.
                    # T = self.wells.update_trans(self.grid , T_inc)
                else:
                    source = self.source
             
                # Put the terms in the form A * x = b 
                # gravity + (sources & sinks) + accumulation
                rhs =  g + source -  (1/dt) * B.dot(p_old.reshape([-1,1]))
                LHS = T - B / dt
                # Apply boundary conditions. They override all the other conditions.
                A , b = apply_boundary(self.grid, LHS, rhs, self.boundary, self.wells)
                # SOLVE THE SYSTEM OF EQUATIONS
                # Solve with sparse solver. Is faster, uses less memory.
                p_new = linalg.spsolve(A, b)

                # Break loop if relative error is less than the tolerance
                relative_error = (np.linalg.norm(p_guess-p_new)/np.linalg.norm(p_guess)) 
                if relative_error < tol:
                    p_guess = p_new 
                    break
                else:
                    p_guess = p_new


            # If convergence was not reached, decrease time step
            if relative_error > tol and  ATS:
                if k_ats == 0:
                    k_ats = 1
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
            results = np.hstack((results, p_new.reshape([self.grid.cellnumber, 1])))
            #Save rate for each well
            for wi in range(0,len(self.wells.wells)):
                self.wells.wells[wi]['rate_sol'].append( self.wells.wells[wi]['rate'][-1])

            # Set old pressure as the new one
            p_old = p_new

            # Advance one time step
            k = k + 1
            # Reset counter of ATS
            k_ats = 0 

        # New schedule.with the refined time steps..
        schedule = Schedule(timesteps[:k])

        return results, self.wells.wells, schedule


class ImplicitNumerical:
    '''Fully Implicit Solver. 
        With  NUMERICAL implementation for the residual and Jacobian.
    
    Arguments:
        grid: cartesianGrid
        rock: Rock
        wells: Wells
        fluid: singleFluid
        p_init (np.array) vector with the initial pressure for each cell
        gravity (bool): True, consider gravity effects (default). 
            False, neglect gravity effects.
        sources (np.array)  vector (cellnumber x 1 ) with the source term for each block.
            (-) is injection, (+) is production. 
        '''
    def __init__(self, grid, rock, fluid, wells, source, p_init, boundary,  gravity = True):
            ''' Initialize the fully implicit solver.'''
            self.grid = grid
            self.rock = rock
            self.fluid = fluid
            self.wells = wells
            self.source = source
            self.p_init = p_init
            self.gravity = gravity
            self.boundary = boundary
        

    def solve(self, schedule, max_iter=10, tol=1E-12, eps=1E-6, linear_solver=None):
        '''Run the solver for each timesteps in the schedule.
        Arguments:
            schedule (object) : schedule
            linear_solver (function) : f(A, b) = x (To be implemented in the future)
            max_iter (int) : the maximum number of iterations for the Newton-Raphson loop.
            tol (float) : tolerance for Newton-Raphson 
        Returns:
            pressure (np.array) : an array with dimensions [ cellnumber x (timesteps + 1) ]
        '''
        # --- OUTSIDE THE TIME LOOP ---

        # Empty-matrix to store results. One extra column to save initial pressure.
        results=np.empty([self.grid.cellnumber, schedule.timesteps.size + 1])
        results[:,0] = self.p_init

        # Compute the geometric part of the transmissibility.
        gtx, gty, gtz = geometricTrans(self.grid, self.rock)
        # Compute the difference in depth for the gravity term.
        dzx, dzy, dzz = dh(self.grid, self.grid.depth)
        DZ = dh_matrix(self.grid, dzx, dzy, dzz)
        
        #Start with the initial pressure
        p_old = self.p_init
        p_guess = p_old

        # --- INSIDE THE TIME LOOP ---
        for k in range(0,schedule.timesteps.size):
            # Print message to show progress of simulation.
            message = 'Solving timestep {} / {}'.format(k+1, schedule.timesteps.size)
            print(message)
            #Select time step
            dt = schedule.timesteps[k]        
            
           # We define a function to handle the residual.
            def residual(p_guess):
                ''' Returns the residual as a function of pressure.'''              
                self.wells.update_wells(p_guess)
                # Compute the fluid part of the transmissibility.
                ftx, fty, ftz, rhox, rhoy, rhoz = fluidTrans(self.grid, self.fluid, p_guess)
                # Multiply the geometric part and the fluid part of the transmissibilities.
                tx = gtx * ftx
                ty = gty * fty
                tz = gtz * ftz
                # Assemble the transmissibility matrix. 
                # T_inc is the transmissibility matrix without the main diagonal
                T, T_inc = transTerm(self.grid, tx, ty, tz)
                # Compute the accumulation term with the previous pressure solution
                B = accumTerm(self.grid, self.rock , self.fluid, self.p_init, p_old, p_guess)
                # Compute the gravity term.
                # If gravity == True
                if self.gravity:
                    # Gamma matrix
                    gamma_matrix = gamma(self.grid, rhox, rhoy, rhoz)
                    g = gravityTerm(self.grid, DZ, gamma_matrix, T_inc)
                # If gravity == False
                else:
                    g = np.zeros([self.grid.cellnumber, 1])
             
                # Apply the well conditions only if there are any wells. 
                if len(self.wells.wells) > 0 :
                    # Update the source term with the well rate.
                    source = self.wells.update_source_term(self.source)
                    # Update the transmissibility term with the wells.
                    T = self.wells.update_trans(self.grid , T_inc)
                else:
                    source = self.source
             
                # Put the terms in the form A * x = b 
                # gravity + sources & sinks + accumulation
                rhs =  g + source + (1/dt) * B.dot(p_guess.reshape([-1,1]) - p_old.reshape([-1,1]))
                LHS = T
                # Apply boundary conditions. They override all the other conditions.
                A , b = apply_boundary(self.grid, LHS, rhs, self.boundary, self.wells)
                
                return A.dot(p_guess.reshape([-1,1])) - b

            # Solve nonlinear equation with Newton-Rahpson Method
            p_new, iter_number, error = newton_raphson_numerical(residual, p_guess, tol = tol, max_iter = max_iter, eps=eps)
            print(error)
            p_guess = p_new   

            # Save results
            results[:,k + 1] = p_new.reshape([self.grid.cellnumber,])
            #Save rate for each well
            for wi in range(0,len(self.wells.wells)):

                self.wells.wells[wi]['rate_sol'].append( self.wells.wells[wi]['rate'][-1])

            # Set old pressure as the new one
            p_old = p_new

        return results, self.wells.wells
 

class ImplicitAnalytic:
    '''Fully Implicit Solver. 
        With the ANALYTIC implementation of the residual and Jacobian. 
    Arguments:
        grid: cartesianGrid
        rock: Rock
        wells: Wells
        fluid: singleFluid
        p_init (np.array) vector with the initial pressure for each cell
        gravity (bool): True, consider gravity effects (default). 
            False, neglect gravity effects.
        sources (np.array)  vector (cellnumber x 1 ) with the source term for each block.
            (-) is injection, (+) is production. 
        '''
    def __init__(self, grid, rock, fluid, wells, source, p_init, boundary,  gravity = True):
            ''' Initialize the fully implicit solver.'''
            self.grid = grid
            self.rock = rock
            self.fluid = fluid
            self.wells = wells
            self.source = source
            self.p_init = p_init
            self.gravity = gravity
            self.boundary = boundary
        

    def solve(self, schedule, max_iter = 10, tol= 1E-12, linear_solver=None):
        '''Run the simulation for each timesteps in the schedule.
        Arguments:
            schedule (object) : schedule
            linear_solver (function) : f(A, b) = x (To be implemented in the future)
            tol (float) : tolerance for Newton-Raphson 
            max_iter (int) : the maximum number of iterations for the Newton-Raphson loop.
            
        Returns:
            pressure (np.array) : an array with dimensions [ cellnumber x (timesteps + 1) ]
        '''
        # --- OUTSIDE THE TIME LOOP ---

        # Empty-matrix to store results. One extra column to save initial pressure.
        results=np.empty([self.grid.cellnumber, schedule.timesteps.size + 1])
        results[:,0] = self.p_init

        # Compute the geometric part of the transmissibility.
        gtx, gty, gtz = geometricTrans(self.grid, self.rock)
        # Compute the difference in depth for the gravity term.
        dzx, dzy, dzz = dh(self.grid, self.grid.depth)
        DZ = dh_matrix(self.grid, dzx, dzy, dzz)
        
        #Start with the initial pressure
        p_old = self.p_init
        p_new = p_old

        # --- INSIDE THE TIME LOOP ---
        for k in range(0,schedule.timesteps.size):
            # Print message to show progress of simulation.
            message = 'Solving timestep {:<2d} / {:<2d}'.format(k+1, schedule.timesteps.size)
            print(message)
            #Select timestep
            dt = schedule.timesteps[k]        
            
           # We define a function to handle the residual.
            def residual_and_jacobian(p_new):
                ''' Returns the residual and Jacobian as a function of pressure.
                Arguments: 
                    p_new(np.array) : pressure vector
                Returns:
                    residual (np.array)
                    jacobian (sparse.matrix)
                '''                
                # Update well 
                self.wells.update_wells(p_new)
                # Compute the fluid part of the transmissibility.
                ftx, fty, ftz, rhox, rhoy, rhoz = fluidTrans(self.grid, self.fluid, p_new)
                # Multiply the geometric part and the fluid part of the transmissibilities.
                tx = gtx * ftx
                ty = gty * fty
                tz = gtz * ftz
                # Assemble the transmissibility matrix. 
                # T_inc is the transmissibility matrix without the main diagonal
                T, T_inc = transTerm(self.grid, tx, ty, tz)
                # Compute the acummulation term with the previous pressure solution
                B = accumTerm(self.grid, self.rock , self.fluid, self.p_init, p_old, p_new)
                # Compute the gravity term.
                # If gravity == True
                if self.gravity:
                    # Gamma matrix
                    gamma_matrix = gamma(self.grid, rhox, rhoy, rhoz)
                    g = gravityTerm(self.grid, DZ, gamma_matrix, T_inc)
                # If gravity == False
                else:
                    # We need the gamma matrix for the Jacobian
                    gamma_matrix = gamma(self.grid, rhox, rhoy, rhoz)
                    g = np.zeros([self.grid.cellnumber, 1])
             
                # Apply the well conditions only if there are any wells. 
                if len(self.wells.wells) > 0 :
                    # Update the source term with the well rate.
                    source = self.wells.update_source_term(self.source)
                    # Update the transmissibility term with the wells.
                    #T = self.wells.update_trans(self.grid , T_inc)
                else:
                    source = self.source

                #Residual
                r = ( T.dot(p_new.reshape([-1,1])) - 
                    (1/dt) * B.dot(p_new.reshape([-1,1]) - p_old.reshape([-1,1])) 
                    - source - g )
                # Jacobian
                J = jacobian_of_residual(self.grid, self.rock, self.fluid, self.wells, gtx, gty, gtz, DZ, dt, p_old, p_new, T_inc, B, gamma_matrix)

                return r , J

            # Solve with Newton-Rahpson Method
            p_new, iter_number, error = newton_raphson_for_fully_implicit( residual_and_jacobian, p_new, tol,  max_iter)
        
            # Save results
            results[:,k + 1] = p_new.reshape([self.grid.cellnumber,])
            #Save rate for each well
            for wi in range(0,len(self.wells.wells)):
                self.wells.wells[wi]['rate_sol'].append( self.wells.wells[wi]['rate'][-1])
            # Set old pressure as the new one
            p_old = p_new

        return results, self.wells.wells


def newton_raphson_numerical( f, x0, tol, max_iter, eps, linear_solver = None):
    '''
    Newton-Raphson method to obtain the root of the function.
    Arguments:
        f (function) : function of the form f(x) = 0
        x0 (np.array) : vector with first guess for x
        tol (float) : stopping criteria. relative error.
        max_iter (int) : stopping criteria. Maximum number of iterations.
        eps (float) : epsilon for Jacobian. dx = (f(x  + eps) - f(x))/ eps
        linear_solver (function) :  f(A,b) = x (To be implemented later)
    Returns :
        x (np.array) : vector that solves f(x) = 0
        k (int) : number of iterations
        error (float) : relative error
    '''
    # Algorithm
    # Shift iteration counter by +1 to present results in a more natural way.
    for k in range(0, max_iter):
  
        J = jacobian (f, x0, eps)
        fx0 = f(x0)
        x = x0 - linalg.spsolve(J, fx0)          
        # If relative error is less than tolerance, break loop
        error = np.linalg.norm(x-x0)/np.linalg.norm(x)
        
        # Print message to show status
        message = ' \t Newton-Raphson solver : {:2d}/{:2d}. Error: {:.2E}'.format(k + 1, max_iter, error)
        print(message)

        if  error < tol:
            break
        x0 = x
    return x , k , error 


def newton_raphson_for_fully_implicit( f, x0,  tol , max_iter, linear_solver = None ):
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
    '''
    # Algorithm
    # Shift iteration counter by +1 to present results in a more natural way.
    for k in range(0, max_iter):
        
        fx0, J = f(x0)

        x = x0 - linalg.spsolve(J, fx0)                       
        # If relative error is less than tolerance, break loop
        error = np.linalg.norm(x-x0)/np.linalg.norm(x)
        # Print message to show status
        message = ' \t Newton-Raphson solver : {:2d}/{:2d}. Error: {:.2E}'.format(k + 1, max_iter, error)
        print(message)

        if  error < tol:
            break
        x0 = x

    return x , k , error 


def jacobian(f, x, eps):
    '''
    Computes  the jacobian of a function with the perturbation method
    Arguments:
        f (function): function with a vector as an input,  f(x)
        x (np.array) : vector
        eps (float) : epsilon to compute the derivative
    Returns:
        J (sparse.matrix) : jacobian matrix in csr format 
    '''
    n = x.size
    J = sparse.lil_matrix((n, n))
    fx = f(x)

    for i in range(0, n):
        x_eps =  x
        x_eps[i] = x_eps[i] + eps
        # Compute new column for the matrix
        ji = (f(x_eps)-fx)/eps
        # Insert in matrix
        J[: , i] = ji.reshape([-1,1])

    return J.tocsr()


def fluidTrans(grid, fluid, pressure):
    '''Computes the fluid part of the transmissibility for each cell.
    The fluid transmissibility is the arithmetic average between two cells. 
    Arguments:
        grid (object) : cartesianGrid
        fluid (object) : singleFluid
        pressure (np.array) : vector with the pressure for each cell.
    Returns: 
        Tx, Ty, Tz (np.array) : transmissibility matrices for each dimension.
        rhox, rhoy, rhoz (np.array) : density matrices for each dimension.
    '''
    nrows,ncols,nlayers = grid.dim
    #Averaging constant
    w = 0.5
    #The pressures in matrix form
    p = pressure.reshape(grid.dim, order='F')
    #Pressure average
    # in x
    px = np.add(w * p.swapaxes(0, 1)[:,0:-1,:] , (1 - w) * p.swapaxes(0, 1)[:,1:,:])
    # in y
    py = np.add(w * p[:,0:-1,:] , (1-w) * p[:,1:,:])
    # in z
    pz = np.add(w * p.swapaxes(1,2)[:,0:-1,:] , (1 - w) * p.swapaxes(1, 2)[:,1:,:])
    # Fluid properties from pressure average.
    # Viscosity
    miux = fluid.miu(px)
    miuy = fluid.miu(py)
    miuz = fluid.miu(pz)
    # Formation volume factor
    fvfx = fluid.fvf(px)
    fvfy = fluid.fvf(py)
    fvfz = fluid.fvf(pz)
    # Density
    rhox = fluid.rho(px)
    rhoy = fluid.rho(py)
    rhoz = fluid.rho(pz)
    
    # Fluid part of the transmissibility
    Tx = 1/miux * 1/fvfx
    Ty = 1/miuy * 1/fvfy
    Tz = 1/miuz * 1/fvfz
    # Complete the transmissibilities with zeros
    Tx = np.hstack((Tx, np.zeros([ncols, 1, nlayers]))).swapaxes(0, 1)
    Ty = np.hstack((Ty, np.zeros([nrows, 1, nlayers])))
    Tz = np.hstack((Tz, np.zeros([nrows, 1, ncols]))).swapaxes(1, 2)
    # Complete density matrix with zeros
    rhox = np.hstack((rhox, np.zeros([ncols, 1, nlayers]))).swapaxes(0, 1)
    rhoy = np.hstack((rhoy, np.zeros([nrows, 1, nlayers])))
    rhoz = np.hstack((rhoz, np.zeros([nrows, 1, ncols]))).swapaxes(1, 2)

    return Tx, Ty, Tz, rhox, rhoy, rhoz


def geometricTrans(grid, rock):
    '''Computes the geometric part of the transmissibility for each cell.
    The geometric transmissibility is the harmonic average between two cells.
    Arguments:
        grid: cartesianGrid
        rock: Rock
    Returns: 
        Tx, Ty, Tz (np.array) : transmissibility matrices for each dimension.
    '''
    nrows, ncols, nlayers = grid.dim
    # Area perpendicular to each direction
    # For an uniform cartesian grid
    Ayz = grid.cellsize[1] * grid.cellsize[2]
    Axz = grid.cellsize[0] * grid.cellsize[2]
    Axy = grid.cellsize[0] * grid.cellsize[1]
    # The permeabilities in matrix form
    kx = rock.perm.reshape(grid.dim, order = 'F')
    # For now, it is assumed that the reservoir has isotropic properties
    ky = kx
    kz = kx
    # Transmissibilities
    # in x
    Tx = np.divide(np.multiply(kx.swapaxes(0, 1)[:,0:-1,:] , kx.swapaxes(0, 1)[:, 1:, :]),
                 np.add(kx.swapaxes(0, 1)[:, 0:-1, :] , kx.swapaxes(0,1)[:, 1:,:]))

    Tx = np.hstack((Tx, np.zeros([ncols, 1, nlayers]))).swapaxes(0, 1)

    # in y
    Ty = np.divide(np.multiply(ky[:, 0:-1, :] , ky[:, 1:, :]),
        np.add(ky[:, 0:-1, :] , ky[:, 1:, :]))

    Ty = np.hstack((Ty, np.zeros([nrows, 1, nlayers])))

    # in z
    Tz = np.divide(np.multiply(kz.swapaxes(1,2)[:,0:-1,:] , kz.swapaxes(1,2)[:,1:,:]),
                 np.add(kz.swapaxes(1,2)[:,0:-1,:] , kz.swapaxes(1,2)[:,1:,:]))


    Tz = np.hstack((Tz,np.zeros([nrows,1,ncols]))).swapaxes(1,2)

    #Complete the transmissibilities by multiplying by constants
    Tx=2*Ayz/grid.cellsize[0]*Tx
    Ty=2*Axz/grid.cellsize[1]*Ty
    Tz=2*Axy/grid.cellsize[2]*Tz
    
    return Tx, Ty, Tz 


def transTerm(grid, tx, ty, tz):
    '''Assembles the transmissibility matrix. 
    Arguments:
        grid: cartesianGrid.
        tx, ty, tz (np. array): matrix with the transmissibilities for each dimension.
    Returns:
        T (sparse.csr): transmissibility matrix.
        T_inc (sparse.csr): transmissibility matrix without the main diagonal. 
    '''
    # Dimensions
    nrows, ncols, nlayers = grid.dim
    # Reshape the transmissibility to vector form
    tx = tx.reshape([grid.cellnumber, ], order='F')   
    ty = ty.reshape([grid.cellnumber, ], order='F')   
    tz = tz.reshape([grid.cellnumber, ], order='F')

    b = tx[:-1]
    c = tx[:-1]
    d = ty[:-nrows]
    e = ty[:-nrows]
    f = tz[:-nrows * ncols]
    g = tz[:-nrows * ncols]

    #Insert diagonals in the matrix. 
    #Create sparse matrix with format 'lil' because it is less expensive to modify. 
    T_inc = insertDiagonals(grid, b, c, d, e, f, g, format='csr')
    # Add all the elements in each row to get the main diagonal.
    a = (-1) * T_inc.sum(axis = 1)
    # Insert main diagonal
    ix = np.arange(0, grid.cellnumber, 1)
    T = T_inc.tolil() 
    T[ix,ix] = np.array(a).reshape([-1,])
    # Return the transmissibility matrix in csr format. Is faster for computations. 
    return T.tocsr(), T_inc


def gamma(grid, rhox, rhoy, rhoz):
    ''' gamma =  rho * g / gc . (gc = 1 in SI units.)
    Arguments:
        grid: cartesianGrid.
        rhox, rhoy, rhoz (np.array) : matrices with the density for each
            cell, calculated from the average pressure, for each dimension.
    Returns:
        Y (scipy.sparse) : matrix with gamma for each cell.
    '''
    # Dimensions
    nrows, ncols, nlayers = grid.dim
    # Reshape the density matrices to vector form
    rhox = rhox.reshape([grid.cellnumber, ], order='F')    
    rhoy = rhoy.reshape([grid.cellnumber, ], order='F')    
    rhoz = rhoz.reshape([grid.cellnumber, ], order='F')
    # Diagonals
    b = rhox[:-1]
    c = b
    d = rhoy[:-nrows]
    e = d
    f = rhoz[:-nrows * ncols]
    g = f
    # Insert diagonals in sparse matrix
    Y = insertDiagonals(grid, b, c, d, e, f, g, format='csr')
    Y =  constants.g * Y
    return Y


def dh(grid, h):
    '''
    Returns the matrices with the difference in "h" between cells. 
    Arguments:
        grid: cartesianGrid.
        h (np.array) : vector with property "h" for each cell. 
            ie. "h" can be pressure, depth, etc. 
    Returns: 
        dhx, dhy, dhz (np.array) : delta "h" in each direction.
            ie. (h_{i + 1} - h_{i})
    '''
    # Dimensions
    nrows, ncols, nlayers = grid.dim
    # The cell " h " in matrix form
    h = h.reshape(grid.dim, order='F')
    # Difference in depth in each direction
    # in x
    dhx = np.subtract(h.swapaxes(0, 1)[:, 0:-1, :] , h.swapaxes(0,1)[:, 1:,:])
    dhx = np.hstack((dhx, np.zeros([ncols, 1, nlayers]))).swapaxes(0, 1)
    # in y
    dhy = np.subtract(h[:, 0:-1, :] , h[:, 1:, :])
    dhy = np.hstack((dhy, np.zeros([nrows, 1, nlayers])))
    # in z
    dhz = np.subtract(h.swapaxes(1,2)[:,0:-1,:] , h.swapaxes(1,2)[:,1:,:])
    dhz = np.hstack((dhz, np.zeros([nrows,1,ncols]))).swapaxes(1,2)

    return dhx, dhy, dhz


def dh_matrix(grid, dhx, dhy, dhz):
    '''Difference in "h" matrix.
    Arguments:
        grid: cartesianGrid.
        dhx, dhy, dhz (np.array) : delta "h" in each direction.
    Returns:
        DH (scipy.sparse): sparse matrix. 
    '''
    # Dimensions
    nrows, ncols, nlayers = grid.dim
    # Reshape the dz matrices to vector form
    dhx = dhx.reshape([grid.cellnumber, ], order='F')    
    dhy = dhy.reshape([grid.cellnumber, ], order='F')    
    dhz = dhz.reshape([grid.cellnumber, ], order='F')
    
    b = dhx[:-1]
    c = (-1) * b
    d = dhy[:-nrows]
    e = (-1) * d
    f = dhz[:-nrows * ncols]
    g = (-1) * f
    # Insert diagonals in a matrix.     
    DH = insertDiagonals(grid, b, c, d, e, f, g, format = 'csr')  
    return DH

def gravityTerm(grid, dz_matrix, gamma_matrix, T_inc):
    '''Assembles the gravity matrix. 
    Arguments:
        grid: cartesianGrid.
        dz_matrix (sparse): sparse matrix with the dz
        gamma_matrix (sparse) : matrix with the gamma term 
        T_inc (sparse) : transmissibility matrix without the main diagonal
    Returns:
        g (np.array): column vector with the gravity term.
    ''' 
    # Multiply element-wise
    G = T_inc.multiply( gamma_matrix.multiply(dz_matrix) )
    # Add the elements in each row to get the main diagonal.
    g = G.sum(axis = 1)
    # The sum returns "a" as a matrix. We need it in vector form.
    g = np.array(g)

    return g.reshape([grid.cellnumber,1])


def accumTerm(grid, rock, fluid, p_init, p_old, p_new):
    ''' Returns the accumulation matrix.  It contains the total compressibility
        in its main diagonal. 
    Arguments:
        grid: cartesianGrid
        rock : Rock
        fluid : singleFluid
        p_init (np.array) : vector with initial pressure.
        p_old (np.array) : vector with pressure solution of previous timestep.
        p_new (np.array) : vector with current  pressure.
    Returns:
        B (sparse) : diagonal matrix with the total compressibility.
    '''
    # Could precompute values to save time
    fvf_init = fluid.fvf(p_init)
    #FVF with previous timestep pressure solution
    fvf_old = fluid.fvf(p_old)
    # Porosity with current pressure
    poro_new = rock.porofunc(p_new)
    # Total compressibility. Main diagonal
    ct = (poro_new * fluid.cf / fvf_init) +  (rock.cr * rock.poro / fvf_old)
    # Multiply by cell volume.
    ct = ct * grid.cellvolume
    return sparse.diags(ct, 0, format='csr')


def insertDiagonals(grid, b, c, d, e, f, g, format='csr'):
    ''' Returns a sparse matrix with the diagonals.
    Arguments:
        grid: cartesianGrid.
        b, c, d, e, f, g (np.arrays): column vectors with the diagonals.
        format (string): format of sparse matrix.  
    Returns: 
        M (scipy.sparse) : sparse matrix of the form :

    # [ a c - e - g - ]
    # [ b a c - e - g ]
    # [ - b a c - e - ]
    # [ d - b a c - e ]
    # [ - d - b a c - ]
    # [ f - d - b a c ]
    # [ - f - d - b a ]
    '''
    # Dimensions
    nrows, ncols, nlayers = grid.dim

    indexes=[]
    diagonals=[]

    # Select the diagonals to be inserted into the matrix
    if nrows > 1 :
        indexes = indexes + [-1, 1]
        diagonals.append(b.T)
        diagonals.append(c.T)
    
    if ncols > 1 :
        indexes = indexes + [-nrows, nrows]
        diagonals.append(d.T)
        diagonals.append(e.T)
    
    if nlayers > 1 :
        indexes = indexes + [-nrows * ncols, nrows * ncols]
        diagonals.append(f.T)
        diagonals.append(g.T)
         
    #Insert diagonals into the matrix. 
    M = sparse.diags(diagonals, indexes, format=format )
    return M


class Schedule():
    '''Schedule to control the simulation.
    Attributes
        timesteps (numpy.array)
        tottime : sum of timesteps
    '''

    def __init__(self, timesteps):
        '''
        Arguments:
            timesteps (numpy.array): vector with the timesteps.
            Example:
            timesteps=np.array([1,1,1,1,2,2,2,10])*units.day
        '''
        self.timesteps = timesteps
        self.tottime = np.sum(self.timesteps)


class Wells():
    ''' Wells object with support functions to manage the wells in the reservoir.
    Attributes:
        grid: cartesianGrid
        rock: Rock
        fluid: singleFluid , blackOil
        wells (list): list with all the wells. 
        d_rate (scipy.sparse): matrix with the d_rate / d_pi 
        add_vertical_well (function)
        update_wells (function)
        update_source_term (function)
        update_trans (function)
    '''

    def __init__(self, grid, rock, fluid):
        '''
        Arguments:
            grid: cartesianGrid
            rock : Rock
            fluid : singleFluid, blackOil
        '''
        self.grid = grid
        self.rock = rock
        self.fluid = fluid
        self.wells = []
        
        # if single phase
        if self.fluid.phasenumber == 1:
            self.d_rate = sparse.lil_matrix((grid.cellnumber , 1 ))
        # if two phase
        elif self.fluid.phasenumber == 2:
            dempty = np.zeros([grid.cellnumber, ])
            self.d_rate = [dempty, dempty, dempty, dempty]



    def add_vertical_well(self, rw, location, bhp, 
            skin ,  name=None , well_equation = '5PA' ):
        ''' Add a vertical well with constant bottom hole pressure.
        The well is in the center of the cell, and perforated only in that cell.
            Uses Peaceman equation. Assumes isotropic permeability. 

        Arguments:
            rw (float): well radius. 
            location (int): cell where the well is located. 
            bhp (float): constant bottom hole pressure.
            skin (float): dimensionless skin factor.
            name (string): label to identify the well.
            well_equation (string): '5PA' or '9PA', five or nine point approximation.            
        Returns:
            Adds a new well to self.wells.
        '''
        # Dimensions
        nrows, ncols, nlayers = self.grid.dim
        Sx , Sy, Sz = self.grid.cellsize
        # Equivalent radius.Assuming an isotropic reservoir. 
        req = 0.14 * (Sx**2 + Sy**2) ** 0.5
            
        # Five-point approximation..
        if well_equation == '5PA':
            tna = 1
            tnb = 0
        # Nine-point approximation
        elif well_equation == '9PA':
            tna = 2/3
            tnb = 1/6
        else:
            raise Exception ("'{}' is not a valid well equation.".format(well_equation))

        # The neighbor cells are numbered as follow.
        # |  |   |   |   |  |
        # |  | 5 | 1 | 7 |  |
        # |  | 3 | 0 | 4 |  |
        # |  | 6 | 2 | 8 |  |
        # |  |   |   |   |  |

        # List to save the neighbor cells information.
        # The first column is the cell location. 
        # The second column  is the corresponding face transmissibility.
        neighbor_cells = np.zeros([8,2])
        neighbor_cells[0,0] = (location - 1 )
        neighbor_cells[1,0] = (location + 1)
        neighbor_cells[2,0] = (location - nrows)
        neighbor_cells[3,0] = (location + nrows)
        neighbor_cells[4,0] = (location - nrows + 1)
        neighbor_cells[5,0] = (location - nrows - 1 )
        neighbor_cells[6,0] = (location + nrows - 1 )
        neighbor_cells[7,0] = (location + nrows + 1 )

        neighbor_cells[0,1] = tna 
        neighbor_cells[1,1] = tna 
        neighbor_cells[2,1] = tna 
        neighbor_cells[3,1] = tna 
        neighbor_cells[4,1] = tnb 
        neighbor_cells[5,1] = tnb 
        neighbor_cells[6,1] = tnb 
        neighbor_cells[7,1] = tnb 

        if self.fluid.phasenumber == 1:
            rate = [0]
            rate_sol = [0]
        elif self.fluid.phasenumber == 2:
            # First water, second oil
            rate = [[0],[0]]
            rate_sol = [[0],[0]]
        else:
            pass

        self.wells.append({'name': name, 'bhp': bhp, 'location': location,'rate' : rate , 
            'rate_sol': rate_sol,  'T0' : None, 'neighbor_cells': neighbor_cells, 'rw':rw, 'req':req, 'skin':skin})
        return


    def update_wells(self, p_cell, sw_cell = None):
        '''Updates  the flow of each well, as well as the rate derivative. 
        Arguments:            
            p_cell(np.array) : cell pressure.
            sw_cell (np.array) : cell water saturation (only for blackOil)
        '''
        Sx, Sy, Sz = self.grid.cellsize
        nrows, ncols, nlayers = self.grid.dim
        
        # Assuming a isotropic reservoir. 
        # Equivalent radius.
        req = 0.14 * (Sx**2 + Sy**2) ** 0.5

        for iw in range(0,len(self.wells)):

            location = self.wells[iw]['location']
            rw = self.wells[iw]['rw']
            req = self.wells[iw]['req']
            skin = self.wells[iw]['skin']
            bhp = self.wells[iw]['bhp']
            pi = p_cell[location]
            perm = self.rock.perm[location]

            if self.fluid.phasenumber == 1:
                # Update the Well Index
                miu = self.fluid.miu(pi)
                fvf = self.fluid.fvf(pi)
                WI =  (2 * np.pi * perm * Sz ) / ( miu * fvf * (np.log( req /rw  ) + skin ) )
                # Transmissibility 
                T0 = ( perm * Sz ) / ( miu * fvf )
                #self.wells[iw]['WI']= WI
                # Calculate well rate
                rate = WI * ( pi - bhp)
                # No negative rate. This is a production well
                if rate < 0:
                    rate = 0 
                # Save rate value in well list
                self.wells[iw]['rate'].append(rate)
                self.wells[iw]['T0'] = T0

                # Well rate derivative
                eps = 1 * u.psi
                dfluid = (( self.fluid.miu(pi + eps) * self.fluid.fvf(pi + eps) ) ** -1 
                    - ( self.fluid.miu(pi) * self.fluid.fvf(pi) ) ** -1 )/ eps
                dWI_dP = 2 * np.pi * perm * Sz * dfluid
                self.d_rate[location] = WI + (pi - bhp) * dWI_dP 

            if self.fluid.phasenumber == 2:

                swi = sw_cell[location]

                # Update the Well Index
                # For water
                miuw = self.fluid.waterphase.miu(pi)
                fvfw = self.fluid.waterphase.fvf(pi)
                krw = self.fluid.waterphase.kr(swi)
                WIw =  (2 * np.pi * perm * krw * Sz ) / ( miuw * fvfw * (np.log( req /rw  ) + skin ) )
                # Calculate well rate. Ignoring capillary pressure effects.
                water_rate = WIw * ( pi - bhp)
                # No negative rate. This is a production well
                if water_rate < 0:
                    water_rate = 0 

                #if water_rate > 301 *u.barrel /u.day:
                #    water_rate = 300 *u.barrel /u.day
                # For oil
                miuo = self.fluid.oilphase.miu(pi)
                fvfo = self.fluid.oilphase.fvf(pi)
                kro = self.fluid.oilphase.kr(swi)
                WIo =  (2 * np.pi * perm * kro * Sz ) / ( miuo * fvfo * (np.log( req /rw  ) + skin ) )
                # Calculate well rate. Ignoring capillary pressure.
                oil_rate = WIo * ( pi - bhp)

                if oil_rate < 0:
                    oil_rate = 0 

                # Save rate in well list
                self.wells[iw]['rate'][0].append(water_rate)
                self.wells[iw]['rate'][1].append(oil_rate)

                
                # Well rate derivative. 
                eps_p = 1 * u.psi
                
                dfluidw = (( self.fluid.waterphase.miu(pi + eps_p) * self.fluid.waterphase.fvf(pi + eps_p) ) ** -1 
                    - ( self.fluid.waterphase.miu(pi) * self.fluid.waterphase.fvf(pi) ) ** -1 )/ eps_p

                dfluido = (( self.fluid.oilphase.miu(pi + eps_p) * self.fluid.oilphase.fvf(pi + eps_p) ) ** -1 
                    - ( self.fluid.oilphase.miu(pi) * self.fluid.oilphase.fvf(pi) ) ** -1 )/ eps_p

                dkrw = self.fluid.waterphase.d_kr(swi)
                dkro = self.fluid.oilphase.d_kr(swi)

                dWIw_dP = 2 * np.pi * perm * krw *  Sz * dfluidw
                dWIo_dP = 2 * np.pi * perm * kro *  Sz * dfluido
                dWIw_dSw = 2 * np.pi * perm * Sz /miuw / fvfw * dkrw
                dWIo_dSw = 2 * np.pi * perm * Sz /miuo / fvfo * dkro

                # dqw_dp 
                self.d_rate[0][location] = WIw + (pi - bhp) * dWIw_dP
                # dqw_dsw 
                self.d_rate[1][location] = WIw + (pi - bhp) *  dWIw_dSw
                # dqo_dp
                self.d_rate[2][location] = WIo + (pi - bhp) * dWIo_dP
                # dqo_dsw
                self.d_rate[3][location] = WIo + (pi - bhp) * dWIo_dSw

        return 


    def update_source_term(self, source):
        '''
        Modifies the source term for the cells where a well is located.
        Arguments:
            source (np.array) : source term.
        Return:
            source : source term with well rate.
        '''
        for iw in range(0,len(self.wells)):
            location = self.wells[iw]['location']
            # Grab the latest rate calculated
            # Insert the rate in the cell location
            if self.fluid.phasenumber == 1:
                source[location] = self.wells[iw]['rate'][-1]

            elif self.fluid.phasenumber == 2:
                source[location * 2    ] = self.wells[iw]['rate'][0][-1] #water
                source[location * 2 + 1] = self.wells[iw]['rate'][1][-1] #oil

        return source


    def update_trans(self, grid , T_inc):
        '''
        ONLY FOR ONE PHASE FLOW
        Modifies the rows of the transmissibility matrix corresponding to the well block
        and the surrounding cells. 
        Arguments:
            grid : cartesianGrid.
            T_inc (sparse): transmissibility matrix without the main diagonal.
        Returns:
            T (sparse.csr): complete transmissibility matrix. 
        '''
        # Dimensions
        nrows, ncols, nlayers = grid.dim

        # Convert to 'lil' format to modify easily.
        T_inc = T_inc.tolil()

        # Erase the  elements of the diagonals in the z's layers
        for iw in range(0,len(self.wells)):

            location = self.wells[iw]['location']
            Tn =  self.wells[iw]['neighbor_cells'][:,1] * self.wells[iw]['T0']

            # Erase the connection between the well cell and the top and botom layer.
            if nlayers > 1 :
                # If well is not in bottom layer
                if location >= (nrows * ncols):
                    T_inc[location, location - (nrows*ncols)] = 0
                # If well is not in top layer
                if location <= ( nrows * ncols * (nlayers - 1 )  - 1  ):
                    T_inc[location, location + (nrows*ncols)] = 0
            
            # Transmissibility of the neighboring cells. Using a nine-point scheme.
            # |  |   |   |   |  |
            # |  | 5 | 1 | 7 |  |
            # |  | 3 | 0 | 4 |  |
            # |  | 6 | 2 | 8 |  |
            # |  |   |   |   |  |
            # Connect the well cell with the sourrounding cells
            T_inc[location, location - 1]           = Tn[0]
            T_inc[location, location + 1]           = Tn[1]
            T_inc[location, location - nrows]       = Tn[2]
            T_inc[location, location + nrows]       = Tn[3]
            T_inc[location, location - nrows - 1]   = Tn[4]
            T_inc[location, location - nrows + 1]   = Tn[5]
            T_inc[location, location +  nrows - 1]  = Tn[6]
            T_inc[location, location + nrows + 1]   = Tn[7]
            
        # Add all the elements in each row to get the main diagonal.
        a = (-1) * T_inc.sum(axis = 1)
        # Insert main diagonal
        T_inc.setdiag(a,0)
        # Return the transmissibility matrix in csr format. Is faster for computations. 
        return T_inc.tocsr()


def apply_boundary(grid, T, rhs, boundary, wells):
    ''' Modify the transmissibility matrix 
    with the specified boundary conditions.
    Arguments:
        grid: cartesianGrid.
        T (scipy.sparse) : transmissibility matrix.
        rhs : right hand side
        boundary (object) : boundary conditions.
    Returns:
        T (scipy.sparse) : modified matrix
        boundary_rhs (np.array) : vector that must be added to the RHS
    '''
    # Filter vector. 
    filter_vector = np.ones(grid.cellnumber, )
    # index of the cells with boundary conditions
    index=np.empty_like([], dtype=int)
    # The locations are in ascending order of priority.
    # If the same cell has two or more boundary conditions,
    # the last boundary condition to be applied is the
    # one that remains. 
    valid_locations=['E', 'W', 'N', 'S', 'T', 'B']
    for bi in valid_locations: 
        
        if boundary.boundaries[bi][0] == 'no-flow':
            # The transmissibility matrix T is already constructed assuming a 
            # no flow condition for all boundaries
            pass

        elif boundary.boundaries[bi][0] == 'constant-pressure':
            #get cellnumbers
            n = compass_to_coordinates(grid, bi)
            index = np.hstack((index, n))
            # insert  value in rhs vector
            rhs[n] = boundary.boundaries[bi][1]
        # There is still no support for "delta-pressure"
        else:
            pass
     
    # Insert zeros in filter_vector. 
    filter_vector[index] = 0
    filter_vector = sparse.csr_matrix(filter_vector.reshape([-1,1]))
    # Erase the rows with zeros.
    T = T.multiply(filter_vector)
    # transform to modify inexpensively
    T = T.tolil()
    # insert 1's in diagonals
    T[index, index] = 1
    T = T.tocsr()
    return T, rhs.reshape([-1,1])


class Boundary():
    '''Apply Dirichlet or Neumann boundaries to the space.
    Attributes:
        boundaries
        set_boundary_condition
    '''

    def __init__(self ):    
        '''Initialize all boundaries with no flow condition.'''
        self.boundaries={'N':('no-flow', 0),
                         'S':('no-flow', 0), 
                         'E':('no-flow', 0), 
                         'W':('no-flow', 0), 
                         'T':('no-flow', 0), 
                         'B':('no-flow', 0)}

    def set_boundary_condition(self, location, boundary_type, value):
        ''' Modifies the boundary conditions of the space.
            Arguments:
                location (string): 'North', 'South', 'East', 'West', 'Top', 'Bottom'
                boundary_type (string): 'no-flow', 'constant-pressure', 'delta-pressure'
                value: value of the boundary. For the 'no-flow' option, the value of the boundary is ignored
        '''
        # Confirm that location provided is a valid value
        #valid_locations=['N', 'S', 'E', 'W', 'T', 'B']
        if location in self.boundaries :
            pass
        else:
            raise Exception('Location "{}" is not a valid option.'.format(location))
        
        # Confirm that boundary type provided is a valid value
        valid_boundaries = ['no_flow', 'constant-pressure','delta-pressure' ]
        if boundary_type in valid_boundaries :
            pass
        else:
            raise Exception('Boundary type "{}" is not a valid option.'.format(boundary_type))
        
        #Add boundary to dictionary
        self.boundaries[location] = (boundary_type, value)
        return 


def compass_to_coordinates(grid, location):
    '''Return the cell number of the cells located in the 
    layer determined by the compass direction. 
    Assumes a natural grid ordering.
    Arguments:
        grid: cartesianGrid.
        location (string): N, S, E, W, T, B
    Return:
        n (np.array): cellnumbers
    '''
    nrows, ncols, nlayers = grid.dim
    
    #TOP
    if location == 'T' :
        n = np.arange(nrows * ncols * (nlayers - 1 ) , grid.cellnumber)
    #BOTTOM
    elif location == 'B' :
        n = np.arange(0 , nrows * ncols)
    #WEST
    elif location == 'W' :
        start = 0
        end = nrows
        offset = nrows * ncols
        nbase = np.arange(start, end)
        if nlayers == 1:
            n = nbase
        else:
            n = nbase
            for i in range(1 , nlayers ):
                n = np.hstack(( n , nbase + i * offset))    
    #EAST
    elif location == 'E':
        start  = nrows * (ncols - 1)
        end = nrows * ncols
        offset = nrows * ncols
        nbase = np.arange(start , end )
        if nlayers == 1 :
            n = nbase
        else: 
            n = nbase
            for i in range(1 , nlayers ):
                n = np.hstack(( n , nbase + i * offset)) 
    #NORTH 
    elif location == 'N':
        n = np.arange(0 , grid.cellnumber, nrows)
    #SOUTH
    elif location == 'S':
        n = np.arange(nrows - 1 , grid.cellnumber , nrows)
    else:
        raise Exception('Invalid location')
                        
    return n


def derivative_fluidTrans(grid, fluid, pressure):
    ''' Partial derivative of fluid part of the transmissibility term with respect to pressure.
    Arguments:
        grid (object)
        fluid (object)
        pressure (np.array) : vector with the pressures for each cell.
    Returns:
        d_txi, d_tyi, d_tz (np.array) : transmissibility derivative with respect to cell pressure
        d_tx, d_ty, d_tz (np.array) : transmissibility derivative with respect to next cell pressure
        d_rhox, d_rhoy, d_rhoz (np.array) : density derivative        
    '''
    nrows,ncols,nlayers = grid.dim
    #Averaging constant. THE SAME USED IN fluidTrans function.
    w = 0.5
    #The pressures in matrix form
    p = pressure.reshape(grid.dim, order='F')
    #Pressure  in the neighboring cells
    px = p
    py = p
    pz = p 

    # Fluid properties 
    # Viscosity
    miux = fluid.miu(px)
    miuy = fluid.miu(py)
    miuz = fluid.miu(pz)
    # Formation volume factor
    fvfx = fluid.fvf(px)
    fvfy = fluid.fvf(py)
    fvfz = fluid.fvf(pz)
    
    # Derivatives of:
    eps = 1E-6
    # Viscosity
    d_miux = w * (fluid.miu(px + eps) - fluid.miu(px))/ eps
    d_miuy = d_miux
    d_miuz = d_miux
    #Formation volume factor
    d_fvfx = w * (fluid.fvf(px + eps) - fluid.fvf(px))/ eps
    d_fvfy = d_fvfx
    d_fvfz = d_fvfx

    # Compute the derivatives in each direction
   
    d_tx = (-1/(fvfx * miux) ** 2) * (miux * d_fvfx + fvfx * d_miux)
    d_ty = (-1/(fvfy * miuy) ** 2) * (miuy * d_fvfy + fvfy * d_miuy)
    d_tz = (-1/(fvfz * miuz) ** 2) * (miuz * d_fvfz + fvfz * d_miuz)
    
    # Pressure in the "i" cells
    pxi = p
    pyi = p
    pzi = p

    # Fluid properties 
    # Viscosity
    miuxi = fluid.miu(pxi)
    miuyi = fluid.miu(pyi)
    miuzi = fluid.miu(pzi)
    # Formation volume factor
    fvfxi = fluid.fvf(pxi)
    fvfyi = fluid.fvf(pyi)
    fvfzi = fluid.fvf(pzi)
    
    # Derivatives of:
    # Viscosity
    d_miuxi = (1- w) * (fluid.miu(pxi + eps) - fluid.miu(pxi))/ eps
    d_miuyi = d_miux
    d_miuzi = d_miux
   
    # Formation volume factor    
    d_fvfxi = (1- w) * (fluid.fvf(pxi + eps) - fluid.fvf(pxi))/ eps
    d_fvfyi = d_fvfx
    d_fvfzi = d_fvfx

    # Compute the derivatives in each direction
    d_txi = (-1/(fvfxi * miuxi)) * (miuxi * d_fvfxi + fvfxi * d_miuxi)
    d_tyi = (-1/(fvfyi * miuyi)) * (miuyi * d_fvfyi + fvfyi * d_miuyi)
    d_tzi = (-1/(fvfzi * miuzi)) * (miuzi * d_fvfzi + fvfzi * d_miuzi)
    
    # Density derivative
    d_rhox = (1- w) * (fluid.miu(pxi + eps) - fluid.miu(pxi))/ eps
    d_rhoy = (1- w) * (fluid.miu(pyi + eps) - fluid.miu(pyi))/ eps
    d_rhoz = (1- w) * (fluid.miu(pzi + eps) - fluid.miu(pzi))/ eps
 
    return d_txi, d_tyi, d_tzi, d_tx, d_ty, d_tz, d_rhox, d_rhoy, d_rhoz



def jacobian_of_residual(grid, rock, fluid, wells,  gtx, gty, gtz, DZ, dt, p_old, p_new, T_inc, B, gamma_matrix):
    '''
    Assembles the analytic Jacobian of the residual. 
    To use with the fully implicit method.
    grid: cartesianGrid
    fluid : fluid object
    pressure : vector with pressures for all cells 
    gtx, gty, gtz : vectors with the geometric part of the transmissibility
    '''
    # Delta pressure in each direction
    dpx, dpy, dpz = dh(grid, p_new)
    DP = dh_matrix(grid, dpx, dpy, dpz)
    # Derivative of the fluid part
    d_ftxi, d_ftyi, d_ftzi, d_ftx, d_fty, d_ftz, d_rhox, d_rhoy, d_rhoz = derivative_fluidTrans(grid, fluid, p_new)
    # Multiply by geometric part
    # For cell "i"
    d_txi = gtx * d_ftxi
    d_tyi = gty * d_ftyi
    d_tzi = gtz * d_ftzi
    # For neighboring cells
    d_tx = gtx * d_ftx
    d_ty = gty * d_fty
    d_tz = gtz * d_ftz
    # Assemble two matrices without a main diagonal
    no_use1, dT_neighbor = transTerm(grid, d_tx, d_ty, d_tz)
    n_use2, dT_i = transTerm(grid, d_txi, d_tyi, d_tzi)
    # === Derivative of acummTerm ===
    eps = 1E-6
    d_poro = (rock.porofunc(p_new + eps) - rock.porofunc(p_new ) ) / eps
    d_acumm = grid.cellvolume * d_poro * fluid.cf
    d_acumm = d_acumm.reshape([grid.cellnumber, 1])
    
    # === Derivative of gravity ====
    d_gamma_matrix = gamma(grid, d_rhox, d_rhoy, d_rhoz)
    d_gravity = DZ.multiply(dT_i.multiply(gamma_matrix) + T_inc.multiply(d_gamma_matrix))
    # Add column wise, and get a single main diagonal and transform to vector.
    d_gravity = d_gravity.sum(axis = 1)
    d_gravity = np.array(d_gravity)
    d_gravity = d_gravity.reshape([grid.cellnumber, 1])

    # === Derivative of residual for neighboring cells ===
    dR_next = T_inc + dT_neighbor.multiply(DP)

    # === Derivative of residual for "i" cells ===
    dR_i = (-1) * T_inc  +  dT_i.multiply(DP) - B/dt
    # Add all elements column-wise, and get a single main diagonal and transform to vector.
    dr_i = dR_i.sum(axis = 1)
    dr_i = np.array(dr_i)
    dr_i = dr_i.reshape([grid.cellnumber, 1])
    # Add the accumulation derivative, flow derivative and gravity derivative to main diagonal vector.
    # All of the terms are vectors or scalars
    dr_i = dr_i   - d_acumm * (p_new.reshape(-1,1) - p_old.reshape(-1,1)) / dt  - wells.d_rate - d_gravity
    # === Insert dr_i in main diagonal
    J = dR_next.tolil()
    dr_i = np.array(dr_i)
    J.setdiag(dr_i.reshape([-1,]), 0)
    
    return J.tocsr()


def main():
    ''' It is run if this module is run. '''

    print(' Simulator module.')


if __name__ == "__main__":
    main()
    sys.exit()
