'''

This module has functions and solvers to implement 
Reduced Order Modelling in the simulator.

'''

from scipy.sparse import linalg


def linear_solver(A,b, basis):

        # Project right and left hand side of equation
        A_r = (basis.T.dot(A)).dot(basis) 
        b_r = basis.T.dot(b)
        # Solve reduced system
        x_r = linalg.spsolve(Ar, br)
        x_r = x_r.reshape([-1,1])
        # Project back
        x = basis.dot(x_r)

return x



def main():
    ''' It is run if this module is run. '''

    print(' Reduced Order Modelling functions for porous media flow simulator.')


if __name__ == "__main__":
    main()
    sys.exit()