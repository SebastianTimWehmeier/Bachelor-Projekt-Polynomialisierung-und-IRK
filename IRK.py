import casadi as ca
import numpy as np
from math import  sqrt

class Model:
    def __init__(self):
       self.x = None
       self.nx: int = 0
       self.model =None

class RootSolvingProblem:
    def __init__(self):
       self.x = None
       self.dimX: int = 0
       self.parameter = None
       self.dimP: int = 0
       self.Problem =None # a sym representation of the function that should be zero
       self.IDMatrix = None
       self.Problem_func = None # the function that should be zero
       self.Problem_Jac_func= None # the first derivative
       self.Problem_H_v_func = None # the second derivative directly multiplied by a vector v
       #we  have to multiply the second derivative  with a vector to ensure not  have to work with a tensor

class IRK:
    def __init__(self, model:Model,root_solver_type: str, dt: float,
                 max_iter: int = 100, tol: float = 1e-6,
                 stages: int = 2):
        """
        Initialize Implicit Runge Kutta
        
        :param model: Model to simulate
        :type model: Model
        :param root_solver_type: Name of the root solving algorithm
        :type root_solver_type: str
        :param dt: time step distance
        :type dt: float
        :param max_iter: max num of iteration
        :type max_iter: int
        :param tol: tolerance
        :type tol: float
        :param stages: stages of the Gauss-Legandre quadrature 
        :type stages: int
        """
        self.model = model

        #assign which root solver you want to use
        self.rootSolver =None
        if root_solver_type == "Halleys":
            self.rootSolver = self.Halleys
        elif root_solver_type == "Newton":
            self.rootSolver = self.Newton
        else:
            raise ValueError("this root solver doesn't exsits")
        
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
        self.stages = stages
        # initialize  the weights A and B 
        if stages == 2:
            self.A = ca.DM([[1/4, 1/4-sqrt(3)/6],
                    [1/4+sqrt(3)/6, 1/4]])
            self.B = ca.DM([1/2, 1/2])
        else: 
            raise NotImplemented("not implemented")

        self.model_func = ca.Function("model_func", [self.model.x], [self.model.model])
        self.Initialize_RootSolving_Problem()


   
    def Initialize_RootSolving_Problem(self):
        """
        This function initializes the root finding problem
        which has the form of:
        0 = k_i - f(x0 + dt* (sum all k_j*a_j))

        in the end we will have a vector which consists  of all the k_i stack on top of each other

        self.RP.x are the k_i we are looking for
        self.RP.parameter  is  the current x0 

        """
        self.RP = RootSolvingProblem()
        self.RP.dimX = self.stages * self.model.nx #the dimension of the k_i stacked on top of each other
        for i in range(self.stages):
            if self.RP.x is None:
                self.RP.x = ca.SX.sym(f"K_{i+1}", self.model.nx)
            else:
                self.RP.x = ca.vertcat(self.RP.x, ca.SX.sym(f"K_{i+1}", self.model.nx)) #[k1_1, k1_2, ..., k1_i,   K2_1,K2_2 ,..., K2_i, ..., KI_i...]

        self.RP.dimP = self.model.nx #dimension of model.x will be treated as a const. parameter in each call of the root solver
        self.RP.parameter = ca.SX.sym("f_sym", self.model.nx)
        
        for i in range(self.stages):# 
            k_i = self.RP.x[i*self.model.nx: (i+1)*self.model.nx]
            K_times_A =  ca.mtimes(ca.reshape( self.RP.x,self.model.nx, self.stages ),self.A[i,:].T)# K_i summed up with the weights a_i
            root_i = k_i- self.model_func(self.RP.parameter+self.dt*(K_times_A))# the problem  0= k_i-f(x0+sum all (k_i*a_i))
            if self.RP.Problem is None:
                self.RP.Problem = root_i
            else:
                self.RP.Problem = ca.vertcat(self.RP.Problem,root_i)

        self.RP.Problem_func = ca.Function("Problem_func", [self.RP.x , self.RP.parameter ], [self.RP.Problem])
        jac = ca.jacobian(self.RP.Problem,self.RP.x)
        self.RP.Problem_Jac_func = ca.Function("jac_func", [self.RP.x , self.RP.parameter] ,[jac])

        v_sym = ca.SX.sym("v_sym", self.RP.dimX)#sym V
        self.RP.Problem_H_v_func = ca.Function("H_v", [self.RP.x,self.RP.parameter,v_sym],[ ca.jtimes(jac, self.RP.x, v_sym)]) # sym F''* V
        self.RP.IDMatrix = ca.DM.eye(self.RP.dimX)

        self.lastIterationCount =None # stores the last number of iterations needed by the root solver to solve that problem

    def Newton(self, k0,x0):
        """
        Newton's method
        root solving algorithm
        

        :param k0:  start initialization of the k_i
        :param x0: current x value for which we calculate the k to determine the next step
        
        """

        for i in range(self.max_iter):
            jac_inv_F = np.linalg.solve(self.RP.Problem_Jac_func(k0,x0), self.RP.Problem_func(k0,x0))
            k0 = k0-jac_inv_F
            if(np.linalg.norm(self.RP.Problem_func(k0,x0), np.inf) < self.tol):
                self.lastIterationCount=i+1
                return k0
        self.lastIterationCount= self.max_iter
        return k0
        
    def Halleys(self, k0,x0): 
        """
        Halley's method
        root solving algorithm
        

        :param k0:  start initialization of the k_i
        :param x0: current x value for which we calculate the k to determine the next step
        """
        for i in range(self.max_iter):
            jacTmp = self.RP.Problem_Jac_func(k0, x0)
            
            JInvF = np.linalg.solve(jacTmp, self.RP.Problem_func(k0,x0)) # F'^{-1} F
            L= np.linalg.solve(jacTmp, self.RP.Problem_H_v_func(k0,x0,JInvF)) # L = F' * F'' * (F'{-1}* F)
            D = np.linalg.solve((self.RP.IDMatrix-(0.5)*L).T,L.T).T       # D =  L * (I-0.5*L)^{-1}
            update = (self.RP.IDMatrix+(0.5)*D)@JInvF #  (I+0.5* D )* (F'^{-1} F)


            k0 = k0 -update

            # check if all components are near zero
            if(np.linalg.norm(self.RP.Problem_func(k0,x0), np.inf) < self.tol):
                self.lastIterationCount=i+1
                return k0
        self.lastIterationCount = self.max_iter
        return k0
    
 
    def solve(self, x0):
        """
        calculets the next time step
        
        :param x0: start value for the next time step
        """
        # solve for K for a given x0
        # Because rootSolver gives a vector with all k_i stacked on top  of each other,
        # we have to reshape it into a matrix where each ith colum  is a k_i vector
        K = ca.reshape(self.rootSolver(ca.DM.zeros(2*self.model.nx),x0), self.model.nx, self.stages)
        
        # update the value for the next time step 
        return x0 + self.dt * (K@self.B)# x0 + dt * sum all k_i*b_i

        



if __name__ == "__main__":

    
    import matplotlib.pyplot as plt
    Problem = Model()
    Problem.nx = 2
    Problem.x = ca.SX.sym("x", Problem.nx)
    Problem.model =ca.vertcat(Problem.x[0], Problem.x[1]**2)
   
    NumIterations= 1000-1
    dt = 0.001
    startCondition = ca.DM([1,1])
    ImplicitRK = IRK(Problem,"Newton",dt,1000)

    F_plot = [startCondition]
    T_plot = [0]
    for i in range(1,NumIterations):
        F_plot += [ImplicitRK.solve(F_plot[i-1])]
        T_plot += [i*dt ]
    
    #plot answer
    Y_Plot1 = np.array( [float(F_plot[i][0]) for i in range(len(F_plot))])
    Y_Plot2 = np.array( [float(F_plot[i][1]) for i in range(len(F_plot))])

    x_plot = np.array(T_plot, dtype=float)
    plt.figure( figsize = (12,4))
    plt.subplot(1,2,1)
    plt.plot(x_plot,Y_Plot1)
    plt.subplot(1,2,2)
    plt.plot(x_plot,Y_Plot2)


    plt.show()
    



    

