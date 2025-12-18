import casadi as ca
import numpy as np
from HalleysMethod import *

class Model:
    def __init__(self):
       self.x = None
       self.nx: int = 0
       self.model = None
        # model(x) = \dot{x}


class IrkSolver:
    def __init__(self, model: Model, root_solver_type: str, dt: float,
                 max_iter: int = 100, tol: float = 1e-6,
                 degree: int = 2, ):
        #
        # self.model = model
        # TODO: create casadi functions
        # degree -> stages
        raise NotImplementedError()


    def solve(self, x0):
        # simulates for dt
        return x_next


def IRK(function: Model, x0, h, number_of_time_steps, root_solver, degree):

    answer_ft = [None]*number_of_time_steps
    answer_t  = [None]*number_of_time_steps

    f_iter = x0
    answer_ft[0] = f_iter
    answer_t[0]  = 0

    A = ca.DM([[1/4, 1/4],
               [1/4, 1/4]])
    B = ca.DM([1/2, 1/2])


    evalFunction = ca.Function("f", [function.x], [function.model])


    # create model to calcuate the parameters K 
    K = ModelWithParameter()
    K.nx = 2 * function.nx
    K.x  =ca.vertcat(ca.SX.sym("k_1",function.nx ), ca.SX.sym("k_2",function.nx ))
    K.px = function.nx
    K.parameter = ca.SX.sym("f_sym", function.nx)
    k1 = K.x[0:function.nx]
    k2 = K.x[function.nx:2*function.nx]

    K.model = ca.vertcat(
        k1 - evalFunction(K.parameter + h*(A[0,0]*k1 + A[0,1]*k2)),
        k2 - evalFunction(K.parameter+ h*(A[1,0]*k1 + A[1,1]*k2))
    )

    for i in range(1, number_of_time_steps):

        # solve for K for a given f_iter
        K_values = root_solver(K, ca.DM.zeros(2*function.nx), ca.DM(f_iter), 0.1)

        # update the value for the next time step 
        f_iter = f_iter + h * (B[0] * K_values[0] + B[1] * K_values[1])

        answer_ft[i] = f_iter
        answer_t[i]  = i*h

    return answer_ft, answer_t
