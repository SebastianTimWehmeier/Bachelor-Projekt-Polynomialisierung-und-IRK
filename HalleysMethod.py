import casadi as ca
import numpy as np

class ModelWithParameter:
    def __init__(self):
       self.x = None
       self.nx: int = 0
       self.parameter = None
       self.px: int = 0
       self.model =None


# Halley's method is a root solver:
# solves F(x, p) = 0
#

class RootFindingProblem:
    def __init__(self, residual_expr: ca.SX, x: ca.SX, p: ca.SX):
        # self.residual_fun = residual_fun
        # self.


def halleys(model: ModelWithParameter, x0,p, epsilon):
    F_fun = ca.Function('fun',[model.x, model.parameter],[ model.model])
    x_iter = x0

    jac = ca.jacobian(model.model,model.x)#sym F'
    jac_fun = ca.Function("jac_function", [model.x, model.parameter], [jac])#F'

    v_sym = ca.SX.sym("v_sym", model.nx)#sym V

    H = ca.jtimes(jac, model.x, v_sym) # sym F''* V
    H_eval = ca.Function("H", [model.x,model.parameter, v_sym],[H])# function F''* V
    
    I = ca.DM.eye(model.nx) # identity Matrix 

    while True:
        
        # TODO: use factorization for the two solves?
        JacInvF_evalued = np.linalg.solve(jac_fun(x_iter, p), F_fun(x_iter,p)) # F'^{-1} F
        L_evalued= np.linalg.solve(jac_fun(x_iter,p), H_eval(x_iter,p,JacInvF_evalued)) # L = F' * F'' * (F'{-1}* F)

        D = np.linalg.solve((I-(0.5)*L_evalued).T,L_evalued.T).T       # D =  L * (I-0.5*L)^{-1}

        update_evalued = (I+(0.5)*D)@JacInvF_evalued #  (I+0.5* D )* (F'^{-1} F)


        x_iter = x_iter -update_evalued
        answer_vector = F_fun(x_iter,p)
        # check if all components are near zero
        if np.linalg.norm(update_evalued, np.inf) < epsilon:
            break

    return x_iter