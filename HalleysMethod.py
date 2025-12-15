import casadi as ca
import numpy as np

class ModelWithParameter:
    def __init__(self):
       self.x = None
       self.nx: int = 0
       self.parameter = None
       self.px: int = 0
       self.model =None

def halleys(model: ModelWithParameter, x0,p, epsilon):
    F_evalue = ca.Function('evalue',[model.x, model.parameter],[ model.model])
    x_iter = x0

    jac = ca.jacobian(model.model,model.x)#sym F'
    jac_evalue = ca.Function("jac_function", [model.x, model.parameter], [jac])#F'

    v_sym = ca.SX.sym("v_sym", model.nx)#sym V

    H = ca.jtimes(jac, model.x, v_sym) # sym F''* V
    H_eval = ca.Function("H", [model.x,model.parameter, v_sym],[H])# function F''* V
    
    I = ca.DM.eye(model.nx) # identity Matrix 

    while True:
        
        JacInvF_evalued = np.linalg.solve(jac_evalue(x_iter, p), F_evalue(x_iter,p)) # F'^{-1} F
        L_evalued= np.linalg.solve(jac_evalue(x_iter,p), H_eval(x_iter,p,JacInvF_evalued)) # L = F' * F'' * (F'{-1}* F)

        D = np.linalg.solve((I-(0.5)*L_evalued).T,L_evalued.T).T       # D =  L * (I-0.5*L)^{-1}

        update_evalued = (I+(0.5)*D)@JacInvF_evalued #  (I+0.5* D )* (F'^{-1} F)


        x_iter = x_iter -update_evalued
        answer_vector = F_evalue(x_iter,p)
        # check if all components are near zero
        if( all([(True if -epsilon<answer_vector[i]<epsilon else False) for i in range(model.nx)])):
            break

    return x_iter