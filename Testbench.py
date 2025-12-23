from IRK import IRK, Model
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

# create System 
Problem1 = Model()
Problem1.nx = 1
Problem1.x = ca.SX.sym("x", Problem1.nx)
Problem1.model = ca.exp(-2*Problem1.x)+ ca.exp(-4*Problem1.x)
# create polynomialized version of the system
P2 = Model()
P2.nx =3
P2.x = ca.SX.sym("xw1w2",P2.nx)
P2.model = ca.vertcat(P2.x[1]+P2.x[2],(-2)*P2.x[1]*(P2.x[1]+P2.x[2]), (-4)*P2.x[2]*(P2.x[1]+P2.x[2]))


# simulation hyper parameter
simulatedTime = 100
dt = 0.01
RK1 = IRK(Problem1,"Newton", dt,10000)
RK2 = IRK(P2, "Halleys",dt, 10000 )

# start Condition
startCondition1 = ca.DM(1)
startCondition2 = ca.DM([1,0.1353352832366127,0.01831563888873418]) # w1 = e^-2*1 = 0.13533... ;  w2 = e^-4*1 = 0.01831... ;


F_plot1 = [startCondition1]
F_plot2 = [startCondition2]
T_plot = [0]
# simulate
for i in range(1,int(simulatedTime/dt)):
    F_plot1 += [RK1.solve(F_plot1[i-1])]
    F_plot2 +=[RK2.solve(F_plot2[i-1])]
    T_plot += [i*dt ]

#plot answer
Y_Plot1 = np.array( [float(F_plot1[i]) for i in range(len(F_plot1))])
Y_Plot2 = np.array( [float(F_plot2[i][0]) for i in range(len(F_plot2))])
Y_Plot3 = np.array( [abs(float(F_plot1[i]-F_plot2[i][0])) for i in range(len(F_plot2))])


x_plot = np.array(T_plot, dtype=float)
plt.figure( figsize = (12,4))
plt.subplot(1,3,1)
plt.plot(x_plot,Y_Plot1)
plt.subplot(1,3,2)
plt.plot(x_plot,Y_Plot2)
plt.subplot(1,3,3)
plt.plot(x_plot,Y_Plot3)
plt.show()
