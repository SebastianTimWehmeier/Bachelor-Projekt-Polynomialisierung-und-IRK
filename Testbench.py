from IRK import IRK, Model
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import time

def TestBench(P1:Model, P2:Model, numOfsteps:int, dt: float, startCondition1, startCondition2):

    RK1 = IRK(P1,"Newton", dt,10000,1e-12)
    RK2 = IRK(P2, "Halleys",dt, 10000,1e-12)



    F_plot1 = [startCondition1]
    needed_iterations_plot1 = []
    executionTimePlot1 = []

    F_plot2 = [startCondition2]
    needed_iterations_plot2 = []
    executionTimePlot2= []




    T_plot = [0]
    # simulate
    for i in range(1,int(numOfsteps)):
        starttime = time.perf_counter_ns()
        answer1 = RK1.solve(F_plot1[i-1])
        executionTimePlot1 +=[time.perf_counter_ns()-starttime]
        F_plot1 += [answer1]
        needed_iterations_plot1 +=[RK1.lastIterationCount]

        starttime = time.perf_counter_ns()
        answer2 = RK2.solve(F_plot2[i-1])
        executionTimePlot2 +=[ time.perf_counter_ns()-starttime]
        F_plot2 +=[answer2]
        needed_iterations_plot2+=[RK2.lastIterationCount]
        
        T_plot += [i*dt ]

    #plot answer
    Y_Plot1 = np.array( [float(F_plot1[i]) for i in range(len(F_plot1))])
    Y_Plot2 = np.array( [float(F_plot2[i][0]) for i in range(len(F_plot2))])
    Y_Plot3 = np.array( [abs(float(F_plot1[i]-F_plot2[i][0])) for i in range(len(F_plot2))])



    x_plot = np.array(T_plot, dtype=float)
    plt.figure( figsize = (20,10))

    plt.subplot(2,4,1)
    plt.plot(x_plot,Y_Plot1)
    plt.title("solved with Newton")

    plt.subplot(2,4,2)
    plt.plot(x_plot,Y_Plot2)
    plt.title("solved with Halleys")

    plt.subplot(2,4,3)
    plt.plot(x_plot,Y_Plot3)
    plt.title("difference")

    plt.subplot(2,4,4)
    plt.plot(x_plot[1:], needed_iterations_plot1)
    plt.title("iterrations needed (Newton)")

    plt.subplot(2,4,5)
    plt.plot(x_plot[1:], needed_iterations_plot2)
    plt.title("iterrations needed (Halleys)")

    plt.subplot(2,4,6)
    plt.plot(x_plot[1:], executionTimePlot1)
    plt.title("execution time in ns (Newton)")

    plt.subplot(2,4,7)
    plt.plot(x_plot[1:], executionTimePlot2)
    plt.title("execution time in ns (Halleys)")

    plt.show()


if __name__ == "__main__":

    # create System 
    P1 = Model()
    P1.nx = 1
    P1.x = ca.SX.sym("x", P1.nx)
    P1.model = ca.exp(-2*P1.x)+ ca.exp(-4*P1.x)
    # create polynomialized version of the system
    P2 = Model()
    P2.nx =3
    P2.x = ca.SX.sym("xw1w2",P2.nx)
    P2.model = ca.vertcat(P2.x[1]+P2.x[2],(-2)*P2.x[1]*(P2.x[1]+P2.x[2]), (-4)*P2.x[2]*(P2.x[1]+P2.x[2]))


    # simulation hyper parameter
    numOfsteps = 10000
    dt = 0.01
    # start Condition
    startCondition1 = ca.DM(1)
    startCondition2 = ca.DM([1,0.1353352832366127,0.01831563888873418]) # w1 = e^-2*1 = 0.13533... ;  w2 = e^-4*1 = 0.01831... ;
    TestBench(P1,P2,numOfsteps, dt, startCondition1, startCondition2)

