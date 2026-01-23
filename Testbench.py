from IRK import IRK, Model
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import time
from sortedcontainers import SortedList

def TestBench(P1:Model, P2:Model, numOfsteps:int, dt: float, startCondition1, startCondition2):

    RK1 = IRK(P1,"Newton", dt,10000,1e-12,stages=2)
    RK2 = IRK(P2, "Halleys",dt, 10000,1e-12, stages=2)

    # simulate
    F_plot1, T_plot,  needed_iterations_plot1,executionTimePlot1 = simulateFor(RK1, numOfsteps,dt,startCondition1)
    F_plot2, _,needed_iterations_plot2, executionTimePlot2 = simulateFor(RK2, numOfsteps,dt,startCondition2)

    #plot answer
    Y_Plot1 = np.array( [float(F_plot1[i]) for i in range(len(F_plot1))])
    Y_Plot2 = np.array( [float(F_plot2[i][0]) for i in range(len(F_plot2))])
    Y_Plot3 = np.array( [abs(float(F_plot1[i]-F_plot2[i][0])) for i in range(len(F_plot2))])



    x_plot = np.array(T_plot, dtype=float)
    plt.figure( figsize = (14,9))
    



    plt.subplot(2,4,1)
    plt.plot(x_plot,Y_Plot1)
    plt.title("solved with Newton")

    plt.subplot(2,4,2)
    plt.plot(x_plot,Y_Plot2)
    plt.title("solved with Halleys")

    plt.subplot(2,4,3)
    plt.plot(x_plot,Y_Plot3)
    plt.title("difference")

    ax = plt.subplot(2,4,4)
    plt.plot(x_plot[1:], needed_iterations_plot1, label= 'Newton')
    plt.plot(x_plot[1:], needed_iterations_plot2, label = "Halley")
    plt.legend(bbox_to_anchor=(1, 1))
    #plt.ylim(0,3)
    #ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("iterrations needed ")


    textstr = f"on average {(sum(executionTimePlot1)/(1e6*len(executionTimePlot1))):2f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    plt.subplot(2,4,5)
    plt.plot(x_plot[1:], executionTimePlot1)
    plt.gca().text(0.03, 0.95, textstr, transform=plt.gca().transAxes,
                    fontsize=12, verticalalignment='top', bbox=props)
    plt.title("execution time in ns (Newton)")

    plt.subplot(2,4,6)

    plt.plot(x_plot[1:], executionTimePlot2)
    textstr = f"on average {(sum(executionTimePlot2)/(1e6*len(executionTimePlot2))):2f}"
    plt.gca().text(0.03, 0.95, textstr, transform=plt.gca().transAxes,
                    fontsize=12, verticalalignment='top', bbox=props)
    plt.title("execution time in ns (Halleys)")

    plt.show()


def TestBench2(P1:Model, P2:Model, timeToSimulate:int, dt: float, startCondition1, startCondition2):
    RK1 = IRK(P1,"Newton", dt,10000,1e-6,stages=2)
    RK2 = IRK(P2, "Halleys",dt, 10000,1e-6, stages=2)


        
    # simulate
    F_plot1, T_plot, needed_iterations_plot1, executionTimePlot1 = simulateFor(RK1,int(timeToSimulate/dt),dt,startCondition1)
    F_plot2, _, needed_iterations_plot2, executionTimePlot2 = simulateFor(RK2,int(timeToSimulate/dt),dt,startCondition2)
    #plot answer
    originalGraph = np.array( [float(F_plot1[i]) for i in range(len(F_plot1))])
    difference = np.array([abs(float(F_plot1[i]-F_plot2[i][0])) for i in range(len(F_plot2))])

    errorPlot1 = []
    meanTimePlot1 = []
    averageIterationPlot1= []

    errorPlot2 = []
    meanTimePlot2 = []
    averageIterationPlot2 = []

    deltaT = []
    for i in range(2,10):#
        numberOfSteps = 2**i
        print(numberOfSteps)
        RK1.dt = timeToSimulate/numberOfSteps
        RK2.dt = timeToSimulate/numberOfSteps
        deltaT+=[RK1.dt]

        F_plotTmp1, _, iterAverage1, exeTimeMean1 = simulateForReturnMeanTime(RK1, numberOfSteps,RK1.dt,startCondition1)  
        F_plotTmp2, _, iterAverage2, exeTimeMean2 = simulateForReturnMeanTime(RK2, numberOfSteps,RK2.dt,startCondition2)  

        errorPlot1 +=[float(abs(F_plot1[-1]-(F_plotTmp1[-1])))]
        errorPlot2 +=[float(abs(F_plot1[-1]-F_plotTmp2[-1][0]))]

        meanTimePlot1 +=[exeTimeMean1]
        meanTimePlot2 +=[exeTimeMean2]

        averageIterationPlot1 += [iterAverage1]
        averageIterationPlot2 += [iterAverage2]
    T_plot,originalGraph,difference,
    deltaT.reverse()
    errorPlot1.reverse()
    errorPlot2.reverse()
    averageIterationPlot1.reverse()
    averageIterationPlot2.reverse()
    meanTimePlot1.reverse()
    meanTimePlot2.reverse()
    
    pyPlot(T_plot, originalGraph, "Time", "Value", "orginal Graph")
    pyPlot(T_plot, difference, "Time", "Value", "differnce between original System and ")
    pyPlot(deltaT, errorPlot1, "delta T", "error", "differnce depending on the step size (Newton)")
    pyPlot(deltaT, errorPlot2, "delta T", "error", "differnce depending on the step size (Halleys)")





def pyPlot(x,y,x_lable:str, y_lable:str,title: str, row=None):
    y_tmp =y
    plt.plot(x,y_tmp)
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)

    plt.title(title)
    plt.show()

  

def simulateFor(ImRK: IRK,numOfSteps:int, dt:float ,startcondition)-> tuple[list,list,list,list]:
    F_plot = [startcondition]
    executionTimePlot =[]
    needed_iterations_plot =  []
    T_plot = [0]

    for i in range(1,numOfSteps):
        starttime = time.perf_counter_ns()
        answer = ImRK.solve(F_plot[i-1])
        executionTimePlot +=[time.perf_counter_ns()-starttime]
        F_plot += [answer]
        needed_iterations_plot +=[ImRK.lastIterationCount]
       
        T_plot += [i*dt ]

    return F_plot, np.array(T_plot, dtype=float), needed_iterations_plot,executionTimePlot
def simulateForReturnMeanTime(ImRK: IRK,numOfSteps:int, dt:float ,startcondition)-> tuple[list,list,float,float]:
    F_plot = [startcondition]
    executionTimePlot = SortedList()
    needed_iterations_average =  0
    T_plot = [0]

    for i in range(1,numOfSteps):
        starttime = time.perf_counter_ns()
        answer = ImRK.solve(F_plot[i-1])
        executionTimePlot.add(time.perf_counter_ns()-starttime)
        F_plot += [answer]
        needed_iterations_average +=ImRK.lastIterationCount
       
        T_plot += [i*dt ]

    return F_plot, np.array(T_plot, dtype=float),needed_iterations_average/numOfSteps, executionTimePlot[len(executionTimePlot)//2]
    




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
    timesimu = 100
    numOfsteps =5
    dt = timesimu/numOfsteps
    # start Condition
    startCondition1 = ca.DM(1)
    startCondition2 = ca.DM([1,0.1353352832366127,0.01831563888873418]) # w1 = e^-2*1 = 0.13533... ;  w2 = e^-4*1 = 0.01831... ;
    TestBench(P1,P2,numOfsteps, dt, startCondition1, startCondition2)

    #TestBench2(P1,P2,100,10,startCondition1,startCondition2)
    