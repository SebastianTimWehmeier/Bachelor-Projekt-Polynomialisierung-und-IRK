from IRK import IRK, Model
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import time
from sortedcontainers import SortedList
import math

def TestBench(P1:Model, P2:Model, numOfsteps:int, dt: float, startCondition1, startCondition2):

    RK1 = IRK(P1,"Newton", dt,100000,1e-12,stages=2)
    RK2 = IRK(P2, "Halleys",dt, 100000,1e-12, stages=2)

    # simulate
    F_plot1, T_plot,  needed_iterations_plot1,executionTimePlot1 = simulateFor(RK1, numOfsteps,dt,startCondition1)
    F_plot2, _,needed_iterations_plot2, executionTimePlot2 = simulateFor(RK2, numOfsteps,dt,startCondition2)

    #plot answer
    Y_Plot1 = np.array( [float(F_plot1[i][0]) for i in range(len(F_plot1))])
    Y_Plot2 = np.array( [float(F_plot2[i][0]) for i in range(len(F_plot2))])
    Y_Plot3 = np.array( [abs(float(F_plot1[i][0]-F_plot2[i][0])) for i in range(len(F_plot2))])



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
    plt.title("execution time  (Newton)")

    plt.subplot(2,4,6)

    plt.plot(x_plot[1:], executionTimePlot2)
    textstr = f"on average {(sum(executionTimePlot2)/(1e6*len(executionTimePlot2))):2f}"
    plt.gca().text(0.03, 0.95, textstr, transform=plt.gca().transAxes,
                    fontsize=12, verticalalignment='top', bbox=props)
    plt.title("execution time  (Halleys)")

    plt.show()


def TestBench2(P1:Model, P2:Model,P3:Model ,timeToSimulate:int, dt: float, startCondition1, startCondition2):
    RK1 = IRK(P1,"Newton", dt,10000,1e-6,stages=2)
    RK2 = IRK(P2, "Halleys",dt, 10000,1e-6, stages=2)
    RK3 = IRK(P3, "Newton",dt,10000,1e-6,stages=2)

        
    # simulate
    F_plot1, T_plot, _, _ = simulateFor(RK1,int(timeToSimulate/dt),dt,startCondition1)
    F_plot2, _, _, _ = simulateFor(RK2,int(timeToSimulate/dt),dt,startCondition2)
    F_plot3, _, _, _ = simulateFor(RK3,int(timeToSimulate/dt),dt,startCondition2)

    #plot answer
    originalGraph = np.array( [float(F_plot1[i][0]) for i in range(len(F_plot1))])
    polygraphHalley =  np.array( [float(F_plot2[i][0]) for i in range(len(F_plot2))])
    polygraphNewton =  np.array( [float(F_plot3[i][0]) for i in range(len(F_plot3))])

    difference1 = np.array([abs(float(F_plot1[i][0]-F_plot2[i][0])) for i in range(len(F_plot2))])
    difference2 = np.array([abs(float(F_plot1[i][0]-F_plot3[i][0])) for i in range(len(F_plot3))])


    errorPlot1 = []
    meanTimePlot1 = []
    averageIterationPlot1= []

    errorPlot2 = []
    meanTimePlot2 = []
    averageIterationPlot2 = []
    

    errorPlot3 = []
    meanTimePlot3 = []
    averageIterationPlot3 = []

    deltaT = []
    for i in range(1,10):#
        numberOfSteps = 2**i
        print(numberOfSteps)
        RK1.dt = timeToSimulate/numberOfSteps
        RK2.dt = timeToSimulate/numberOfSteps
        RK3.dt = timeToSimulate/numberOfSteps

        deltaT+=[RK1.dt]

        F_plotTmp1, _, iterAverage1, exeTimeMean1 = simulateForReturnMeanTime(RK1, numberOfSteps,RK1.dt,startCondition1)  
        F_plotTmp2, _, iterAverage2, exeTimeMean2 = simulateForReturnMeanTime(RK2, numberOfSteps,RK2.dt,startCondition2)  
        F_plotTmp3, _, iterAverage3, exeTimeMean3 = simulateForReturnMeanTime(RK3, numberOfSteps,RK3.dt,startCondition2)  


        errorPlot1 +=[float(abs(F_plot1[-1][0]-(F_plotTmp1[-1][0])))]
        errorPlot2 +=[float(abs(F_plot1[-1][0]-F_plotTmp2[-1][0]))]
        errorPlot3 +=[float(abs(F_plot1[-1][0]-F_plotTmp3[-1][0]))]



        meanTimePlot1 +=[exeTimeMean1]
        meanTimePlot2 +=[exeTimeMean2]
        meanTimePlot3 +=[exeTimeMean3]


        averageIterationPlot1 += [iterAverage1]
        averageIterationPlot2 += [iterAverage2]
        averageIterationPlot3 += [iterAverage3]

    T_plot,originalGraph,difference1,difference2,
    deltaT.reverse()
    errorPlot1.reverse()
    errorPlot2.reverse()
    errorPlot3.reverse()

    averageIterationPlot1.reverse()
    averageIterationPlot2.reverse()
    averageIterationPlot3.reverse()

    meanTimePlot1.reverse()
    meanTimePlot2.reverse()
    meanTimePlot3.reverse()

    pyPlot(T_plot, originalGraph, "Time", "Value", "orginal Graph")
    pyPlot(T_plot,polygraphHalley , "Time", "Value", "orginal Graph")

    pyPlot2(T_plot, difference1,difference2, "Time", "error","orginal vs. poly. Halley", "orginal vs. poly. newton", "differnce between original System and ")
    pyPlot3(deltaT, errorPlot1,errorPlot2,errorPlot3, "delta T", "error","orginal (Newton)", "poly (Halley)", "poly (Newton)", "differnce depending on the step size ")
    pyPlot3(deltaT, meanTimePlot1,meanTimePlot2,meanTimePlot3, "delta T","mean time","orginal (Newton)", "poly (Halley)", "poly (Newton)" , "mean time per rootsolver iteration" )
    pyPlot3(deltaT, averageIterationPlot1,averageIterationPlot2,averageIterationPlot3, "delta T","average iterations","orginal (Newton)", "poly (Halley)", "poly (Newton)" , "average iterations need by the rootsolver " )





def pyPlot(x,y,x_label:str, y_label:str,title: str, row=None):
    y_tmp =y
    plt.plot(x,y_tmp)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title)
    plt.show()
def pyPlot2(x,y1,y2,x_label:str, y_label:str,y1_label:str, y2_label:str,title: str, row=None):
    
    plt.plot(x,y1,label = y1_label)
    plt.plot(x,y2, label=y2_label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(bbox_to_anchor=(0.5, 1))

    plt.title(title)
    plt.show()
def pyPlot3(x,y1,y2,y3,x_axis_label:str, y_axis_label:str,y1_label:str, y2_label:str,y3_label:str,  title: str):
    
    plt.plot(x,y1,label = y1_label)
    plt.plot(x,y2, label=y2_label)
    plt.plot(x,y3, label =y3_label)


    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend(bbox_to_anchor=(0.5, 1))

    plt.title(title)
    plt.show()

  

def simulateFor(ImRK: IRK,numOfSteps:int, dt:float ,startcondition)-> tuple[list,list,list,list]:
    F_plot = [startcondition]
    executionTimePlot =[]
    needed_iterations_plot =  []
    T_plot = [0]

    for i in range(0,numOfSteps):
        starttime = time.perf_counter_ns()
        answer = ImRK.solve(F_plot[i])
        executionTimePlot +=[time.perf_counter_ns()-starttime]
        F_plot += [answer]
        needed_iterations_plot +=[ImRK.lastIterationCount]
       
        T_plot += [(i+1)*dt ]

    return F_plot, np.array(T_plot, dtype=float), needed_iterations_plot,executionTimePlot
def simulateForReturnMeanTime(ImRK: IRK,numOfSteps:int, dt:float ,startcondition)-> tuple[list,list,float,float]:
    F_plot = [startcondition]
    executionTimePlot = SortedList()
    needed_iterations_average =  0
    T_plot = [0]

    for i in range(0,numOfSteps):
        starttime = time.perf_counter_ns()
        answer = ImRK.solve(F_plot[i])
        executionTimePlot.add(time.perf_counter_ns()-starttime)
        F_plot += [answer]
        needed_iterations_average +=ImRK.lastIterationCount
       
        T_plot += [(i+1)*dt ]

    return F_plot, np.array(T_plot, dtype=float),needed_iterations_average/numOfSteps, executionTimePlot[len(executionTimePlot)//2]
    


def TestBench3(P1:Model, P2:Model, startCondition1, startCondition2):
    evalPoint = [0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50]
    enddiff1 = []
    enddiff2 = []
    enddiff3 = []

    iter1 = []
    iter2 = []
    iter3 = []

    meant1 = []
    meant2 = []
    meant3 = []

    meanTimetimesaverageIter1 = []
    meanTimetimesaverageIter2 = []
    meanTimetimesaverageIter3 = []



    RK0 = IRK(P1,"Newton", 0.01,100000,1e-12,stages=2)

    # simulate
    F_plot0, T_plot,  needed_iterations_plot1,executionTimePlot1 = simulateFor(RK0, 10000,0.01,startCondition1)
    Y_Plot0 = np.array( [float(F_plot0[i][0]) for i in range(len(F_plot0))])

    for dt in evalPoint:
        RK1 = IRK(P1, "Halleys",dt, 100000,1e-12, stages=2)
        RK2 = IRK(P2, "Halleys",dt, 100000,1e-12, stages=2)
        RK3 = IRK(P2, "Newton",dt, 100000,1e-12, stages=2)

        F_plot1, T_plot1, needIter1, meantime1 = simulateForReturnMeanTime(RK1, int(100/dt),dt,startCondition1)
        F_plot2, T_plot2, needIter2, meantime2 = simulateForReturnMeanTime(RK2, int(100/dt),dt,startCondition2)
        F_plot3, T_plot3,needIter3, meantime3 = simulateForReturnMeanTime(RK3, int(100/dt),dt,startCondition2)

        #plot answer
        Y_Plot1 = np.array( [float(F_plot1[i][0]) for i in range(len(F_plot1))])
        Y_Plot2 = np.array( [float(F_plot2[i][0]) for i in range(len(F_plot2))])
        Y_Plot3 = np.array( [float(F_plot3[i][0]) for i in range(len(F_plot3))])

        #Y_Plot3 = np.array( [abs(float(F_plot1[i][0]-F_plot2[i][0])) for i in range(len(F_plot2))])
        enddiff1+=[math.log(abs(float(F_plot0[-1][0])- Y_Plot1[-1]))]
        enddiff2+=[math.log(abs(float(F_plot0[-1][0])- Y_Plot2[-1]))]
        enddiff3+=[math.log(abs(float(F_plot0[-1][0])- Y_Plot3[-1]))]
        iter1 +=[math.log(needIter1)]
        iter2 +=[math.log(needIter2)]
        iter3 +=[math.log(needIter3)]
        meant1 +=[ math.log(meantime1)]
        meant2 +=[ math.log(meantime2)]
        meant3 +=[ math.log(meantime3)]
        meanTimetimesaverageIter1 += [math.log(needIter1*meantime1)]
        meanTimetimesaverageIter2 += [math.log(needIter2*meantime2)]
        meanTimetimesaverageIter3 += [math.log(needIter3*meantime3)]







    x_plot = np.array(T_plot, dtype=float)
    #x_plot2 = np.array(T_plot2, dtype=float)
    x_plot2 = [math.log(i) for i in evalPoint]
    #x_plot2 = np.array(evalPoint, dtype=float)


    


    pyPlot(x_plot,Y_Plot0, "t", "value", "original solution")
    
    pyPlot3(x_plot2,enddiff1,enddiff2,enddiff3,"log(Δt)", "log(error)", "original (Newton)", "poly. (Halley)", "poly. (Newton)", "error depending on the step size")
    

    pyPlot3(x_plot2, iter1,iter2,iter3, "log(Δt)", "log(average iteration)","original (Newton)", "poly. (Halley)", "poly. (Newton)","average iteration depending on the step size" )
   
    pyPlot3(x_plot2, meant1,meant2,meant3, "log(Δt)", "log(ns)","original (Newton)", "poly. (Halley)", "poly. (Newton)","mean time per iteration depending on the step size" )
    
    pyPlot3(x_plot2,meanTimetimesaverageIter1 ,meanTimetimesaverageIter2,meanTimetimesaverageIter3, "log(Δt)", "log(ns * average iteration count)","original (Newton)", "poly. (Halley)", "poly. (Newton)","(mean time * average iteration) depending on the step size" )





if __name__ == "__main__":

    # create System 
    P1 = Model()
    P1.nx = 2
    P1.x = ca.SX.sym("x", P1.nx)
    P1.model =ca.vertcat( ca.exp(-2*P1.x[0])+ ca.exp(-4*P1.x[0]),ca.exp(-2*P1.x[1])+ ca.exp(-4*P1.x[1]) )
    # create polynomialized version of the system (to solve with halleys)
    P2 = Model()
    P2.nx =6
    P2.x = ca.SX.sym("xyw1w2w3w4",P2.nx)
    P2.model = ca.vertcat(P2.x[2]+P2.x[3],P2.x[4]+P2.x[5], -2*P2.x[2]**2-2* P2.x[2]*P2.x[3], -4*P2.x[3]**2-4* P2.x[2]*P2.x[3],
                           -2*P2.x[4]**2-2* P2.x[4]*P2.x[5], -4*P2.x[5]**2-4* P2.x[4]*P2.x[5]  )




    # simulation hyper parameter
    timesimu = 100
    numOfsteps = 3
    dt = timesimu/numOfsteps
    # start Condition
    startCondition1 = ca.DM([1,1])
    startCondition2 = ca.DM([1,1,0.1353352832366127,0.01831563888873418,0.1353352832366127,0.01831563888873418]) # w1 = e^-2*1 = 0.13533... ;  w2 = e^-4*1 = 0.01831... ;
    #TestBench(P1,P2,numOfsteps, dt, startCondition1, startCondition2)

    #TestBench2(P1,P2,P2,100,0.01,startCondition1,startCondition2)
    
    TestBench3(P1,P2, startCondition1, startCondition2)
