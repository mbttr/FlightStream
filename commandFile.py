# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:50:23 2021

@author: bmoor
"""

import os
import wingClass
import sceneClass
import writeFlightStreamScript
import calculateParameters
import readFlightStreamOutput
import writeCSV
import numpy as np
import subprocess
import time
import datetime
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from shutil import copyfile

class iterCounter:
    def __init__(self):
        self.count = 1

def aeroEfficiency(x, cons):
    # print(x)
    wing = wingClass.wing(RA[0], RT[0], Lambda, naca, writeGeom, folder, n_span, n_airfoil, N, curvetype, x)
    wing.CSVFile = folder+wing.filename+"_iter"+str(iterCount.count)+".csv"
    writeCSV.writeCSV(wing)
    outputFile = folder+"outputFile_"+wing.filename+"_iter"+str(iterCount.count)+".txt"
    savedRunFile = folder+"savedRunFile_"+wing.filename+".fsm"
    scriptFile = folder+"scriptFlightStream_"+wing.filename+"_iter"+str(iterCount.count)+".txt"
    scene = sceneClass.scene(alpha[0], V_inf, outputFile, savedRunFile, simulationTemplate)
    if writeScripts == 'y':
        writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
    if runFS == 'y':
        os.chdir(folderFS)
        p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
        os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code')
    if readOutput == 'y':
        CL, CDi, Cm, numIter = readFlightStreamOutput.readFlightStreamOutput(outputFile)
    if calcParameters == 'y':
        kappaD, xAC = calculateParameters.calculateParameters(CL,CDi,Cm,wing,scene)
    
    cons.xAC = xAC
    cons.kAC = cons.xAC - cons.xAC0
    
    print(iterCount.count,x,cons.kAC,kappaD)
    iterCount.count += 1
    
    return kappaD

tstart = time.time()
now = datetime.datetime.now()

RA = np.linspace(4,20,9)
RT = np.linspace(0.25,1,4)
Lambda = np.linspace(0,40,41)
naca = '0012'
alpha = np.linspace(5,-5,1)
V_inf = 10

n_span = 160
n_airfoil = 80

N_sections = np.array([32])
curvetype = 'linear'
# if curvetype == 'defined':
#     N_sections = np.array([12])

writeGeom = 'n'
writeScripts = 'n'
runFS = 'n'
readOutput = 'y'
calcParameters = 'y'
alpharange = 'n'

folder = "C:/Users/Bruno-USU/Desktop/Research/FlightStream_Data/Final_Data_July/Crescent/"#"C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/"+str(now.month)+"-"+str(now.day)+"/Run_Time_"+str(now.hour)+"-"+str(now.minute)+"-"+str(now.second)+"/"
folderFS = "C:/Users/Bruno-USU/Desktop/FlightStream_June21/"
# os.makedirs(folder)

simulationTemplate = "C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/simulationTemplate.fsm"


if curvetype == "optimize":
    copyfile('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/commandFile.py', folder+'commandFile.py')
    N = N_sections[0]
     
    class constClass:
        def __init__(self,xAC0,xAC):
            self.xAC0 = xAC0
            self.xAC = xAC
            self.kAC = self.xAC-self.xAC0
    
        def constraintFun1(self,x):
            # print("constraint 1",-self.kAC + 0.3)
            # return -self.kAC + 0.4 
            b=0.4
            a=0.35
            return abs((b-a)/2) - abs(self.kAC - (a+b)/2)
    
        def constraintFun2(self,x):
            # print("constraint 2",self.kAC - 0.27)
            return self.kAC - 0.35 
    
    wing0 = wingClass.wing(RA[0], RT[0], 0, naca, writeGeom, folder, n_span, n_airfoil, N, 'linear', 0)
    writeCSV.writeCSV(wing0)
    outputFile0 = folder+"outputFile_"+wing0.filename+".txt"
    savedRunFile0 = folder+"savedRunFile_"+wing0.filename+".fsm"
    scriptFile0 = folder+"scriptFlightStream_"+wing0.filename+".txt"
    scene0 = sceneClass.scene(alpha[0], V_inf, outputFile0, savedRunFile0, simulationTemplate)
    if writeScripts == 'y':
        writeFlightStreamScript.writeFlightStreamScript(wing0,scene0,scriptFile0)
    if runFS == 'y':
        os.chdir(folderFS)
        p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile0)
        os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code')
    if readOutput == 'y':
        CL0, CDi0, Cm0, numIter = readFlightStreamOutput.readFlightStreamOutput(outputFile0)
    if calcParameters == 'y':
        kappaD0, xAC0 = calculateParameters.calculateParameters(CL0,CDi0,Cm0,wing0,scene0)
    
    cons = constClass(xAC0,xAC0)

    # array([0.65556915, 0.62690503, 0.45271761])
    # x0 = np.zeros(N-1)
    x0 = np.array([0.02728809, 0.10936083, 0.24684843, 0.44081745])
    
    const = ({'type': 'ineq',
              'fun': cons.constraintFun1
              })#,{'type': 'ineq',
              #'fun': cons.constraintFun2
             #})
                 
    bnds = ((-2,2),(-2,2),(-2,2),(-2,2))
    
    iterCount = iterCounter()
        
    x = optimize.minimize(aeroEfficiency, x0, args=(cons), constraints = const, #bounds=bnds,
                                                  tol=1e-3,
                                                  method='SLSQP').x

    # BASINHOPPING
    # minimizer_kwargs={"method":"SLSQP","constraints":const,"bounds":bnds,"args":(cons),"tol":1e-3}
    # x_from0 = optimize.basinhopping(aeroEfficiency, x0, minimizer_kwargs=minimizer_kwargs, niter=10, disp = True).x 
    
    # print("xfrom0",x_from0)
    
    # iterCount = iterCounter()
    
    # x0 = np.tan(np.deg2rad(10))*np.linspace(0,wing0.b_semi,N)
    # x0 = x0[1:]
    # x_from10 = optimize.basinhopping(aeroEfficiency, x0, minimizer_kwargs=minimizer_kwargs, niter=10, disp = True).x

    # print("xfrom10",x_from10)

    # x0 = np.tan(np.deg2rad(20))*np.linspace(0,wing0.b_semi,N)
    # x0 = x0[1:]
    # x_from20 = optimize.basinhopping(aeroEfficiency, x0, minimizer_kwargs=minimizer_kwargs, niter=10, disp = True).x

if curvetype == "optimizequad":
    copyfile('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/commandFile.py', folder+'commandFile.py')
    N = N_sections[0]
     
    class constClass:
        def __init__(self,xAC0,xAC):
            self.xAC0 = xAC0
            self.xAC = xAC
            self.kAC = self.xAC-self.xAC0
    
        def constraintFun1(self,x):
            # print("constraint 1",-self.kAC + 0.3)
            # return -self.kAC + 0.4 
            b=0.4
            a=0.35
            return abs((b-a)/2) - abs(self.kAC - (a+b)/2)
    
        def constraintFun2(self,x):
            # print("constraint 2",self.kAC - 0.27)
            return self.kAC - 0.35 
    
    wing0 = wingClass.wing(RA[0], RT[0], 0, naca, writeGeom, folder, n_span, n_airfoil, N, 'linear', 0)
    writeCSV.writeCSV(wing0)
    outputFile0 = folder+"outputFile_"+wing0.filename+".txt"
    savedRunFile0 = folder+"savedRunFile_"+wing0.filename+".fsm"
    scriptFile0 = folder+"scriptFlightStream_"+wing0.filename+".txt"
    scene0 = sceneClass.scene(alpha[0], V_inf, outputFile0, savedRunFile0, simulationTemplate)
    if writeScripts == 'y':
        writeFlightStreamScript.writeFlightStreamScript(wing0,scene0,scriptFile0)
    if runFS == 'y':
        os.chdir(folderFS)
        p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile0)
        os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code')
    if readOutput == 'y':
        CL0, CDi0, Cm0, numIter = readFlightStreamOutput.readFlightStreamOutput(outputFile0)
    if calcParameters == 'y':
        kappaD0, xAC0 = calculateParameters.calculateParameters(CL0,CDi0,Cm0,wing0,scene0)
    
    cons = constClass(xAC0,xAC0)

    # array([0.65556915, 0.62690503, 0.45271761])
    # x0 = np.zeros(N-1)
    x0 = np.array([0, 0.17, 0])
    
    const = ({'type': 'ineq',
              'fun': cons.constraintFun1
              })#,{'type': 'ineq',
              #'fun': cons.constraintFun2
             #})
                 
    bnds = ((-5,5),(-5,5))
    
    iterCount = iterCounter()
        
    # x = optimize.minimize(aeroEfficiency, x0, args=(cons), constraints = const, bounds=bnds,
    #                                               tol=1e-3,
    #                                               method='SLSQP').x
    
    minimizer_kwargs={"method":"SLSQP","constraints":const,"args":(cons),"tol":1e-3} #,"bounds":bnds
    x = optimize.basinhopping(aeroEfficiency, x0, minimizer_kwargs=minimizer_kwargs, niter=10, disp = True).x 
    
elif curvetype == "defined":

    print('joepie')

    x = np.array([-0.46843379,  0.41728863,  0.65977313, -0.08540865])
    
    if alpharange == 'y':
        CL = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        CDi = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        Cm = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        numIter = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        
        kappaD = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        xAC_c = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        kappaAC = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))

        for i in range(len(RA)):
            for j in range(len(RT)):
                for k in range(len(Lambda)):        
                    for n in range(len(alpha)):
                        print(RA[i], RT[j], Lambda[k], alpha[n])
                        wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, n_airfoil, N_sections[0], curvetype, x)
                        writeCSV.writeCSV(wing)
                        outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_alpha"+str(alpha[n])+".txt"
                        savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_alpha"+str(alpha[n])+".fsm"
                        scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_alpha"+str(alpha[n])+".txt"
                        scene = sceneClass.scene(alpha[n], V_inf, outputFile, savedRunFile, simulationTemplate)
                        if writeScripts == 'y':
                            writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
                        if runFS == 'y':
                            os.chdir(folderFS)
                            p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
                            os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
                        if readOutput == 'y':
                            CL[i,j,k,n], CDi[i,j,k,n], Cm[i,j,k,n], numIter[i,j,k,n] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
                        if calcParameters == 'y':
                            kappaD[i,j,k,n], xAC_c[i,j,k,n] = calculateParameters.calculateParameters(CL[i,j,k,n],CDi[i,j,k,n],Cm[i,j,k,n],wing,scene)
                            kappaAC[i,j,k,n] = (xAC_c[i,j,k,n] - xAC_c[i,j,0,n])

    elif alpharange == 'n':

        wing0 = wingClass.wing(RA[0], RT[0], Lambda[0], naca, writeGeom, folder, n_span, n_airfoil, N_sections[0], "linear", 0)
        writeCSV.writeCSV(wing0)
        outputFile0 = folder+"outputFile_"+wing0.filename+".txt"
        savedRunFile0 = folder+"savedRunFile_"+wing0.filename+".fsm"
        scriptFile0 = folder+"scriptFlightStream_"+wing0.filename+".txt"
        scene0 = sceneClass.scene(alpha[0], V_inf, outputFile0, savedRunFile0, simulationTemplate)
        if writeScripts == 'y':
            writeFlightStreamScript.writeFlightStreamScript(wing0,scene0,scriptFile0)
        if runFS == 'y':
            os.chdir(folderFS)
            p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile0)
            os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code')
        if readOutput == 'y':
            CL0, CDi0, Cm0, numIter = readFlightStreamOutput.readFlightStreamOutput(outputFile0)
        if calcParameters == 'y':
            kappaD0, xAC0 = calculateParameters.calculateParameters(CL0,CDi0,Cm0,wing0,scene0)
    
    
        
        print(RA[0], RT[0])
        wing = wingClass.wing(RA[0], RT[0], Lambda[0], naca, writeGeom, folder, n_span, n_airfoil, N_sections[0], curvetype, x)
        writeCSV.writeCSV(wing)
        outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_xDefined.txt"
        savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_xDefined.fsm"
        scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_xDefined.txt"
        scene = sceneClass.scene(alpha[0], V_inf, outputFile, savedRunFile, simulationTemplate)
        if writeScripts == 'y':
            writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
        if runFS == 'y':
            os.chdir(folderFS)
            p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
            os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
        if readOutput == 'y':
            CL, CDi, Cm, numIter = readFlightStreamOutput.readFlightStreamOutput(outputFile)
        if calcParameters == 'y':
            kappaD, xAC_c = calculateParameters.calculateParameters(CL,CDi,Cm,wing,scene)
            kappaAC = (xAC_c - xAC0)      
        
        print(wing.x_quarterChord,"kappaAC: ",kappaAC,"kappaD: ",kappaD)

else: 
    if alpharange == 'n':
        CL = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
        CDi = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
        Cm = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
        numIter = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
        
        kappaD = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
        kappaL = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
        xAC_c = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
        kappaAC = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
        
        for i in range(len(RA)):
            for j in range(len(RT)):
                for k in range(len(Lambda)):
                    for n in range(len(N_sections)):
                        print(RA[i], RT[j], Lambda[k])
                        wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, n_airfoil, N_sections[n], curvetype, 0)
                        writeCSV.writeCSV(wing)
                        outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
                        savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
                        scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
                        scene = sceneClass.scene(alpha[0], V_inf, outputFile, savedRunFile, simulationTemplate)
                        if writeScripts == 'y':
                            writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
                        if runFS == 'y':
                            os.chdir(folderFS)
                            p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
                            os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
                        if readOutput == 'y':
                            CL[i,j,k,n], CDi[i,j,k,n], Cm[i,j,k,n], numIter[i,j,k,n] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
                        if calcParameters == 'y':
                            kappaD[i,j,k,n], xAC_c[i,j,k,n], kappaL[i,j,k,n] = calculateParameters.calculateParameters(CL[i,j,k,n],CDi[i,j,k,n],Cm[i,j,k,n],wing,scene)
                            kappaAC[i,j,k,n] = (xAC_c[i,j,k,n] - xAC_c[i,j,0,n])
        
        print(wing.x_quarterChord[0],"kappaAC: ",kappaAC,"kappaD: ",kappaD)
    elif alpharange == 'y':
        CL = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        CDi = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        Cm = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        numIter = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        
        kappaD = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        xAC_c = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        kappaAC = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        
        for i in range(len(RA)):
            for j in range(len(RT)):
                for k in range(len(Lambda)):
                    for n in range(len(alpha)):
                        print(RA[i], RT[j], Lambda[k], alpha[n])
                        wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, n_airfoil, N_sections[0], curvetype, 0)
                        writeCSV.writeCSV(wing)
                        outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
                        savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
                        scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
                        scene = sceneClass.scene(alpha[n], V_inf, outputFile, savedRunFile, simulationTemplate)
                        if writeScripts == 'y':
                            writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
                        if runFS == 'y':
                            os.chdir(folderFS)
                            p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
                            os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
                        if readOutput == 'y':
                            CL[i,j,k,n], CDi[i,j,k,n], Cm[i,j,k,n], numIter[i,j,k,n] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
                        if calcParameters == 'y':
                            kappaD[i,j,k,n], xAC_c[i,j,k,n], kappaL[i,j,k,n] = calculateParameters.calculateParameters(CL[i,j,k,n],CDi[i,j,k,n],Cm[i,j,k,n],wing,scene)
                            kappaAC[i,j,k,n] = (xAC_c[i,j,k,n] - xAC_c[i,j,0,n])
        
        print(wing.x_quarterChord[0],alpha[n],"kappaAC: ",kappaAC,"kappaD: ",kappaD)            

# plt.plot(wing.x,wing.y)

# print("CL: ",CL,"CDi: ",CDi,"Cm: ",Cm)
        
np.save(folder+"CL.npy",CL)
np.save(folder+"CDi.npy",CDi)
np.save(folder+"Cm.npy",Cm)
np.save(folder+"numIter.npy",numIter)
np.save(folder+"kappaD.npy",kappaD)
np.save(folder+"kappaL.npy",kappaL)
np.save(folder+"kappaAC.npy",kappaAC)
np.save(folder+"xAC_c.npy",xAC_c)



# now = datetime.datetime.now()



# RA = np.linspace(8,18,1)
# RT = np.linspace(0.25,1,1)
# Lambda = np.linspace(6.19,40,1)
# naca = '0012'
# alpha = np.linspace(4.02963527929869,-5,1)
# V_inf = 10

# n_span = 160
# n_airfoil = 80

# N_sections = np.array([5])
# curvetype = 'defined'
# # if curvetype == 'defined':
# #     N_sections = np.array([12])

# writeGeom = 'y'
# writeScripts = 'y'
# runFS = 'y'
# readOutput = 'y'
# calcParameters = 'y'
# alpharange = 'y'

# folder = "C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/"+str(now.month)+"-"+str(now.day)+"/Run_Time_"+str(now.hour)+"-"+str(now.minute)+"-"+str(now.second)+"/"
# folderFS = "C:/Users/Bruno-USU/Desktop/FlightStream_June21/"
# os.makedirs(folder)

# simulationTemplate = "C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/simulationTemplate.fsm"


# if curvetype == "optimize":
#     copyfile('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/commandFile.py', folder+'commandFile.py')
#     N = N_sections[0]
     
#     class constClass:
#         def __init__(self,xAC0,xAC):
#             self.xAC0 = xAC0
#             self.xAC = xAC
#             self.kAC = self.xAC-self.xAC0
    
#         def constraintFun1(self,x):
#             # print("constraint 1",-self.kAC + 0.3)
#             # return -self.kAC + 0.4 
#             b=0.4
#             a=0.35
#             return abs((b-a)/2) - abs(self.kAC - (a+b)/2)
    
#         def constraintFun2(self,x):
#             # print("constraint 2",self.kAC - 0.27)
#             return self.kAC - 0.35 
    
#     wing0 = wingClass.wing(RA[0], RT[0], 0, naca, writeGeom, folder, n_span, n_airfoil, N, 'linear', 0)
#     writeCSV.writeCSV(wing0)
#     outputFile0 = folder+"outputFile_"+wing0.filename+".txt"
#     savedRunFile0 = folder+"savedRunFile_"+wing0.filename+".fsm"
#     scriptFile0 = folder+"scriptFlightStream_"+wing0.filename+".txt"
#     scene0 = sceneClass.scene(alpha[0], V_inf, outputFile0, savedRunFile0, simulationTemplate)
#     if writeScripts == 'y':
#         writeFlightStreamScript.writeFlightStreamScript(wing0,scene0,scriptFile0)
#     if runFS == 'y':
#         os.chdir(folderFS)
#         p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile0)
#         os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code')
#     if readOutput == 'y':
#         CL0, CDi0, Cm0, numIter = readFlightStreamOutput.readFlightStreamOutput(outputFile0)
#     if calcParameters == 'y':
#         kappaD0, xAC0 = calculateParameters.calculateParameters(CL0,CDi0,Cm0,wing0,scene0)
    
#     cons = constClass(xAC0,xAC0)

#     # array([0.65556915, 0.62690503, 0.45271761])
#     # x0 = np.zeros(N-1)
#     x0 = np.array([0.02728809, 0.10936083, 0.24684843, 0.44081745])
    
#     const = ({'type': 'ineq',
#               'fun': cons.constraintFun1
#               })#,{'type': 'ineq',
#               #'fun': cons.constraintFun2
#               #})
                 
#     bnds = ((-2,2),(-2,2),(-2,2),(-2,2))
    
#     iterCount = iterCounter()
        
#     x = optimize.minimize(aeroEfficiency, x0, args=(cons), constraints = const, #bounds=bnds,
#                                                   tol=1e-3,
#                                                   method='SLSQP').x

#     # BASINHOPPING
#     # minimizer_kwargs={"method":"SLSQP","constraints":const,"bounds":bnds,"args":(cons),"tol":1e-3}
#     # x_from0 = optimize.basinhopping(aeroEfficiency, x0, minimizer_kwargs=minimizer_kwargs, niter=10, disp = True).x 
    
#     # print("xfrom0",x_from0)
    
#     # iterCount = iterCounter()
    
#     # x0 = np.tan(np.deg2rad(10))*np.linspace(0,wing0.b_semi,N)
#     # x0 = x0[1:]
#     # x_from10 = optimize.basinhopping(aeroEfficiency, x0, minimizer_kwargs=minimizer_kwargs, niter=10, disp = True).x

#     # print("xfrom10",x_from10)

#     # x0 = np.tan(np.deg2rad(20))*np.linspace(0,wing0.b_semi,N)
#     # x0 = x0[1:]
#     # x_from20 = optimize.basinhopping(aeroEfficiency, x0, minimizer_kwargs=minimizer_kwargs, niter=10, disp = True).x

# if curvetype == "optimizequad":
#     copyfile('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/commandFile.py', folder+'commandFile.py')
#     N = N_sections[0]
     
#     class constClass:
#         def __init__(self,xAC0,xAC):
#             self.xAC0 = xAC0
#             self.xAC = xAC
#             self.kAC = self.xAC-self.xAC0
    
#         def constraintFun1(self,x):
#             # print("constraint 1",-self.kAC + 0.3)
#             # return -self.kAC + 0.4 
#             b=0.4
#             a=0.35
#             return abs((b-a)/2) - abs(self.kAC - (a+b)/2)
    
#         def constraintFun2(self,x):
#             # print("constraint 2",self.kAC - 0.27)
#             return self.kAC - 0.35 
    
#     wing0 = wingClass.wing(RA[0], RT[0], 0, naca, writeGeom, folder, n_span, n_airfoil, N, 'linear', 0)
#     writeCSV.writeCSV(wing0)
#     outputFile0 = folder+"outputFile_"+wing0.filename+".txt"
#     savedRunFile0 = folder+"savedRunFile_"+wing0.filename+".fsm"
#     scriptFile0 = folder+"scriptFlightStream_"+wing0.filename+".txt"
#     scene0 = sceneClass.scene(alpha[0], V_inf, outputFile0, savedRunFile0, simulationTemplate)
#     if writeScripts == 'y':
#         writeFlightStreamScript.writeFlightStreamScript(wing0,scene0,scriptFile0)
#     if runFS == 'y':
#         os.chdir(folderFS)
#         p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile0)
#         os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code')
#     if readOutput == 'y':
#         CL0, CDi0, Cm0, numIter = readFlightStreamOutput.readFlightStreamOutput(outputFile0)
#     if calcParameters == 'y':
#         kappaD0, xAC0 = calculateParameters.calculateParameters(CL0,CDi0,Cm0,wing0,scene0)
    
#     cons = constClass(xAC0,xAC0)

#     # array([0.65556915, 0.62690503, 0.45271761])
#     # x0 = np.zeros(N-1)
#     x0 = np.array([0, 0.17, 0])
    
#     const = ({'type': 'ineq',
#               'fun': cons.constraintFun1
#               })#,{'type': 'ineq',
#               #'fun': cons.constraintFun2
#               #})
                 
#     bnds = ((-5,5),(-5,5))
    
#     iterCount = iterCounter()
        
#     # x = optimize.minimize(aeroEfficiency, x0, args=(cons), constraints = const, bounds=bnds,
#     #                                               tol=1e-3,
#     #                                               method='SLSQP').x
    
#     minimizer_kwargs={"method":"SLSQP","constraints":const,"args":(cons),"tol":1e-3} #,"bounds":bnds
#     x = optimize.basinhopping(aeroEfficiency, x0, minimizer_kwargs=minimizer_kwargs, niter=10, disp = True).x 
    
# elif curvetype == "defined":

#     print('joepie')

#     x = np.array([-0.53174215,  0.28958347,  0.70561597,  0.47835169])
    
#     if alpharange == 'y':
#         CL = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         CDi = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         Cm = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         numIter = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        
#         kappaD = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         xAC_c = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         kappaAC = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))

#         for i in range(len(RA)):
#             for j in range(len(RT)):
#                 for k in range(len(Lambda)):        
#                     for n in range(len(alpha)):
#                         print(RA[i], RT[j], Lambda[k], alpha[n])
#                         wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, n_airfoil, N_sections[0], curvetype, x)
#                         writeCSV.writeCSV(wing)
#                         outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_alpha"+str(alpha[n])+".txt"
#                         savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_alpha"+str(alpha[n])+".fsm"
#                         scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_alpha"+str(alpha[n])+".txt"
#                         scene = sceneClass.scene(alpha[n], V_inf, outputFile, savedRunFile, simulationTemplate)
#                         if writeScripts == 'y':
#                             writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
#                         if runFS == 'y':
#                             os.chdir(folderFS)
#                             p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
#                             os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
#                         if readOutput == 'y':
#                             CL[i,j,k,n], CDi[i,j,k,n], Cm[i,j,k,n], numIter[i,j,k,n] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
#                         if calcParameters == 'y':
#                             kappaD[i,j,k,n], xAC_c[i,j,k,n] = calculateParameters.calculateParameters(CL[i,j,k,n],CDi[i,j,k,n],Cm[i,j,k,n],wing,scene)
#                             kappaAC[i,j,k,n] = (xAC_c[i,j,k,n] - xAC_c[i,j,0,n])

#     elif alpharange == 'n':

#         wing0 = wingClass.wing(RA[0], RT[0], Lambda[0], naca, writeGeom, folder, n_span, n_airfoil, N_sections[0], "linear", 0)
#         writeCSV.writeCSV(wing0)
#         outputFile0 = folder+"outputFile_"+wing0.filename+".txt"
#         savedRunFile0 = folder+"savedRunFile_"+wing0.filename+".fsm"
#         scriptFile0 = folder+"scriptFlightStream_"+wing0.filename+".txt"
#         scene0 = sceneClass.scene(alpha[0], V_inf, outputFile0, savedRunFile0, simulationTemplate)
#         if writeScripts == 'y':
#             writeFlightStreamScript.writeFlightStreamScript(wing0,scene0,scriptFile0)
#         if runFS == 'y':
#             os.chdir(folderFS)
#             p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile0)
#             os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code')
#         if readOutput == 'y':
#             CL0, CDi0, Cm0, numIter = readFlightStreamOutput.readFlightStreamOutput(outputFile0)
#         if calcParameters == 'y':
#             kappaD0, xAC0 = calculateParameters.calculateParameters(CL0,CDi0,Cm0,wing0,scene0)
    
    
        
#         print(RA[0], RT[0])
#         wing = wingClass.wing(RA[0], RT[0], Lambda[0], naca, writeGeom, folder, n_span, n_airfoil, N_sections[0], curvetype, x)
#         writeCSV.writeCSV(wing)
#         outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_xDefined.txt"
#         savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_xDefined.fsm"
#         scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_xDefined.txt"
#         scene = sceneClass.scene(alpha[0], V_inf, outputFile, savedRunFile, simulationTemplate)
#         if writeScripts == 'y':
#             writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
#         if runFS == 'y':
#             os.chdir(folderFS)
#             p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
#             os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
#         if readOutput == 'y':
#             CL, CDi, Cm, numIter = readFlightStreamOutput.readFlightStreamOutput(outputFile)
#         if calcParameters == 'y':
#             kappaD, xAC_c = calculateParameters.calculateParameters(CL,CDi,Cm,wing,scene)
#             kappaAC = (xAC_c - xAC0)      
        
#         print(wing.x_quarterChord,"kappaAC: ",kappaAC,"kappaD: ",kappaD)

# else: 
#     if alpharange == 'n':
#         CL = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
#         CDi = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
#         Cm = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
#         numIter = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
        
#         kappaD = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
#         xAC_c = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
#         kappaAC = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
        
#         for i in range(len(RA)):
#             for j in range(len(RT)):
#                 for k in range(len(Lambda)):
#                     for n in range(len(N_sections)):
#                         print(RA[i], RT[j], Lambda[k])
#                         wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, n_airfoil, N_sections[n], curvetype, 0)
#                         writeCSV.writeCSV(wing)
#                         outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
#                         savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
#                         scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
#                         scene = sceneClass.scene(alpha[0], V_inf, outputFile, savedRunFile, simulationTemplate)
#                         if writeScripts == 'y':
#                             writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
#                         if runFS == 'y':
#                             os.chdir(folderFS)
#                             p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
#                             os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
#                         if readOutput == 'y':
#                             CL[i,j,k,n], CDi[i,j,k,n], Cm[i,j,k,n], numIter[i,j,k,n] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
#                         if calcParameters == 'y':
#                             kappaD[i,j,k,n], xAC_c[i,j,k,n] = calculateParameters.calculateParameters(CL[i,j,k,n],CDi[i,j,k,n],Cm[i,j,k,n],wing,scene)
#                             kappaAC[i,j,k,n] = (xAC_c[i,j,k,n] - xAC_c[i,j,0,n])
        
#         print(wing.x_quarterChord[0],"kappaAC: ",kappaAC,"kappaD: ",kappaD)
#     elif alpharange == 'y':
#         CL = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         CDi = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         Cm = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         numIter = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        
#         kappaD = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         xAC_c = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         kappaAC = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        
#         for i in range(len(RA)):
#             for j in range(len(RT)):
#                 for k in range(len(Lambda)):
#                     for n in range(len(alpha)):
#                         print(RA[i], RT[j], Lambda[k], alpha[n])
#                         wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, n_airfoil, N_sections[0], curvetype, 0)
#                         writeCSV.writeCSV(wing)
#                         outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
#                         savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
#                         scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
#                         scene = sceneClass.scene(alpha[n], V_inf, outputFile, savedRunFile, simulationTemplate)
#                         if writeScripts == 'y':
#                             writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
#                         if runFS == 'y':
#                             os.chdir(folderFS)
#                             p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
#                             os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
#                         if readOutput == 'y':
#                             CL[i,j,k,n], CDi[i,j,k,n], Cm[i,j,k,n], numIter[i,j,k,n] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
#                         if calcParameters == 'y':
#                             kappaD[i,j,k,n], xAC_c[i,j,k,n] = calculateParameters.calculateParameters(CL[i,j,k,n],CDi[i,j,k,n],Cm[i,j,k,n],wing,scene)
#                             kappaAC[i,j,k,n] = (xAC_c[i,j,k,n] - xAC_c[i,j,0,n])
        
#         print(wing.x_quarterChord[0],alpha[n],"kappaAC: ",kappaAC,"kappaD: ",kappaD)            

# # plt.plot(wing.x,wing.y)

# # print("CL: ",CL,"CDi: ",CDi,"Cm: ",Cm)
        
# np.save(folder+"CL.npy",CL)
# np.save(folder+"CDi.npy",CDi)
# np.save(folder+"Cm.npy",Cm)
# np.save(folder+"numIter.npy",numIter)
# np.save(folder+"kappaD.npy",kappaD)
# np.save(folder+"kappaAC.npy",kappaAC)
# np.save(folder+"xAC_c.npy",xAC_c)







# now = datetime.datetime.now()






# RA = np.linspace(8,18,1)
# RT = np.linspace(0.25,1,1)
# Lambda = np.linspace(0,9.5,1)
# naca = '0012'
# alpha = np.linspace(4.1489099632138355,-5,1)
# V_inf = 10

# n_span = 160
# n_airfoil = 80

# N_sections = np.array([5])
# curvetype = 'defined'
# # if curvetype == 'defined':
# #     N_sections = np.array([12])

# writeGeom = 'y'
# writeScripts = 'y'
# runFS = 'y'
# readOutput = 'y'
# calcParameters = 'y'
# alpharange = 'y'

# folder = "C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/"+str(now.month)+"-"+str(now.day)+"/Run_Time_"+str(now.hour)+"-"+str(now.minute)+"-"+str(now.second)+"/"
# folderFS = "C:/Users/Bruno-USU/Desktop/FlightStream_June21/"
# os.makedirs(folder)

# simulationTemplate = "C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/simulationTemplate.fsm"


# if curvetype == "optimize":
#     copyfile('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/commandFile.py', folder+'commandFile.py')
#     N = N_sections[0]
     
#     class constClass:
#         def __init__(self,xAC0,xAC):
#             self.xAC0 = xAC0
#             self.xAC = xAC
#             self.kAC = self.xAC-self.xAC0
    
#         def constraintFun1(self,x):
#             # print("constraint 1",-self.kAC + 0.3)
#             # return -self.kAC + 0.4 
#             b=0.4
#             a=0.35
#             return abs((b-a)/2) - abs(self.kAC - (a+b)/2)
    
#         def constraintFun2(self,x):
#             # print("constraint 2",self.kAC - 0.27)
#             return self.kAC - 0.35 
    
#     wing0 = wingClass.wing(RA[0], RT[0], 0, naca, writeGeom, folder, n_span, n_airfoil, N, 'linear', 0)
#     writeCSV.writeCSV(wing0)
#     outputFile0 = folder+"outputFile_"+wing0.filename+".txt"
#     savedRunFile0 = folder+"savedRunFile_"+wing0.filename+".fsm"
#     scriptFile0 = folder+"scriptFlightStream_"+wing0.filename+".txt"
#     scene0 = sceneClass.scene(alpha[0], V_inf, outputFile0, savedRunFile0, simulationTemplate)
#     if writeScripts == 'y':
#         writeFlightStreamScript.writeFlightStreamScript(wing0,scene0,scriptFile0)
#     if runFS == 'y':
#         os.chdir(folderFS)
#         p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile0)
#         os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code')
#     if readOutput == 'y':
#         CL0, CDi0, Cm0, numIter = readFlightStreamOutput.readFlightStreamOutput(outputFile0)
#     if calcParameters == 'y':
#         kappaD0, xAC0 = calculateParameters.calculateParameters(CL0,CDi0,Cm0,wing0,scene0)
    
#     cons = constClass(xAC0,xAC0)

#     # array([0.65556915, 0.62690503, 0.45271761])
#     # x0 = np.zeros(N-1)
#     x0 = np.array([0.02728809, 0.10936083, 0.24684843, 0.44081745])
    
#     const = ({'type': 'ineq',
#               'fun': cons.constraintFun1
#               })#,{'type': 'ineq',
#               #'fun': cons.constraintFun2
#               #})
                 
#     bnds = ((-2,2),(-2,2),(-2,2),(-2,2))
    
#     iterCount = iterCounter()
        
#     x = optimize.minimize(aeroEfficiency, x0, args=(cons), constraints = const, #bounds=bnds,
#                                                   tol=1e-3,
#                                                   method='SLSQP').x

#     # BASINHOPPING
#     # minimizer_kwargs={"method":"SLSQP","constraints":const,"bounds":bnds,"args":(cons),"tol":1e-3}
#     # x_from0 = optimize.basinhopping(aeroEfficiency, x0, minimizer_kwargs=minimizer_kwargs, niter=10, disp = True).x 
    
#     # print("xfrom0",x_from0)
    
#     # iterCount = iterCounter()
    
#     # x0 = np.tan(np.deg2rad(10))*np.linspace(0,wing0.b_semi,N)
#     # x0 = x0[1:]
#     # x_from10 = optimize.basinhopping(aeroEfficiency, x0, minimizer_kwargs=minimizer_kwargs, niter=10, disp = True).x

#     # print("xfrom10",x_from10)

#     # x0 = np.tan(np.deg2rad(20))*np.linspace(0,wing0.b_semi,N)
#     # x0 = x0[1:]
#     # x_from20 = optimize.basinhopping(aeroEfficiency, x0, minimizer_kwargs=minimizer_kwargs, niter=10, disp = True).x

# if curvetype == "optimizequad":
#     copyfile('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/commandFile.py', folder+'commandFile.py')
#     N = N_sections[0]
     
#     class constClass:
#         def __init__(self,xAC0,xAC):
#             self.xAC0 = xAC0
#             self.xAC = xAC
#             self.kAC = self.xAC-self.xAC0
    
#         def constraintFun1(self,x):
#             # print("constraint 1",-self.kAC + 0.3)
#             # return -self.kAC + 0.4 
#             b=0.4
#             a=0.35
#             return abs((b-a)/2) - abs(self.kAC - (a+b)/2)
    
#         def constraintFun2(self,x):
#             # print("constraint 2",self.kAC - 0.27)
#             return self.kAC - 0.35 
    
#     wing0 = wingClass.wing(RA[0], RT[0], 0, naca, writeGeom, folder, n_span, n_airfoil, N, 'linear', 0)
#     writeCSV.writeCSV(wing0)
#     outputFile0 = folder+"outputFile_"+wing0.filename+".txt"
#     savedRunFile0 = folder+"savedRunFile_"+wing0.filename+".fsm"
#     scriptFile0 = folder+"scriptFlightStream_"+wing0.filename+".txt"
#     scene0 = sceneClass.scene(alpha[0], V_inf, outputFile0, savedRunFile0, simulationTemplate)
#     if writeScripts == 'y':
#         writeFlightStreamScript.writeFlightStreamScript(wing0,scene0,scriptFile0)
#     if runFS == 'y':
#         os.chdir(folderFS)
#         p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile0)
#         os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code')
#     if readOutput == 'y':
#         CL0, CDi0, Cm0, numIter = readFlightStreamOutput.readFlightStreamOutput(outputFile0)
#     if calcParameters == 'y':
#         kappaD0, xAC0 = calculateParameters.calculateParameters(CL0,CDi0,Cm0,wing0,scene0)
    
#     cons = constClass(xAC0,xAC0)

#     # array([0.65556915, 0.62690503, 0.45271761])
#     # x0 = np.zeros(N-1)
#     x0 = np.array([0, 0.17, 0])
    
#     const = ({'type': 'ineq',
#               'fun': cons.constraintFun1
#               })#,{'type': 'ineq',
#               #'fun': cons.constraintFun2
#               #})
                 
#     bnds = ((-5,5),(-5,5))
    
#     iterCount = iterCounter()
        
#     # x = optimize.minimize(aeroEfficiency, x0, args=(cons), constraints = const, bounds=bnds,
#     #                                               tol=1e-3,
#     #                                               method='SLSQP').x
    
#     minimizer_kwargs={"method":"SLSQP","constraints":const,"args":(cons),"tol":1e-3} #,"bounds":bnds
#     x = optimize.basinhopping(aeroEfficiency, x0, minimizer_kwargs=minimizer_kwargs, niter=10, disp = True).x 
    
# elif curvetype == "defined":

#     print('joepie')

#     x = np.array([-0.4684, 0.4173, 0.6598, -0.0854])
    
#     if alpharange == 'y':
#         CL = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         CDi = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         Cm = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         numIter = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        
#         kappaD = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         xAC_c = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         kappaAC = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))

#         for i in range(len(RA)):
#             for j in range(len(RT)):
#                 for k in range(len(Lambda)):        
#                     for n in range(len(alpha)):
#                         print(RA[i], RT[j], Lambda[k], alpha[n])
#                         wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, n_airfoil, N_sections[0], curvetype, x)
#                         writeCSV.writeCSV(wing)
#                         outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_alpha"+str(alpha[n])+".txt"
#                         savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_alpha"+str(alpha[n])+".fsm"
#                         scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_alpha"+str(alpha[n])+".txt"
#                         scene = sceneClass.scene(alpha[n], V_inf, outputFile, savedRunFile, simulationTemplate)
#                         if writeScripts == 'y':
#                             writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
#                         if runFS == 'y':
#                             os.chdir(folderFS)
#                             p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
#                             os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
#                         if readOutput == 'y':
#                             CL[i,j,k,n], CDi[i,j,k,n], Cm[i,j,k,n], numIter[i,j,k,n] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
#                         if calcParameters == 'y':
#                             kappaD[i,j,k,n], xAC_c[i,j,k,n] = calculateParameters.calculateParameters(CL[i,j,k,n],CDi[i,j,k,n],Cm[i,j,k,n],wing,scene)
#                             kappaAC[i,j,k,n] = (xAC_c[i,j,k,n] - xAC_c[i,j,0,n])

#     elif alpharange == 'n':

#         wing0 = wingClass.wing(RA[0], RT[0], Lambda[0], naca, writeGeom, folder, n_span, n_airfoil, N_sections[0], "linear", 0)
#         writeCSV.writeCSV(wing0)
#         outputFile0 = folder+"outputFile_"+wing0.filename+".txt"
#         savedRunFile0 = folder+"savedRunFile_"+wing0.filename+".fsm"
#         scriptFile0 = folder+"scriptFlightStream_"+wing0.filename+".txt"
#         scene0 = sceneClass.scene(alpha[0], V_inf, outputFile0, savedRunFile0, simulationTemplate)
#         if writeScripts == 'y':
#             writeFlightStreamScript.writeFlightStreamScript(wing0,scene0,scriptFile0)
#         if runFS == 'y':
#             os.chdir(folderFS)
#             p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile0)
#             os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code')
#         if readOutput == 'y':
#             CL0, CDi0, Cm0, numIter = readFlightStreamOutput.readFlightStreamOutput(outputFile0)
#         if calcParameters == 'y':
#             kappaD0, xAC0 = calculateParameters.calculateParameters(CL0,CDi0,Cm0,wing0,scene0)
    
    
        
#         print(RA[0], RT[0])
#         wing = wingClass.wing(RA[0], RT[0], Lambda[0], naca, writeGeom, folder, n_span, n_airfoil, N_sections[0], curvetype, x)
#         writeCSV.writeCSV(wing)
#         outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_xDefined.txt"
#         savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_xDefined.fsm"
#         scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_xDefined.txt"
#         scene = sceneClass.scene(alpha[0], V_inf, outputFile, savedRunFile, simulationTemplate)
#         if writeScripts == 'y':
#             writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
#         if runFS == 'y':
#             os.chdir(folderFS)
#             p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
#             os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
#         if readOutput == 'y':
#             CL, CDi, Cm, numIter = readFlightStreamOutput.readFlightStreamOutput(outputFile)
#         if calcParameters == 'y':
#             kappaD, xAC_c = calculateParameters.calculateParameters(CL,CDi,Cm,wing,scene)
#             kappaAC = (xAC_c - xAC0)      
        
#         print(wing.x_quarterChord,"kappaAC: ",kappaAC,"kappaD: ",kappaD)

# else: 
#     if alpharange == 'n':
#         CL = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
#         CDi = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
#         Cm = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
#         numIter = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
        
#         kappaD = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
#         xAC_c = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
#         kappaAC = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
        
#         for i in range(len(RA)):
#             for j in range(len(RT)):
#                 for k in range(len(Lambda)):
#                     for n in range(len(N_sections)):
#                         print(RA[i], RT[j], Lambda[k])
#                         wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, n_airfoil, N_sections[n], curvetype, 0)
#                         writeCSV.writeCSV(wing)
#                         outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
#                         savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
#                         scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
#                         scene = sceneClass.scene(alpha[0], V_inf, outputFile, savedRunFile, simulationTemplate)
#                         if writeScripts == 'y':
#                             writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
#                         if runFS == 'y':
#                             os.chdir(folderFS)
#                             p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
#                             os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
#                         if readOutput == 'y':
#                             CL[i,j,k,n], CDi[i,j,k,n], Cm[i,j,k,n], numIter[i,j,k,n] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
#                         if calcParameters == 'y':
#                             kappaD[i,j,k,n], xAC_c[i,j,k,n] = calculateParameters.calculateParameters(CL[i,j,k,n],CDi[i,j,k,n],Cm[i,j,k,n],wing,scene)
#                             kappaAC[i,j,k,n] = (xAC_c[i,j,k,n] - xAC_c[i,j,0,n])
        
#         print(wing.x_quarterChord[0],"kappaAC: ",kappaAC,"kappaD: ",kappaD)
#     elif alpharange == 'y':
#         CL = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         CDi = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         Cm = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         numIter = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        
#         kappaD = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         xAC_c = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
#         kappaAC = np.zeros((len(RA),len(RT),len(Lambda),len(alpha)))
        
#         for i in range(len(RA)):
#             for j in range(len(RT)):
#                 for k in range(len(Lambda)):
#                     for n in range(len(alpha)):
#                         print(RA[i], RT[j], Lambda[k], alpha[n])
#                         wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, n_airfoil, N_sections[0], curvetype, 0)
#                         writeCSV.writeCSV(wing)
#                         outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
#                         savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
#                         scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
#                         scene = sceneClass.scene(alpha[n], V_inf, outputFile, savedRunFile, simulationTemplate)
#                         if writeScripts == 'y':
#                             writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
#                         if runFS == 'y':
#                             os.chdir(folderFS)
#                             p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
#                             os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
#                         if readOutput == 'y':
#                             CL[i,j,k,n], CDi[i,j,k,n], Cm[i,j,k,n], numIter[i,j,k,n] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
#                         if calcParameters == 'y':
#                             kappaD[i,j,k,n], xAC_c[i,j,k,n] = calculateParameters.calculateParameters(CL[i,j,k,n],CDi[i,j,k,n],Cm[i,j,k,n],wing,scene)
#                             kappaAC[i,j,k,n] = (xAC_c[i,j,k,n] - xAC_c[i,j,0,n])
        
#         print(wing.x_quarterChord[0],alpha[n],"kappaAC: ",kappaAC,"kappaD: ",kappaD)            

# # plt.plot(wing.x,wing.y)

# # print("CL: ",CL,"CDi: ",CDi,"Cm: ",Cm)
        
# np.save(folder+"CL.npy",CL)
# np.save(folder+"CDi.npy",CDi)
# np.save(folder+"Cm.npy",Cm)
# np.save(folder+"numIter.npy",numIter)
# np.save(folder+"kappaD.npy",kappaD)
# np.save(folder+"kappaAC.npy",kappaAC)
# np.save(folder+"xAC_c.npy",xAC_c)












tend = time.time() - tstart
print(tend)


