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

class iterCounter:
    def __init__(self):
        self.count = 1

def aeroEfficiency(x, cons):
    # print(x)
    wing = wingClass.wing(12, 0.25, Lambda, naca, writeGeom, folder, n_span, n_airfoil, N, curvetype, x)
    wing.CSVFile = folder+wing.filename+"_iter"+str(iterCount.count)+".csv"
    writeCSV.writeCSV(wing)
    outputFile = folder+"outputFile_"+wing.filename+"_iter"+str(iterCount.count)+".txt"
    savedRunFile = folder+"savedRunFile_"+wing.filename+".fsm"
    scriptFile = folder+"scriptFlightStream_"+wing.filename+"_iter"+str(iterCount.count)+".txt"
    scene = sceneClass.scene(alpha, V_inf, outputFile, savedRunFile, simulationTemplate)
    if writeScripts == 'y':
        writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
    if runFS == 'y':
        os.chdir(folderFS)
        p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
        os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code')
    if readOutput == 'y':
        CL, CDi, Cm = readFlightStreamOutput.readFlightStreamOutput(outputFile)
    if calcParameters == 'y':
        kappaD, xAC = calculateParameters.calculateParameters(CL,CDi,Cm,wing,scene)
    
    cons.xAC = xAC
    cons.kAC = cons.xAC - cons.xAC0
    
    print(iterCount.count,x,cons.kAC,kappaD)
    iterCount.count += 1
    
    return kappaD

tstart = time.time()
now = datetime.datetime.now()

RA = np.linspace(8,20,1)
RT = np.linspace(0.25,1,1)
Lambda = np.linspace(40,40,1)
naca = '0012'
alpha = -5
V_inf = 10

n_span = 80
n_airfoil = 80

N_sections = np.array([32])
curvetype = 'linear'

writeGeom = 'y'
writeScripts = 'y'
runFS = 'y'
readOutput = 'y'
calcParameters = 'y'

folder = "C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/"+str(now.month)+"-"+str(now.day)+"/Run_Time_"+str(now.hour)+"-"+str(now.minute)+"-"+str(now.second)+"/"
folderFS = "C:/Users/Bruno-USU/Desktop/FlightStream/"
os.makedirs(folder)

simulationTemplate = "C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/simulationTemplate.fsm"


if curvetype != "optimize":
    
    if curvetype == "defined":
        wing0 = wingClass.wing(RA[0], RT[0], Lambda[0], naca, writeGeom, folder, n_span, n_airfoil, N_sections[0], "linear", 0)
        writeCSV.writeCSV(wing0)
        outputFile0 = folder+"outputFile_"+wing0.filename+".txt"
        savedRunFile0 = folder+"savedRunFile_"+wing0.filename+".fsm"
        scriptFile0 = folder+"scriptFlightStream_"+wing0.filename+".txt"
        scene0 = sceneClass.scene(alpha, V_inf, outputFile0, savedRunFile0, simulationTemplate)
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


        x = np.array([-3.94710088,  0.81221836,  4.6402091,  -4.89999999])
        print(RA[0], RT[0])
        wing = wingClass.wing(RA[0], RT[0], Lambda[0], naca, writeGeom, folder, n_span, n_airfoil, N_sections[0], curvetype, x)
        writeCSV.writeCSV(wing)
        outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_xDefined.txt"
        savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_xDefined.fsm"
        scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_xDefined.txt"
        scene = sceneClass.scene(alpha, V_inf, outputFile, savedRunFile, simulationTemplate)
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
        CL = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
        CDi = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
        Cm = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
        numIter = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
        
        # CL = np.load('CL.npy')
        # CDi = np.load('CDi.npy')
        # Cm = np.load('Cm.npy')
        
        kappaD = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
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
                        scene = sceneClass.scene(alpha, V_inf, outputFile, savedRunFile, simulationTemplate)
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
        
        print(wing.x_quarterChord[0],"kappaAC: ",kappaAC,"kappaD: ",kappaD)
    
    # plt.plot(wing.x,wing.y)
    
    print("CL: ",CL,"CDi: ",CDi,"Cm: ",Cm)
    
    np.save(folder+"CL_20.npy",CL)
    np.save(folder+"CDi_20.npy",CDi)
    np.save(folder+"Cm_20.npy",Cm)
    np.save(folder+"numIter_20.npy",numIter)
    np.save(folder+"kappaD.npy",kappaD)
    np.save(folder+"kappaAC.npy",kappaAC)
    np.save(folder+"xAC_c.npy",xAC_c)

    # n_span = 40

    # for i in range(len(RA)):
    #     for j in range(len(RT)):
    #         for k in range(len(Lambda)):
    #             for n in range(len(N_sections)):
    #                 print(RA[i], RT[j], Lambda[k], N_sections[n])
    #                 wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, n_airfoil, N_sections[n], curvetype, 0)
    #                 writeCSV.writeCSV(wing)
    #                 outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
    #                 savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
    #                 scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
    #                 scene = sceneClass.scene(alpha, V_inf, outputFile, savedRunFile, simulationTemplate)
    #                 if writeScripts == 'y':
    #                     writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
    #                 if runFS == 'y':
    #                     os.chdir(folderFS)
    #                     p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
    #                     os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
    #                 if readOutput == 'y':
    #                     CL[i,j,k,n], CDi[i,j,k,n], Cm[i,j,k,n], numIter[i,j,k,n] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
    #                 if calcParameters == 'y':
    #                     kappaD[i,j,k,n], xAC_c[i,j,k,n] = calculateParameters.calculateParameters(CL[i,j,k,n],CDi[i,j,k,n],Cm[i,j,k,n],wing,scene)
    #                     kappaAC[i,j,k,n] = (xAC_c[i,j,k,n] - xAC_c[i,j,0,n])
    
    # # plt.plot(wing.x,wing.y)
    
    # print(CL,CDi,Cm)
    
    # np.save(folder+"CL_40.npy",CL)
    # np.save(folder+"CDi_40.npy",CDi)
    # np.save(folder+"Cm_40.npy",Cm)
    # np.save(folder+"numIter_40.npy",numIter)
    # np.save(folder+"kappaD.npy",kappaD)
    # np.save(folder+"kappaAC.npy",kappaAC)
    # np.save(folder+"xAC_c.npy",xAC_c)

    # n_span = 80

    # for i in range(len(RA)):
    #     for j in range(len(RT)):
    #         for k in range(len(Lambda)):
    #             for n in range(len(N_sections)):
    #                 print(RA[i], RT[j], Lambda[k], N_sections[n])
    #                 wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, n_airfoil, N_sections[n], curvetype, 0)
    #                 writeCSV.writeCSV(wing)
    #                 outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
    #                 savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
    #                 scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
    #                 scene = sceneClass.scene(alpha, V_inf, outputFile, savedRunFile, simulationTemplate)
    #                 if writeScripts == 'y':
    #                     writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
    #                 if runFS == 'y':
    #                     os.chdir(folderFS)
    #                     p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
    #                     os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
    #                 if readOutput == 'y':
    #                     CL[i,j,k,n], CDi[i,j,k,n], Cm[i,j,k,n], numIter[i,j,k,n] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
    #                 if calcParameters == 'y':
    #                     kappaD[i,j,k,n], xAC_c[i,j,k,n] = calculateParameters.calculateParameters(CL[i,j,k,n],CDi[i,j,k,n],Cm[i,j,k,n],wing,scene)
    #                     kappaAC[i,j,k,n] = (xAC_c[i,j,k,n] - xAC_c[i,j,0,n])
    
    # # plt.plot(wing.x,wing.y)
    
    # print(CL,CDi,Cm)
    
    # np.save(folder+"CL_80.npy",CL)
    # np.save(folder+"CDi_80.npy",CDi)
    # np.save(folder+"Cm_80.npy",Cm)
    # np.save(folder+"numIter_80.npy",numIter)
    # np.save(folder+"kappaD.npy",kappaD)
    # np.save(folder+"kappaAC.npy",kappaAC)
    # np.save(folder+"xAC_c.npy",xAC_c)


    # n_span = 120

    # for i in range(len(RA)):
    #     for j in range(len(RT)):
    #         for k in range(len(Lambda)):
    #             for n in range(len(N_sections)):
    #                 print(RA[i], RT[j], Lambda[k], N_sections[n])
    #                 wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, n_airfoil, N_sections[n], curvetype, 0)
    #                 writeCSV.writeCSV(wing)
    #                 outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
    #                 savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
    #                 scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
    #                 scene = sceneClass.scene(alpha, V_inf, outputFile, savedRunFile, simulationTemplate)
    #                 if writeScripts == 'y':
    #                     writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
    #                 if runFS == 'y':
    #                     os.chdir(folderFS)
    #                     p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
    #                     os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
    #                 if readOutput == 'y':
    #                     CL[i,j,k,n], CDi[i,j,k,n], Cm[i,j,k,n], numIter[i,j,k,n] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
    #                 if calcParameters == 'y':
    #                     kappaD[i,j,k,n], xAC_c[i,j,k,n] = calculateParameters.calculateParameters(CL[i,j,k,n],CDi[i,j,k,n],Cm[i,j,k,n],wing,scene)
    #                     kappaAC[i,j,k,n] = (xAC_c[i,j,k,n] - xAC_c[i,j,0,n])
    
    # # plt.plot(wing.x,wing.y)
    
    # print(CL,CDi,Cm)
    
    # np.save(folder+"CL_120.npy",CL)
    # np.save(folder+"CDi_120.npy",CDi)
    # np.save(folder+"Cm_120.npy",Cm)
    # np.save(folder+"numIter_120.npy",numIter)
    # np.save(folder+"kappaD.npy",kappaD)
    # np.save(folder+"kappaAC.npy",kappaAC)
    # np.save(folder+"xAC_c.npy",xAC_c)


    # n_span = 160

    # for i in range(len(RA)):
    #     for j in range(len(RT)):
    #         for k in range(len(Lambda)):
    #             for n in range(len(N_sections)):
    #                 print(RA[i], RT[j], Lambda[k], N_sections[n])
    #                 wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, n_airfoil, N_sections[n], curvetype, 0)
    #                 writeCSV.writeCSV(wing)
    #                 outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
    #                 savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
    #                 scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
    #                 scene = sceneClass.scene(alpha, V_inf, outputFile, savedRunFile, simulationTemplate)
    #                 if writeScripts == 'y':
    #                     writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
    #                 if runFS == 'y':
    #                     os.chdir(folderFS)
    #                     p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
    #                     os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
    #                 if readOutput == 'y':
    #                     CL[i,j,k,n], CDi[i,j,k,n], Cm[i,j,k,n], numIter[i,j,k,n] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
    #                 if calcParameters == 'y':
    #                     kappaD[i,j,k,n], xAC_c[i,j,k,n] = calculateParameters.calculateParameters(CL[i,j,k,n],CDi[i,j,k,n],Cm[i,j,k,n],wing,scene)
    #                     kappaAC[i,j,k,n] = (xAC_c[i,j,k,n] - xAC_c[i,j,0,n])
    
    # # plt.plot(wing.x,wing.y)
    
    # print(CL,CDi,Cm)
    
    # np.save(folder+"CL_160.npy",CL)
    # np.save(folder+"CDi_160.npy",CDi)
    # np.save(folder+"Cm_160.npy",Cm)
    # np.save(folder+"numIter_160.npy",numIter)
    # np.save(folder+"kappaD.npy",kappaD)
    # np.save(folder+"kappaAC.npy",kappaAC)
    # np.save(folder+"xAC_c.npy",xAC_c)

    # n_span = 200

    # for i in range(len(RA)):
    #     for j in range(len(RT)):
    #         for k in range(len(Lambda)):
    #             for n in range(len(N_sections)):
    #                 print(RA[i], RT[j], Lambda[k], N_sections[n])
    #                 wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, n_airfoil, N_sections[n], curvetype, 0)
    #                 writeCSV.writeCSV(wing)
    #                 outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
    #                 savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
    #                 scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
    #                 scene = sceneClass.scene(alpha, V_inf, outputFile, savedRunFile, simulationTemplate)
    #                 if writeScripts == 'y':
    #                     writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
    #                 if runFS == 'y':
    #                     os.chdir(folderFS)
    #                     p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
    #                     os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code/')
    #                 if readOutput == 'y':
    #                     CL[i,j,k,n], CDi[i,j,k,n], Cm[i,j,k,n], numIter[i,j,k,n] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
    #                 if calcParameters == 'y':
    #                     kappaD[i,j,k,n], xAC_c[i,j,k,n] = calculateParameters.calculateParameters(CL[i,j,k,n],CDi[i,j,k,n],Cm[i,j,k,n],wing,scene)
    #                     kappaAC[i,j,k,n] = (xAC_c[i,j,k,n] - xAC_c[i,j,0,n])
    
    # # plt.plot(wing.x,wing.y)
    
    # print(CL,CDi,Cm)
    
    # np.save(folder+"CL_200.npy",CL)
    # np.save(folder+"CDi_200.npy",CDi)
    # np.save(folder+"Cm_200.npy",Cm)
    # np.save(folder+"numIter_200.npy",numIter)
    # np.save(folder+"kappaD.npy",kappaD)
    # np.save(folder+"kappaAC.npy",kappaAC)
    # np.save(folder+"xAC_c.npy",xAC_c)


# FIGURES
# for i in range(len(RA)):
#     for j in range(len(RT)):
#         for k in range(len(Lambda)):
#             plt.plot(N_sections,CL[i,j,k,:])
# plt.xscale("log")
# plt.show()

# for i in range(len(RA)):
#     for j in range(len(RT)):
#         for k in range(len(Lambda)):            
#             plt.plot(N_sections,CDi[i,j,k,:])
# plt.xscale("log")
# plt.show()

# for i in range(len(RA)):
#     for j in range(len(RT)):
#         for k in range(len(Lambda)):            
#             plt.plot(N_sections,Cm[i,j,k,:])
# plt.xscale("log")
# plt.show()

# # OPTIMIZE LAMBDA

# def aeroEfficiency(Lambda):
#     print(Lambda)
#     wing = wingClass.wing(8, 0.25, Lambda, naca, writeGeom, folder, n_span, n_airfoil, N_sections, curvetype, 0)
#     outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
#     savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
#     scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
#     scene = sceneClass.scene(alpha, V_inf, outputFile, savedRunFile, simulationTemplate)
#     if writeScripts == 'y':
#         writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
#     if runFS == 'y':
#         os.chdir(folderFS)
#         p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
#         os.chdir('C:/Users/bmoor/Desktop/FlightStream/Code/')
#     if readOutput == 'y':
#         CL, CDi, Cm = readFlightStreamOutput.readFlightStreamOutput(outputFile)
#     if calcParameters == 'y':
#         kappaD, xAC = calculateParameters.calculateParameters(CL,CDi,Cm,wing,scene)
    
#     print(CL/CDi)
    
#     return CL/CDi
    
# Lambda0 = 20
    
# Lambda = optimize.minimize(aeroEfficiency, Lambda0,
#                                               tol=1e-4,
#                                               # options={'eps': 1e-3},
#                                               method='BFGS').x


# OPTIMIZE X
else:
    N = N_sections[0]
     
    class constClass:
        def __init__(self,xAC0,xAC):
            self.xAC0 = xAC0
            self.xAC = xAC
            self.kAC = self.xAC-self.xAC0
    
        def constraintFun1(self,x):
            return -cons.kAC + 0.3 
    
        def constraintFun2(self,x):
            return cons.kAC - 0.27 
    
    wing0 = wingClass.wing(12, 0.25, 0, naca, writeGeom, folder, n_span, n_airfoil, N, 'linear', 0)
    writeCSV.writeCSV(wing0)
    outputFile0 = folder+"outputFile_"+wing0.filename+".txt"
    savedRunFile0 = folder+"savedRunFile_"+wing0.filename+".fsm"
    scriptFile0 = folder+"scriptFlightStream_"+wing0.filename+".txt"
    scene0 = sceneClass.scene(alpha, V_inf, outputFile0, savedRunFile0, simulationTemplate)
    if writeScripts == 'y':
        writeFlightStreamScript.writeFlightStreamScript(wing0,scene0,scriptFile0)
    if runFS == 'y':
        os.chdir(folderFS)
        p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile0)
        os.chdir('C:/Users/Bruno-USU/Desktop/Research/FlightStream_Code')
    if readOutput == 'y':
        CL0, CDi0, Cm0 = readFlightStreamOutput.readFlightStreamOutput(outputFile0)
    if calcParameters == 'y':
        kappaD0, xAC0 = calculateParameters.calculateParameters(CL0,CDi0,Cm0,wing0,scene0)
    
    cons = constClass(xAC0,xAC0)

    # array([0.65556915, 0.62690503, 0.45271761])
    x0 = np.zeros(N-1)
    
    const = ({'type': 'ineq',
              'fun': cons.constraintFun2
              },{'type': 'ineq',
              'fun': cons.constraintFun2
             })
    bnds = ((-5,5),(-5,5),(-5,5),(-5,5))
    
    iterCount = iterCounter()
        
    # x = optimize.minimize(aeroEfficiency, x0, args=(cons), constraints = const, bounds=bnds,
    #                                               tol=1e-4,
    #                                               method='SLSQP').x

    # BASINHOPPING
    minimizer_kwargs={"method":"SLSQP","constraints":const,"bounds":bnds,"args":(cons),"tol":1e-3}
    x = optimize.basinhopping(aeroEfficiency, x0, minimizer_kwargs=minimizer_kwargs, niter=2, disp = True).x
    
tend = time.time() - tstart
print(tend)


# fig1 = plt.figure(figsize=(4.25,4.25))
# plt.subplot(111)
# for i in range(len(RA)):
#     plt.plot(kappaAC[i,0,:],kappaD[i,0,:],'k')
# plt.xlabel('kappa_AC')
# plt.ylabel('kappa_D')
# plt.show()

# fig1 = plt.figure(figsize=(4.25,4.25))
# plt.subplot(111)
# for i in range(len(RA)):
#     plt.plot(Lambda,kappaD[i,0,:],'k')
# plt.xlabel('sweep')
# plt.ylabel('kappa_D')
# plt.show()