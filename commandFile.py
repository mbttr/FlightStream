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
import numpy as np
import subprocess
import time
import datetime
import matplotlib.pyplot as plt
import scipy.optimize as optimize

def aeroEfficiency(x):
    # print(x)
    wing = wingClass.wing(8, 0.25, Lambda, naca, writeGeom, folder, n_span, n_airfoil, N_sections, curvetype, x)
    outputFile = folder+"outputFile_"+wing.filename+".txt"
    savedRunFile = folder+"savedRunFile_RA"+wing.filename+".fsm"
    scriptFile = folder+"scriptFlightStream_RA"+wing.filename+".txt"
    scene = sceneClass.scene(alpha, V_inf, outputFile, savedRunFile, simulationTemplate)
    if writeScripts == 'y':
        writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
    if runFS == 'y':
        os.chdir(folderFS)
        p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
        os.chdir('C:/Users/bmoor/Desktop/FlightStream/Code/')
    if readOutput == 'y':
        CL, CDi, Cm = readFlightStreamOutput.readFlightStreamOutput(outputFile)
    if calcParameters == 'y':
        kappaD, xAC = calculateParameters.calculateParameters(CL,CDi,Cm,wing,scene)
    
    print(CL,CDi,CDi/CL)
    return CDi/CL

tstart = time.time()
now = datetime.datetime.now()

RA = np.linspace(20,20,1)
RT = np.linspace(1,1,1)
Lambda = np.linspace(35,40,1)
naca = '0012'
alpha = 5
V_inf = 10

n_span = 80
n_airfoil = 80

N_sections = np.array([32])
curvetype = 'linear'

writeGeom = 'y'
writeScripts = 'n'
runFS = 'n'
readOutput = 'n'
calcParameters = 'n'

folder = "C:/Users/bmoor/Desktop/FlightStream/Code/"+str(now.month)+"-"+str(now.day)+"/Run_Time_"+str(now.hour)+"-"+str(now.minute)+"-"+str(now.second)+"/"
folderFS = "C:/Users/bmoor/Desktop/FlightStream/FlightStream_installation/"
os.makedirs(folder)

simulationTemplate = "C:/Users/bmoor/Desktop/FlightStream/Code/simulationTemplate.fsm"


if curvetype != "optimize":
    CL = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
    CDi = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
    Cm = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
    
    # CL = np.load('CL.npy')
    # CDi = np.load('CDi.npy')
    # Cm = np.load('Cm.npy')
    
    kappaD = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
    xAC = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
    kappaAC = np.zeros((len(RA),len(RT),len(Lambda),len(N_sections)))
    
    for i in range(len(RA)):
        for j in range(len(RT)):
            for k in range(len(Lambda)):
                for n in range(len(N_sections)):
                    print(RA[i], RT[j], Lambda[k], N_sections[n])
                    wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, n_airfoil, N_sections[n], curvetype, 0)
                    outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
                    savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
                    scriptFile = folder+"scriptFlightStream_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
                    scene = sceneClass.scene(alpha, V_inf, outputFile, savedRunFile, simulationTemplate)
                    if writeScripts == 'y':
                        writeFlightStreamScript.writeFlightStreamScript(wing,scene,scriptFile)
                    if runFS == 'y':
                        os.chdir(folderFS)
                        p = subprocess.run("FlightStream.exe -hidden -script "+scriptFile)
                        os.chdir('C:/Users/bmoor/Desktop/FlightStream/Code/')
                    if readOutput == 'y':
                        CL[i,j,k,n], CDi[i,j,k,n], Cm[i,j,k,n] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
                    if calcParameters == 'y':
                        kappaD[i,j,k,n], xAC[i,j,k,n] = calculateParameters.calculateParameters(CL[i,j,k,n],CDi[i,j,k,n],Cm[i,j,k,n],wing,scene)
                        kappaAC[i,j,k,n] = (xAC[i,j,k,n] - xAC[i,j,0,n])/xAC[i,j,0,n]   
    
    # plt.plot(wing.x,wing.y)
    
    print(CL,CDi,Cm)
    
    np.save(folder+"CL.npy",CL)
    np.save(folder+"CDi.npy",CDi)
    np.save(folder+"Cm.npy",Cm)
    np.save(folder+"kappaD.npy",kappaD)
    np.save(folder+"kappaAC.npy",kappaAC)
    np.save(folder+"xAC.npy",xAC)


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
# else:     
#     # array([0.65556915, 0.62690503, 0.45271761])
#     x0 = np.zeros(N_sections-1)
        
#     x = optimize.minimize(aeroEfficiency, x0,
#                                                   tol=1e-4,
#                                                   method='BFGS').x
    
# tend = time.time() - tstart
# print(tend)


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


# GRID CONVERGENCE




# markerScale = 0.75

# fig1 = plt.figure(figsize=(4.25,4.25))
# plt.subplot(111)


# for i in range(len(RA)):
#     for j in range(len(RT)):
#         for k in range(len(Lambda)):
#             print(RA[i], RT[j], Lambda[k])
#             wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, 20)
#             outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
#             savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
#             scene = sceneClass.scene(alpha, V_inf, outputFile, savedRunFile, simulationTemplate)
#             if writeScripts == 'y':
#                 writeFlightStreamScript.writeFlightStreamScript(wing,scene)
#             if runFS == 'y':
#                 os.chdir('C:/Users/bmoor/Desktop/FlightStream/FlightStream_installation/')
#                 p = subprocess.run("FlightStream.exe -hidden -script "+scene.scriptFilename)
#                 os.chdir('C:/Users/bmoor/Desktop/FlightStream/Code/')
#             if readOutput == 'y':
#                 CL[i,j,k], CDi[i,j,k], Cm[i,j,k] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
#             if calcParameters == 'y':
#                 kappaD[i,j,k], xAC[i,j,k] = calculateParameters.calculateParameters(CL[i,j,k],CDi[i,j,k],Cm[i,j,k],wing,scene)
#                 kappaAC[i,j,k] = (xAC[i,j,k] - xAC[i,j,0])/xAC[i,j,0]   


# plt.scatter(Lambda,CDi[i,j,:],edgecolor='k',linewidths=1,facecolor='none',marker='o',s=300*markerScale,label='20 panels chordwise')

# for i in range(len(RA)):
#     for j in range(len(RT)):
#         for k in range(len(Lambda)):
#             print(RA[i], RT[j], Lambda[k])
#             wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, 40)
#             outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
#             savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
#             scene = sceneClass.scene(alpha, V_inf, outputFile, savedRunFile, simulationTemplate)
#             if writeScripts == 'y':
#                 writeFlightStreamScript.writeFlightStreamScript(wing,scene)
#             if runFS == 'y':
#                 os.chdir('C:/Users/bmoor/Desktop/FlightStream/FlightStream_installation/')
#                 p = subprocess.run("FlightStream.exe -hidden -script "+scene.scriptFilename)
#                 os.chdir('C:/Users/bmoor/Desktop/FlightStream/Code/')
#             if readOutput == 'y':
#                 CL[i,j,k], CDi[i,j,k], Cm[i,j,k] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
#             if calcParameters == 'y':
#                 kappaD[i,j,k], xAC[i,j,k] = calculateParameters.calculateParameters(CL[i,j,k],CDi[i,j,k],Cm[i,j,k],wing,scene)
#                 kappaAC[i,j,k] = (xAC[i,j,k] - xAC[i,j,0])/xAC[i,j,0]   


# plt.scatter(Lambda,CDi[i,j,:],edgecolor='k',linewidths=1,facecolor='none',marker='o',s=175*markerScale,label='40 panels chordwise')

# for i in range(len(RA)):
#     for j in range(len(RT)):
#         for k in range(len(Lambda)):
#             print(RA[i], RT[j], Lambda[k])
#             wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, 80)
#             outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
#             savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
#             scene = sceneClass.scene(alpha, V_inf, outputFile, savedRunFile, simulationTemplate)
#             if writeScripts == 'y':
#                 writeFlightStreamScript.writeFlightStreamScript(wing,scene)
#             if runFS == 'y':
#                 os.chdir('C:/Users/bmoor/Desktop/FlightStream/FlightStream_installation/')
#                 p = subprocess.run("FlightStream.exe -hidden -script "+scene.scriptFilename)
#                 os.chdir('C:/Users/bmoor/Desktop/FlightStream/Code/')
#             if readOutput == 'y':
#                 CL[i,j,k], CDi[i,j,k], Cm[i,j,k] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
#             if calcParameters == 'y':
#                 kappaD[i,j,k], xAC[i,j,k] = calculateParameters.calculateParameters(CL[i,j,k],CDi[i,j,k],Cm[i,j,k],wing,scene)
#                 kappaAC[i,j,k] = (xAC[i,j,k] - xAC[i,j,0])/xAC[i,j,0]   


# plt.scatter(Lambda,CDi[i,j,:],edgecolor='k',linewidths=1,facecolor='none',marker='o',s=82.5*markerScale,label='80 panels chordwise')


# for i in range(len(RA)):
#     for j in range(len(RT)):
#         for k in range(len(Lambda)):
#             print(RA[i], RT[j], Lambda[k])
#             wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, 120)
#             outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
#             savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
#             scene = sceneClass.scene(alpha, V_inf, outputFile, savedRunFile, simulationTemplate)
#             if writeScripts == 'y':
#                 writeFlightStreamScript.writeFlightStreamScript(wing,scene)
#             if runFS == 'y':
#                 os.chdir('C:/Users/bmoor/Desktop/FlightStream/FlightStream_installation/')
#                 p = subprocess.run("FlightStream.exe -hidden -script "+scene.scriptFilename)
#                 os.chdir('C:/Users/bmoor/Desktop/FlightStream/Code/')
#             if readOutput == 'y':
#                 CL[i,j,k], CDi[i,j,k], Cm[i,j,k] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
#             if calcParameters == 'y':
#                 kappaD[i,j,k], xAC[i,j,k] = calculateParameters.calculateParameters(CL[i,j,k],CDi[i,j,k],Cm[i,j,k],wing,scene)
#                 kappaAC[i,j,k] = (xAC[i,j,k] - xAC[i,j,0])/xAC[i,j,0]   


# plt.scatter(Lambda,CDi[i,j,:],edgecolor='k',linewidths=1,facecolor='none',marker='o',s=25*markerScale,label='120 panels chordwise')


# for i in range(len(RA)):
#     for j in range(len(RT)):
#         for k in range(len(Lambda)):
#             print(RA[i], RT[j], Lambda[k])
#             wing = wingClass.wing(RA[i], RT[j], Lambda[k], naca, writeGeom, folder, n_span, 160)
#             outputFile = folder+"outputFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".txt"
#             savedRunFile = folder+"savedRunFile_RA"+str(wing.RA)+"_RT"+str(wing.RT)+"_Lambda"+str(wing.Lambda)+".fsm"
#             scene = sceneClass.scene(alpha, V_inf, outputFile, savedRunFile, simulationTemplate)
#             if writeScripts == 'y':
#                 writeFlightStreamScript.writeFlightStreamScript(wing,scene)
#             if runFS == 'y':
#                 os.chdir('C:/Users/bmoor/Desktop/FlightStream/FlightStream_installation/')
#                 p = subprocess.run("FlightStream.exe -hidden -script "+scene.scriptFilename)
#                 os.chdir('C:/Users/bmoor/Desktop/FlightStream/Code/')
#             if readOutput == 'y':
#                 CL[i,j,k], CDi[i,j,k], Cm[i,j,k] = readFlightStreamOutput.readFlightStreamOutput(outputFile)
#             if calcParameters == 'y':
#                 kappaD[i,j,k], xAC[i,j,k] = calculateParameters.calculateParameters(CL[i,j,k],CDi[i,j,k],Cm[i,j,k],wing,scene)
#                 kappaAC[i,j,k] = (xAC[i,j,k] - xAC[i,j,0])/xAC[i,j,0]   
        
                

# plt.scatter(Lambda,CDi[i,j,:],edgecolor='k',linewidths=1,facecolor='none',marker='o',s=5*markerScale,label='160 panels chordwise')


# plt.xlim([0,40])
# plt.xlabel(r'$\Lambda_{c/4}$',fontsize=12)
# plt.ylabel(r'$C_{D_i}$',fontsize=12)
# plt.legend(loc=3,frameon=False)
# fig1.savefig("CDi_spanwise_gridconvergence.pdf", bbox_inches='tight')
# plt.show()