# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:01:45 2021

@author: bmoor
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import re
from mpl_toolkits.mplot3d import Axes3D
import os
import subprocess
import json
import shutil
import fileinput
import csv
import xlsxwriter
import scipy.optimize as optimize
import time
# import machupX as MX
from matplotlib import rc
import pandas as pd

RA = np.linspace(4,20,9)
RT = np.linspace(0.25,1,4)
Lambda_c = np.linspace(0,40,41)
Lambda = np.linspace(0,40,41)

kD_constant = np.load('kappaD_constant.npy')
kD_linear = np.load('kappaD_linear.npy')

kAC_constant = np.load('kappaAC_constant.npy')
kAC_linear = np.load('kappaAC_linear.npy')

xAC_constant = np.load('xAC_constant.npy')
xAC_linear = np.load('xAC_linear.npy')

CL_constant = np.load('CL_constant.npy')
CL_linear = np.load('CL_linear.npy')
kL_constant = CL_constant[:,:,:,0]/CL_constant[:,:,0]
kL_linear = CL_linear[:,:,:,0]/CL_linear[:,:,0]

# PLOTS KAPPA D CONSTANT

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

for j in range(len(RA)): # RT 0.25
    ax1.set_xticks(np.array([0,10,20,30,40]))
    ax1.set_yticks(np.arange(-0.15,0.2,0.05))
    ax1.plot(Lambda_c,kD_constant[j,0,:,0],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))    
for j in range(len(RA)): # RT 0.5
    ax2.set_xticks(np.array([0,10,20,30,40]))
    ax2.set_yticks(np.arange(-0.15,0.2,0.05))
    ax2.plot(Lambda_c,kD_constant[j,1,:,0],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  
for j in range(len(RA)): # RT 0.75
    ax3.set_xticks(np.array([0,10,20,30,40]))
    ax3.set_yticks(np.arange(-0.15,0.3,0.05))
    ax3.plot(Lambda_c,kD_constant[j,2,:,0],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  
for j in range(len(RA)): # RT 1.0
    ax4.set_xticks(np.array([0,10,20,30,40]))
    ax4.set_yticks(np.arange(-0.15,0.3,0.05))
    ax4.plot(Lambda_c,kD_constant[j,3,:,0],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  

ax1.set_ylim([-0.1,0.12])
ax2.set_ylim([-0.1,0.12])
ax3.set_ylim([-0.1,0.2])
ax4.set_ylim([-0.1,0.2])
ax1.set_xlim([0,40])
ax2.set_xlim([0,40])
ax3.set_xlim([0,40])
ax4.set_xlim([0,40])

ax1.text(1,-0.09,r'$R_T = 0.25$',fontsize=12)
ax2.text(1,-0.09,r'$R_T = 0.5$',fontsize=12)
ax3.text(1,-0.09,r'$R_T = 0.75$',fontsize=12)
ax4.text(1,-0.09,r'$R_T = 1.0$',fontsize=12)

ax1.set_ylabel(r'$\kappa_D$',fontsize=12)
ax3.set_ylabel(r'$\kappa_D$',fontsize=12)
ax3.set_xlabel(r'$\Lambda_ {\,c/4}$',fontsize=12)
ax4.set_xlabel(r'$\Lambda_ {\,c/4}$',fontsize=12)

plt.show()
fig.savefig('kappa_D_constant.pdf')


# PLOTS KAPPA D LINEAR

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

for j in range(len(RA)): # RT 0.25
    ax1.set_xticks(np.array([0,10,20,30,40]))
    ax1.set_yticks(np.arange(-0.2,0.6,0.1))
    ax1.plot(Lambda,kD_linear[j,0,:,0],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))    
for j in range(len(RA)): # RT 0.5
    ax2.set_xticks(np.array([0,10,20,30,40]))
    ax2.set_yticks(np.arange(-0.2,0.6,0.1))
    ax2.plot(Lambda,kD_linear[j,1,:,0],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  
for j in range(len(RA)): # RT 0.75
    ax3.set_xticks(np.array([0,10,20,30,40]))
    ax3.set_yticks(np.arange(-0.15,0.3,0.05))
    ax3.plot(Lambda,kD_linear[j,2,:,0],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  
for j in range(len(RA)): # RT 1.0
    ax4.set_xticks(np.array([0,10,20,30,40]))
    ax4.set_yticks(np.arange(-0.15,0.3,0.05))
    ax4.plot(Lambda,kD_linear[j,3,:,0],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  

ax1.set_ylim([-0.1,0.5])
ax2.set_ylim([-0.1,0.5])
ax3.set_ylim([-0.15,0.2])
ax4.set_ylim([-0.15,0.2])
ax1.set_xlim([0,40])
ax2.set_xlim([0,40])
ax3.set_xlim([0,40])
ax4.set_xlim([0,40])

ax1.text(1,-0.08,r'$R_T = 0.25$',fontsize=12)
ax2.text(1,-0.08,r'$R_T = 0.5$',fontsize=12)
ax3.text(1,-0.14,r'$R_T = 0.75$',fontsize=12)
ax4.text(1,-0.14,r'$R_T = 1.0$',fontsize=12)

ax1.set_ylabel(r'$\kappa_D$',fontsize=12)
ax3.set_ylabel(r'$\kappa_D$',fontsize=12)
ax3.set_xlabel(r'$\Lambda_ {\,c/4}$',fontsize=12)
ax4.set_xlabel(r'$\Lambda_ {\,c/4}$',fontsize=12)

plt.show()
fig.savefig('kappa_D_linear.pdf')




########################################################




# PLOTS KAPPA AC CONSTANT

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

for j in range(len(RA)): # RT 0.25
    ax1.set_xticks(np.array([0,10,20,30,40]))
    ax1.set_yticks(np.arange(0,5,0.5))
    ax1.plot(Lambda_c,kAC_constant[j,0,:,0],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))    
for j in range(len(RA)): # RT 0.5
    ax2.set_xticks(np.array([0,10,20,30,40]))
    ax2.set_yticks(np.arange(0,5,0.5))
    ax2.plot(Lambda_c,kAC_constant[j,1,:,0],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  
for j in range(len(RA)): # RT 0.75
    ax3.set_xticks(np.array([0,10,20,30,40]))
    ax3.set_yticks(np.arange(0,5,0.5))
    ax3.plot(Lambda_c,kAC_constant[j,2,:,0],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  
for j in range(len(RA)): # RT 1.0
    ax4.set_xticks(np.array([0,10,20,30,40]))
    ax4.set_yticks(np.arange(0,5,0.5))
    ax4.plot(Lambda_c,kAC_constant[j,3,:,0],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  

ax1.set_ylim([0,4])
ax2.set_ylim([0,4])
ax3.set_ylim([0,4.5])
ax4.set_ylim([0,4.5])
ax1.set_xlim([0,40])
ax2.set_xlim([0,40])
ax3.set_xlim([0,40])
ax4.set_xlim([0,40])

ax1.text(1,3.6,r'$R_T = 0.25$',fontsize=12)
ax2.text(1,3.6,r'$R_T = 0.5$',fontsize=12)
ax3.text(1,4.1,r'$R_T = 0.75$',fontsize=12)
ax4.text(1,4.1,r'$R_T = 1.0$',fontsize=12)

ax1.set_ylabel(r'$\kappa_{AC}$',fontsize=12)
ax3.set_ylabel(r'$\kappa_{AC}$',fontsize=12)
ax3.set_xlabel(r'$\Lambda_ {\,c/4}$',fontsize=12)
ax4.set_xlabel(r'$\Lambda_ {\,c/4}$',fontsize=12)

plt.show()
fig.savefig('kappa_AC_constant.pdf')


# PLOTS KAPPA AC LINEAR

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

for j in range(len(RA)): # RT 0.25
    ax1.set_xticks(np.array([0,10,20,30,40]))
    ax1.set_yticks(np.arange(0,5,0.25))
    ax1.plot(Lambda,kAC_linear[j,0,:,0],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))    
for j in range(len(RA)): # RT 0.5
    ax2.set_xticks(np.array([0,10,20,30,40]))
    ax2.set_yticks(np.arange(0,5,0.25))
    ax2.plot(Lambda,kAC_linear[j,1,:,0],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  
for j in range(len(RA)): # RT 0.75
    ax3.set_xticks(np.array([0,10,20,30,40]))
    ax3.set_yticks(np.arange(0,5,0.25))
    ax3.plot(Lambda,kAC_linear[j,2,:,0],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  
for j in range(len(RA)): # RT 1.0
    ax4.set_xticks(np.array([0,10,20,30,40]))
    ax4.set_yticks(np.arange(0,5,0.25))
    ax4.plot(Lambda,kAC_linear[j,3,:,0],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  

ax1.set_ylim([0,1.5])
ax2.set_ylim([0,1.5])
ax3.set_ylim([0,2])
ax4.set_ylim([0,2])
ax1.set_xlim([0,40])
ax2.set_xlim([0,40])
ax3.set_xlim([0,40])
ax4.set_xlim([0,40])

ax1.text(1,1.35,r'$R_T = 0.25$',fontsize=12)
ax2.text(1,1.35,r'$R_T = 0.5$',fontsize=12)
ax3.text(1,1.8,r'$R_T = 0.75$',fontsize=12)
ax4.text(1,1.8,r'$R_T = 1.0$',fontsize=12)

ax1.set_ylabel(r'$\kappa_{AC}$',fontsize=12)
ax3.set_ylabel(r'$\kappa_{AC}$',fontsize=12)
ax3.set_xlabel(r'$\Lambda_ {\,c/4}$',fontsize=12)
ax4.set_xlabel(r'$\Lambda_ {\,c/4}$',fontsize=12)

plt.show()
fig.savefig('kappa_AC_linear.pdf')




########################################################




# PLOTS KAPPA L CONSTANT

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

for j in range(len(RA)): # RT 0.25
    ax1.set_xticks(np.array([0,10,20,30,40]))
    ax1.set_yticks(np.arange(0,5,0.05))
    ax1.plot(Lambda_c,kL_constant[j,0,:],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))    
for j in range(len(RA)): # RT 0.5
    ax2.set_xticks(np.array([0,10,20,30,40]))
    ax2.set_yticks(np.arange(0,5,0.05))
    ax2.plot(Lambda_c,kL_constant[j,1,:],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  
for j in range(len(RA)): # RT 0.75
    ax3.set_xticks(np.array([0,10,20,30,40]))
    ax3.set_yticks(np.arange(0,5,0.05))
    ax3.plot(Lambda_c,kL_constant[j,2,:],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  
for j in range(len(RA)): # RT 1.0
    ax4.set_xticks(np.array([0,10,20,30,40]))
    ax4.set_yticks(np.arange(0,5,0.05))
    ax4.plot(Lambda_c,kL_constant[j,3,:],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  

ax1.set_ylim([0.75,1.05])
ax2.set_ylim([0.75,1.05])
ax3.set_ylim([0.75,1.05])
ax4.set_ylim([0.75,1.05])
ax1.set_xlim([0,40])
ax2.set_xlim([0,40])
ax3.set_xlim([0,40])
ax4.set_xlim([0,40])

# ax1.text(1,3.6,r'$R_T = 0.25$',fontsize=12)
# ax2.text(1,3.6,r'$R_T = 0.5$',fontsize=12)
# ax3.text(1,4.1,r'$R_T = 0.75$',fontsize=12)
# ax4.text(1,4.1,r'$R_T = 1.0$',fontsize=12)

ax1.set_ylabel(r'$\kappa_{L}$',fontsize=12)
ax3.set_ylabel(r'$\kappa_{L}$',fontsize=12)
ax3.set_xlabel(r'$\Lambda_ {\,c/4}$',fontsize=12)
ax4.set_xlabel(r'$\Lambda_ {\,c/4}$',fontsize=12)

plt.show()
fig.savefig('kappa_L_constant.pdf')


# PLOTS KAPPA L LINEAR

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

for j in range(len(RA)): # RT 0.25
    ax1.set_xticks(np.array([0,10,20,30,40]))
    ax1.set_yticks(np.arange(0,5,0.05))
    ax1.plot(Lambda,kL_linear[j,0,:],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))    
for j in range(len(RA)): # RT 0.5
    ax2.set_xticks(np.array([0,10,20,30,40]))
    ax2.set_yticks(np.arange(0,5,0.05))
    ax2.plot(Lambda,kL_linear[j,1,:],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  
for j in range(len(RA)): # RT 0.75
    ax3.set_xticks(np.array([0,10,20,30,40]))
    ax3.set_yticks(np.arange(0,5,0.05))
    ax3.plot(Lambda,kL_linear[j,2,:],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  
for j in range(len(RA)): # RT 1.0
    ax4.set_xticks(np.array([0,10,20,30,40]))
    ax4.set_yticks(np.arange(0,5,0.05))
    ax4.plot(Lambda,kL_linear[j,3,:],'k',linestyle='solid',linewidth=0.8,alpha=(j*0.0875+0.3))  

ax1.set_ylim([0.7,1.05])
ax2.set_ylim([0.7,1.05])
ax3.set_ylim([0.7,1.05])
ax4.set_ylim([0.7,1.05])
ax1.set_xlim([0,40])
ax2.set_xlim([0,40])
ax3.set_xlim([0,40])
ax4.set_xlim([0,40])

# ax1.text(1,1.35,r'$R_T = 0.25$',fontsize=12)
# ax2.text(1,1.35,r'$R_T = 0.5$',fontsize=12)
# ax3.text(1,1.8,r'$R_T = 0.75$',fontsize=12)
# ax4.text(1,1.8,r'$R_T = 1.0$',fontsize=12)

ax1.set_ylabel(r'$\kappa_{L}$',fontsize=12)
ax3.set_ylabel(r'$\kappa_{L}$',fontsize=12)
ax3.set_xlabel(r'$\Lambda_ {\,c/4}$',fontsize=12)
ax4.set_xlabel(r'$\Lambda_ {\,c/4}$',fontsize=12)

plt.show()
fig.savefig('kappa_L_linear.pdf')




########################################################








# for i in range(len(RA)):
#     for j in range(len(RT)):
#         RAofchoice = i
#         RTofchoice = j
#         plt.plot(Lambda,kD_linear[RAofchoice,RTofchoice,:,0],label="RA="+str(RA[RAofchoice])+", RT="+str(RT[RTofchoice])+"_linear")
#         plt.plot(Lambda,kD_constant[RAofchoice,RTofchoice,:,0],'--',label="RA="+str(RA[RAofchoice])+", RT="+str(RT[RTofchoice])+"_constant")
# plt.xlim([0,40])
# # plt.legend()
# plt.xlabel("Sweep")
# plt.ylabel("kD")
# # plt.ylim([0.5,3])
# plt.show()

# RAofchoice = 2
# RTofchoice = 0

# # plt.plot(Lambda,kD_linear[RAofchoice,RTofchoice,:,0],label="RA="+str(RA[RAofchoice])+", RT="+str(RT[RTofchoice])+"_linear")
# # plt.plot(Lambda,kD_constant[RAofchoice,RTofchoice,:,0],'--',label="RA="+str(RA[RAofchoice])+", RT="+str(RT[RTofchoice])+"_constant")
# # plt.xlim([0,40])
# # # plt.legend()
# # plt.xlabel("Sweep")
# # plt.ylabel("kD")
# # # plt.ylim([0.5,3])
# # plt.show()

# fig1 = plt.figure(figsize=(4.25,4.25))
# plt.subplot(111)
# plt.plot(Lambda,kAC_linear[RAofchoice,RTofchoice,:,0],'k',label="RA="+str(RA[RAofchoice])+", RT="+str(RT[RTofchoice])+", linear")
# plt.plot(Lambda_c,kAC_constant[RAofchoice,RTofchoice,:,0],'k--',label="RA="+str(RA[RAofchoice])+", RT="+str(RT[RTofchoice])+", constant")
# # plt.xlim([0,40])
# # plt.ylim([0,1.8])
# plt.legend()
# plt.xlabel("$\Lambda_{tip}$")
# plt.ylabel("$\kappa_{AC}$")
# # plt.ylim([0.5,3])
# plt.show()
# fig1.savefig('lambdakac.pdf',bbox_inches='tight')

# # plt.plot(delta_xAC_linear[RAofchoice,RTofchoice,:,0],kD_linear[RAofchoice,RTofchoice,:,0],label="RA="+str(RA[RAofchoice])+", RT="+str(RT[RTofchoice])+"_linear")
# # plt.plot(delta_xAC_constant[RAofchoice,RTofchoice,:,0],kD_constant[RAofchoice,RTofchoice,:,0],'--',label="RA="+str(RA[RAofchoice])+", RT="+str(RT[RTofchoice])+"_constant")
# # plt.xlim([min(delta_xAC_linear[RAofchoice,RTofchoice,:,0]),max(delta_xAC_linear[RAofchoice,RTofchoice,:,0])])
# # # plt.ylim([0.12,0.14])
# # plt.legend()
# # plt.xlabel("delta_xAC")
# # plt.ylabel("kD")
# # plt.show()

# fig1 = plt.figure(figsize=(4.25,4.25))
# plt.subplot(111)
# plt.plot(kAC_linear[RAofchoice,RTofchoice,:,0],kD_linear[RAofchoice,RTofchoice,:,0],'k',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", linear")
# plt.plot(kAC_constant[RAofchoice,RTofchoice,:,0],kD_constant[RAofchoice,RTofchoice,:,0],'k--',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", constant")
# # plt.xlim([min(kAC_linear[RAofchoice,RTofchoice,:,0]),max(kAC_linear[RAofchoice,RTofchoice,:,0])])
# plt.legend()
# plt.xlabel("$\kappa_{AC}$")
# plt.ylabel("$\kappa_D$")
# plt.show()
# fig1.savefig('plot12rt025.pdf',bbox_inches='tight')

# f = interpolate.interp1d(kAC_constant[RAofchoice,RTofchoice,:,0],kD_constant[RAofchoice,RTofchoice,:,0])
# kDlinnew = f(kAC_linear[RAofchoice,RTofchoice,:,0])

# # fig1 = plt.figure(figsize=(4.25,4.25))
# # plt.subplot(111)
# # plt.plot(kAC_linear[RAofchoice,RTofchoice,:,0],kD_linear[RAofchoice,RTofchoice,:,0],'k',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", linear")
# # plt.plot(kAC_linear[RAofchoice,RTofchoice,:,0],kDlinnew,'k--',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", constant")
# # plt.ylim([0,0.1])
# # plt.xlim([0,1.25])
# # plt.legend()
# # plt.xlabel("$\kappa_{AC}$")
# # plt.ylabel("$\kappa_D$")
# # plt.show()
# # fig1.savefig('plot12rt025.pdf',bbox_inches='tight')


# # fig1 = plt.figure(figsize=(4.25,4.25))
# # plt.subplot(111)
# # plt.plot(kAC_linear[RAofchoice,RTofchoice,:,0],abs(kD_linear[RAofchoice,RTofchoice,:,0]-kDlinnew),'k',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", diff")
# # plt.ylim([0,0.1])
# # plt.xlim([0,1.25])
# # plt.legend()
# # plt.xlabel("$\kappa_{AC}$")
# # plt.ylabel("$\kappa_D$")
# # plt.show()
# # fig1.savefig('plot12rt025.pdf',bbox_inches='tight')

# f2 = interpolate.interp1d(kAC_constant[RAofchoice,RTofchoice,:,0],Lambda_c)
# f3 = interpolate.interp1d(kD_linear[RAofchoice,RTofchoice,:,0][1:]-kDlinnew[1:],Lambda[1:])
# f4 = interpolate.interp1d(Lambda,kAC_linear[RAofchoice,RTofchoice,:,0])
# f5 = interpolate.interp1d(Lambda,kD_linear[RAofchoice,RTofchoice,:,0])
# f6 = interpolate.interp1d(kAC_linear[RAofchoice,RTofchoice,:,0],Lambda)

# print("Points cross at kD = ", f5(f3(0)), " kAC = ", f4(f3(0)))
# print("This is at Lambda_tip linear = ", f3(0), ", equivalent to Lambda_tip constant = ", f2(f4(f3(0))))

# # RAofchoice = 4
# # RTofchoice = 1

# # fig1 = plt.figure(figsize=(4.25,4.25))
# # plt.subplot(111)
# # plt.plot(kAC_linear[RAofchoice,RTofchoice,:,0],kD_linear[RAofchoice,RTofchoice,:,0],'k',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", linear")
# # plt.plot(kAC_constant[RAofchoice,RTofchoice,:,0],kD_constant[RAofchoice,RTofchoice,:,0],'k--',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", constant")
# # plt.xlim([min(kAC_linear[RAofchoice,RTofchoice,:,0]),max(kAC_linear[RAofchoice,RTofchoice,:,0])])
# # plt.ylim([-0.1,0.1])
# # plt.legend()
# # plt.xlabel("$\kappa_{AC}$")
# # plt.ylabel("$\kappa_D$")
# # plt.show()
# # fig1.savefig('plot12rt05.pdf',bbox_inches='tight')

# # RAofchoice = 4
# # RTofchoice = 2

# # fig1 = plt.figure(figsize=(4.25,4.25))
# # plt.subplot(111)
# # plt.plot(kAC_linear[RAofchoice,RTofchoice,:,0],kD_linear[RAofchoice,RTofchoice,:,0],'k',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", linear")
# # plt.plot(kAC_constant[RAofchoice,RTofchoice,:,0],kD_constant[RAofchoice,RTofchoice,:,0],'k--',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", constant")
# # plt.xlim([min(kAC_linear[RAofchoice,RTofchoice,:,0]),max(kAC_linear[RAofchoice,RTofchoice,:,0])])
# # plt.ylim([-0.1,0.12])
# # plt.legend()
# # plt.xlabel("$\kappa_{AC}$")
# # plt.ylabel("$\kappa_D$")
# # plt.show()
# # fig1.savefig('plot12rt075.pdf',bbox_inches='tight')

# # RAofchoice = 4
# # RTofchoice = 3

# # fig1 = plt.figure(figsize=(4.25,4.25))
# # plt.subplot(111)
# # plt.plot(kAC_linear[RAofchoice,RTofchoice,:,0],kD_linear[RAofchoice,RTofchoice,:,0],'k',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", linear")
# # plt.plot(kAC_constant[RAofchoice,RTofchoice,:,0],kD_constant[RAofchoice,RTofchoice,:,0],'k--',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", constant")
# # plt.xlim([min(kAC_linear[RAofchoice,RTofchoice,:,0]),max(kAC_linear[RAofchoice,RTofchoice,:,0])])
# # plt.ylim([-0.1,0.15])
# # plt.legend()
# # plt.xlabel("$\kappa_{AC}$")
# # plt.ylabel("$\kappa_D$")
# # plt.show()
# # fig1.savefig('plot12rt1.pdf',bbox_inches='tight')


# # RAofchoice = 0
# # RTofchoice = 0

# # fig1 = plt.figure(figsize=(4.25,4.25))
# # plt.subplot(111)
# # plt.plot(kAC_linear[RAofchoice,RTofchoice,:,0],kD_linear[RAofchoice,RTofchoice,:,0],'k',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", linear")
# # plt.plot(kAC_constant[RAofchoice,RTofchoice,:,0],kD_constant[RAofchoice,RTofchoice,:,0],'k--',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", constant")
# # plt.xlim([min(kAC_linear[RAofchoice,RTofchoice,:,0]),max(kAC_linear[RAofchoice,RTofchoice,:,0])])
# # plt.ylim([-0.01,0.1])
# # plt.legend()
# # plt.xlabel("$\kappa_{AC}$")
# # plt.ylabel("$\kappa_D$")
# # plt.show()
# # fig1.savefig('plot4rt025.pdf',bbox_inches='tight')

# # RAofchoice = 3
# # RTofchoice = 0

# # fig1 = plt.figure(figsize=(4.25,4.25))
# # plt.subplot(111)
# # plt.plot(kAC_linear[RAofchoice,RTofchoice,:,0],kD_linear[RAofchoice,RTofchoice,:,0],'k',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", linear")
# # plt.plot(kAC_constant[RAofchoice,RTofchoice,:,0],kD_constant[RAofchoice,RTofchoice,:,0],'k--',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", constant")
# # plt.xlim([min(kAC_linear[RAofchoice,RTofchoice,:,0]),max(kAC_linear[RAofchoice,RTofchoice,:,0])])
# # plt.ylim([0,0.07])
# # plt.legend()
# # plt.xlabel("$\kappa_{AC}$")
# # plt.ylabel("$\kappa_D$")
# # plt.show()
# # fig1.savefig('plot10rt025.pdf',bbox_inches='tight')

# # RAofchoice = 6
# # RTofchoice = 0

# # fig1 = plt.figure(figsize=(4.25,4.25))
# # plt.subplot(111)
# # plt.plot(kAC_linear[RAofchoice,RTofchoice,:,0],kD_linear[RAofchoice,RTofchoice,:,0],'k',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", linear")
# # plt.plot(kAC_constant[RAofchoice,RTofchoice,:,0],kD_constant[RAofchoice,RTofchoice,:,0],'k--',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", constant")
# # plt.xlim([min(kAC_linear[RAofchoice,RTofchoice,:,0]),max(kAC_linear[RAofchoice,RTofchoice,:,0])])
# # plt.ylim([0.04,0.1])
# # plt.legend()
# # plt.xlabel("$\kappa_{AC}$")
# # plt.ylabel("$\kappa_D$")
# # plt.show()
# # fig1.savefig('plot16rt025.pdf',bbox_inches='tight')

# # RAofchoice = 8
# # RTofchoice = 0

# # fig1 = plt.figure(figsize=(4.25,4.25))
# # plt.subplot(111)
# # plt.plot(kAC_linear[RAofchoice,RTofchoice,:,0],kD_linear[RAofchoice,RTofchoice,:,0],'k',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", linear")
# # plt.plot(kAC_constant[RAofchoice,RTofchoice,:,0],kD_constant[RAofchoice,RTofchoice,:,0],'k--',label="$R_A$="+str(RA[RAofchoice])+", $R_T$="+str(RT[RTofchoice])+", constant")
# # plt.xlim([min(kAC_linear[RAofchoice,RTofchoice,:,0]),max(kAC_linear[RAofchoice,RTofchoice,:,0])])
# # plt.ylim([0.04,0.1])
# # plt.legend()
# # plt.xlabel("$\kappa_{AC}$")
# # plt.ylabel("$\kappa_D$")
# # plt.show()
# # fig1.savefig('plot20rt025.pdf',bbox_inches='tight')


# # # plt.plot(kAC_linear[RAofchoice,RTofchoice,:,0],kD_linear[RAofchoice,RTofchoice,:,0],label="RA="+str(RA[RAofchoice])+", RT="+str(RT[RTofchoice])+"_linear")
# # # plt.plot(kAC_constant[RAofchoice,RTofchoice,:,0],kD_constant[RAofchoice,RTofchoice,:,0],'--',label="RA="+str(RA[RAofchoice])+", RT="+str(RT[RTofchoice])+"_constant")
# # # plt.xlim([min(kAC_linear[RAofchoice,RTofchoice,:,0]),max(kAC_linear[RAofchoice,RTofchoice,:,0])])
# # # # plt.ylim([0.12,0.14])
# # # plt.legend()
# # # plt.xlabel("kAC")
# # # plt.ylabel("kD")
# # # plt.show()