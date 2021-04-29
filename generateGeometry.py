# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:37:09 2021

@author: bmoor
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import subprocess
import json
import shutil
import fileinput
from NACA4 import node_NACA
import csv
import xlsxwriter
import scipy.optimize as optimize
import time

N_spanwise = 2
RT = 0.25
RA = 8
Lambda = 45
b = RA*(1+RT)/2
b_semi = b/2
MAC = (1+RT)/2
Sref = b*MAC

naca = '0012'
n_airfoil = 200
c_max = int(naca[0:1])/100
c_loc = int(naca[1:2])/10
t_max = int(naca[2:4])/100

x_root, z_root, camber_root = node_NACA(c_max,c_loc,t_max,n_airfoil)
x_root = x_root - 0.25

x_tip = x_root*RT + np.tan(np.deg2rad(Lambda))*b_semi
z_tip = z_root*RT

z_root = np.flip(z_root)
z_tip = np.flip(z_tip)

coords_root = np.zeros(3*len(x_root))
for i in range(len(x_root)):
    coords_root[3*i] = x_root[i]
    coords_root[3*i+1] = 0
    coords_root[3*i+2] = z_root[i]

coords_tip = np.zeros(3*len(x_tip))
for i in range(len(x_root)):
    coords_tip[3*i] = x_tip[i]
    coords_tip[3*i+1] = b_semi
    coords_tip[3*i+2] = z_tip[i]

coords_other_tip = np.zeros(3*len(x_tip))
for i in range(len(x_root)):
    coords_tip[3*i] = x_tip[i]
    coords_tip[3*i+1] = -b_semi
    coords_tip[3*i+2] = z_tip[i]

n_span = 80
growthtype = 3
growthrate = 1.2
periodicity = 2

with open('wing.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile,delimiter=';')
    
    writer.writerow(['Aircraft','Model'])
    writer.writerow(['Parameter','WingRefArea',Sref])
    writer.writerow(['Parameter','MAC',MAC])
    writer.writerow([])
    writer.writerow(['Component','Wing'])
    writer.writerow(['LiftingSurface','true'])
    writer.writerow(['Mesh',80,20,growthtype,growthrate,periodicity,'false'])
    writer.writerow(['CrossSection']+np.ndarray.tolist(coords_root))
    writer.writerow(['CrossSection']+np.ndarray.tolist(coords_tip))