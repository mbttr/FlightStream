# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:06:40 2021

@author: bmoor
"""

import numpy as np
from NACA4 import node_NACA
import csv
import matplotlib.pyplot as plt

class wing:
    def __init__(self, RA, RT, Lambda, naca, writeGeom, folder, n_span, n_airfoil, N_sections, curvetype, xOptim):  
        self.RA = RA  
        self.RT = RT
        self.Lambda = Lambda
        self.naca = naca
        self.folder = folder

        self.b = self.RA*(1+self.RT)/2
        self.b_semi = self.b/2
        self.MAC = (1+self.RT)/2
        self.S_ref = self.b*self.MAC
        
        self.N_sections = N_sections
        self.curvetype = curvetype

        n_NACA = 200
        c_max = int(self.naca[0:1])/100
        c_loc = int(self.naca[1:2])/10
        t_max = int(self.naca[2:4])/100

        x_root, z_root, camber_root = node_NACA(c_max,c_loc,t_max,n_NACA)
        x_root = x_root - 0.25

        if curvetype == "constant":  
            self.N_sections = 2
            
            self.RTarray = np.linspace(1,self.RT,self.N_sections)
            self.Lambdaarray = np.linspace(self.Lambda,self.Lambda,self.N_sections)
            
            self.x_airfoil = np.zeros((len(x_root),self.N_sections))
            self.x_quarterChord = np.zeros((len(x_root),self.N_sections))
            self.y = np.zeros((len(x_root),self.N_sections))
            self.z = np.zeros((len(x_root),self.N_sections))
            
            self.y[:,:] = np.linspace(0,self.b_semi,self.N_sections)
            self.x_airfoil = x_root[:,None]*self.RTarray
            self.x_quarterChord = np.tan(np.deg2rad(self.Lambda))*self.y
            self.x = self.x_airfoil + self.x_quarterChord
            self.z = z_root[:,None]*self.RTarray
            
            plt.plot(self.y,self.x)
            plt.plot(self.y[0],self.x_quarterChord[0])
            plt.plot(self.y[0],self.x[0])
            plt.plot(self.y[0],self.x[100])
            plt.axis('equal')
            plt.show()
        
        if curvetype == "linear":            
            self.RTarray = np.linspace(1,self.RT,self.N_sections)
            self.Lambdaarray = np.linspace(0,self.Lambda,self.N_sections)
            
            self.x_airfoil = np.zeros((len(x_root),self.N_sections))
            self.x_quarterChord = np.zeros(self.N_sections)
            self.y = np.zeros((len(x_root),self.N_sections))
            self.z = np.zeros((len(x_root),self.N_sections))
            
            self.y[:,:] = np.linspace(0,self.b_semi,self.N_sections)
            self.x_airfoil = x_root[:,None]*self.RTarray
            self.x_quarterChord = np.tan(np.deg2rad(self.Lambdaarray))*self.y
            self.x = self.x_airfoil + self.x_quarterChord
            self.z = z_root[:,None]*self.RTarray

            plt.plot(self.y,self.x)
            plt.plot(self.y[0],self.x_quarterChord[0])
            plt.plot(self.y[0],self.x[0])
            plt.plot(self.y[0],self.x[100])
            plt.axis('equal')
            plt.show()

        if curvetype == "optimize":            
            self.RTarray = np.linspace(1,self.RT,self.N_sections)
            
            self.x_airfoil = np.zeros((len(x_root),self.N_sections))
            self.y = np.zeros((len(x_root),self.N_sections))
            self.z = np.zeros((len(x_root),self.N_sections))
            
            self.y[:,:] = np.linspace(0,self.b_semi,self.N_sections)
            self.x_airfoil = x_root[:,None]*self.RTarray
            self.x_quarterChord = np.zeros(self.N_sections)
            self.x_quarterChord[1:] = xOptim
            self.x = self.x_airfoil + self.x_quarterChord
            self.z = z_root[:,None]*self.RTarray

            plt.plot(self.y,self.x)
            plt.plot(self.y[0],self.x_quarterChord)
            plt.plot(self.y[0],self.x[0])
            plt.plot(self.y[0],self.x[100])
            plt.axis('equal')
            plt.show()
        
        self.z = np.flip(self.z,0)    
        self.y[:,1:] = -self.y[:,1:]
        self.Sweep = np.rad2deg(np.arctan2(self.x[:,1:]-self.x[:,:-1],-(self.y[:,1:]-self.y[:,:-1])))
        
        self.Trefftz = np.max(self.x) + self.b
        
        self.coords = np.zeros((3*len(x_root),self.N_sections))
        
        for i in range(self.N_sections):
            for j in range(len(x_root)):
                self.coords[3*j,i] = self.x[j,i]
                self.coords[3*j+1,i] = self.y[j,i]
                self.coords[3*j+2,i] = self.z[j,i]
        
        growthtype = 3
        growthrate = 1.2
        periodicity = 2
        
        if curvetype == "optimize":
            self.filename = "RA"+str(self.RA)+"_RT"+str(self.RT)+"_LambdaOptimize"
            self.CSVFile = folder+self.filename+".csv"
        else:
            self.filename = "RA"+str(self.RA)+"_RT"+str(self.RT)+"_Lambda"+str(self.Lambda)+".csv"
            self.CSVFile = folder+self.filename+".csv"
            

        if writeGeom == 'y':
            with open(self.CSVFile, 'w',newline='') as csvfile:
                writer = csv.writer(csvfile,delimiter=';')
                
                writer.writerow(['Aircraft','Model'])
                writer.writerow(['Parameter','WingRefArea',self.S_ref])
                writer.writerow(['Parameter','MAC',self.MAC])
                writer.writerow([])
                writer.writerow(['Component','Wing'])
                writer.writerow(['LiftingSurface','true'])
                writer.writerow(['Mesh',n_airfoil,n_span,growthtype,growthrate,periodicity,'false'])
                for i in range(self.N_sections):
                    writer.writerow(['CrossSection']+np.ndarray.tolist(self.coords[:,i]))