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
        
        if self.curvetype == 'elliptic':
            self.b = np.pi*self.RA/4
            self.S_ref = self.b**2/self.RA
            self.MAC = self.S_ref/self.b
            self.b_semi = self.b/2
        
        self.n_span = n_span
        self.n_airfoil = n_airfoil
        
        self.writeGeom = writeGeom

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
            
            # fig1 = plt.figure(figsize=(4.25,4.25))
            # plt.subplot(111)
            # plt.plot(self.y[:,0],-self.x[:,0],'k')
            # plt.plot(self.y[:,-1],-self.x[:,-1],'k')
            # plt.plot(self.y[0],-self.x[0],'k')
            # plt.plot(self.y[0],-self.x[100],'k')
            # plt.plot(-self.y[:,0],-self.x[:,0],'k')
            # plt.plot(-self.y[:,-1],-self.x[:,-1],'k')
            # plt.plot(-self.y[0],-self.x[0],'k')
            # plt.plot(-self.y[0],-self.x[100],'k')
            # plt.xlabel('y')
            # plt.ylabel('x')
            # plt.axis('equal')
            # # plt.axis('off')
            # plt.show()
            # fig1.savefig('wingplot.pdf',bbox_inches='tight')
        
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
            
            fig1 = plt.figure(figsize=(4.25,4.25))
            plt.subplot(111)
            plt.plot(self.y[:,0],-self.x[:,0],'k')
            plt.plot(self.y[:,-1],-self.x[:,-1],'k')
            plt.plot(self.y[0],-self.x[0],'k')
            plt.plot(self.y[0],-self.x[100],'k')
            plt.plot(-self.y[:,0],-self.x[:,0],'k')
            plt.plot(-self.y[:,-1],-self.x[:,-1],'k')
            plt.plot(-self.y[0],-self.x[0],'k')
            plt.plot(-self.y[0],-self.x[100],'k')
            plt.xlabel('y')
            plt.ylabel('x')
            plt.axis('equal')
            # plt.axis('off')
            plt.show()
            # fig1.savefig('wingplot.pdf',bbox_inches='tight')

            # fig1 = plt.figure(figsize=(4.25,4.25))
            # plt.subplot(111)
            # plt.plot(self.y[0]/np.max(self.y[0]),self.Lambdaarray,'k')
            # plt.xlabel('y / (b/2)')
            # plt.ylabel('$\Lambda$')
            # plt.xlim([0,1])
            # plt.ylim([0,20])
            # # plt.axis('equal')
            # # plt.axis('off')
            # plt.show()
            # fig1.savefig('lambdaplot.pdf',bbox_inches='tight')


        if curvetype == "elliptic":    
            self.N_sections = 200
            self.y = np.zeros((len(x_root),self.N_sections)) 
            self.y[:,:] = np.linspace(0,self.b_semi,self.N_sections)
            
            self.c = 4*self.b/(np.pi*self.RA)*np.sqrt(1-(2*self.y/self.b)**2)
            self.RTarray = self.c
            
            self.x_airfoil = np.zeros((len(x_root),self.N_sections))
            self.x_quarterChord = np.zeros(self.N_sections)
            self.z = np.zeros((len(x_root),self.N_sections))
            
            self.x_airfoil = x_root[:,None]*self.RTarray
            self.x = self.x_airfoil + self.x_quarterChord
            self.z = z_root[:,None]*self.RTarray
            
            fig1 = plt.figure(figsize=(4.25,4.25))
            plt.subplot(111)
            plt.plot(self.y[:,0],-self.x[:,0],'k')
            plt.plot(self.y[:,-1],-self.x[:,-1],'k')
            plt.plot(self.y[0],-self.x[0],'k')
            plt.plot(self.y[0],-self.x[100],'k')
            plt.plot(-self.y[:,0],-self.x[:,0],'k')
            plt.plot(-self.y[:,-1],-self.x[:,-1],'k')
            plt.plot(-self.y[0],-self.x[0],'k')
            plt.plot(-self.y[0],-self.x[100],'k')
            plt.xlabel('y')
            plt.ylabel('x')
            plt.axis('equal')
            # plt.axis('off')
            plt.show()
            # fig1.savefig('wingplot.pdf',bbox_inches='tight')

            # fig1 = plt.figure(figsize=(4.25,4.25))
            # plt.subplot(111)
            # plt.plot(self.y[0]/np.max(self.y[0]),self.Lambdaarray,'k')
            # plt.xlabel('y / (b/2)')
            # plt.ylabel('$\Lambda$')
            # plt.xlim([0,1])
            # plt.ylim([0,20])
            # # plt.axis('equal')
            # # plt.axis('off')
            # plt.show()
            # fig1.savefig('lambdaplot.pdf',bbox_inches='tight')


        if curvetype == "quad":   
            self.RTarray = np.linspace(1,self.RT,self.N_sections)
            
            self.x_airfoil = np.zeros((len(x_root),self.N_sections))
            self.y = np.zeros((len(x_root),self.N_sections))
            self.z = np.zeros((len(x_root),self.N_sections))
            
            self.y[:,:] = np.linspace(0,self.b_semi,self.N_sections)
            self.x_airfoil = x_root[:,None]*self.RTarray
            self.x_quarterChord = np.zeros(self.N_sections)
            
            
            self.x_quarterChord = 0*self.y**3 + 0.18*self.y**2 +0*self.y
            self.x = self.x_airfoil + self.x_quarterChord
            self.z = z_root[:,None]*self.RTarray


            fig1 = plt.figure(figsize=(4.25,4.25))
            plt.subplot(111)
            plt.plot(self.y[:,0],-self.x[:,0],'k')
            plt.plot(self.y[:,-1],-self.x[:,-1],'k')
            plt.plot(self.y[0],-self.x[0],'k')
            plt.plot(self.y[0],-self.x[100],'k')
            # plt.plot(-self.y[:,0],-self.x[:,0],'k')
            # plt.plot(-self.y[:,-1],-self.x[:,-1],'k')
            # plt.plot(-self.y[0],-self.x[0],'k')
            # plt.plot(-self.y[0],-self.x[100],'k')
            plt.xlabel('y')
            plt.ylabel('x')
            plt.axis('equal')
            # plt.axis('off')
            plt.show()
            # fig1.savefig('wingplot.pdf',bbox_inches='tight')



        if curvetype == "optimizequad":   
            self.RTarray = np.linspace(1,self.RT,self.N_sections)
            
            self.x_airfoil = np.zeros((len(x_root),self.N_sections))
            self.y = np.zeros((len(x_root),self.N_sections))
            self.z = np.zeros((len(x_root),self.N_sections))
            
            self.y[:,:] = np.linspace(0,self.b_semi,self.N_sections)
            self.x_airfoil = x_root[:,None]*self.RTarray
            self.x_quarterChord = np.zeros(self.N_sections)
            
            
            self.x_quarterChord = xOptim[0]*self.y**3 + xOptim[1]*self.y**2 + xOptim[2]*self.y
            self.x = self.x_airfoil + self.x_quarterChord
            self.z = z_root[:,None]*self.RTarray


            fig1 = plt.figure(figsize=(4.25,4.25))
            plt.subplot(111)
            plt.plot(self.y[:,0],-self.x[:,0],'k')
            plt.plot(self.y[:,-1],-self.x[:,-1],'k')
            plt.plot(self.y[0],-self.x[0],'k')
            plt.plot(self.y[0],-self.x[100],'k')
            # plt.plot(-self.y[:,0],-self.x[:,0],'k')
            # plt.plot(-self.y[:,-1],-self.x[:,-1],'k')
            # plt.plot(-self.y[0],-self.x[0],'k')
            # plt.plot(-self.y[0],-self.x[100],'k')
            plt.xlabel('y')
            plt.ylabel('x')
            plt.axis('equal')
            # plt.axis('off')
            plt.show()
            # fig1.savefig('wingplot.pdf',bbox_inches='tight')


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

            # plt.plot(self.y,self.x)
            # # plt.plot(self.y[0],self.x_quarterChord)
            # plt.plot(self.y[0],self.x[0])
            # plt.plot(self.y[0],self.x[100])
            # plt.axis('equal')
            # plt.show()

            # fig1 = plt.figure(figsize=(4.25,4.25))
            # plt.subplot(111)
            # plt.plot(self.y[:,0],-self.x[:,0],'k')
            # plt.plot(self.y[:,-1],-self.x[:,-1],'k')
            # plt.plot(self.y[0],-self.x[0],'k')
            # plt.plot(self.y[0],-self.x[100],'k')
            # # plt.plot(-self.y[:,0],-self.x[:,0],'k')
            # # plt.plot(-self.y[:,-1],-self.x[:,-1],'k')
            # # plt.plot(-self.y[0],-self.x[0],'k')
            # # plt.plot(-self.y[0],-self.x[100],'k')
            # plt.xlabel('y')
            # plt.ylabel('x')
            # plt.axis('equal')
            # # plt.axis('off')
            # plt.show()
            # # fig1.savefig('wingplot.pdf',bbox_inches='tight')

 
        if curvetype == "defined":   
            self.RTarray = np.linspace(1,self.RT,self.N_sections) #np.array([1,	0.835,	0.665,	0.585,	0.61,	0.58,	0.52,	0.53,	0.47,	0.395,	0.22,	0.02])#
            
            self.x_airfoil = np.zeros((len(x_root),self.N_sections))
            self.y = np.zeros((len(x_root),self.N_sections))
            self.z = np.zeros((len(x_root),self.N_sections))
            
            self.y[:,:] = np.linspace(0,self.b_semi,self.N_sections)#np.array([0,	0.3,	0.5,	0.8,	1.55,	1.8,	1.92,	2.2,	2.6,	2.75,	3.2,	3.4])
            self.x_airfoil = x_root[:,None]*self.RTarray
            self.x_quarterChord = np.zeros(self.N_sections)
            self.x_quarterChord[1:] = xOptim
            self.x = self.x_airfoil + self.x_quarterChord
            self.z = z_root[:,None]*self.RTarray
            
            # self.MAC = 0.544
            # self.S_ref = 3.7

            fig1 = plt.figure(figsize=(4.25,4.25))
            plt.subplot(111)
            plt.plot(self.y[:,0],-self.x[:,0],'k')
            plt.plot(self.y[:,-1],-self.x[:,-1],'k')
            plt.plot(self.y[0],-self.x[0],'k')
            plt.plot(self.y[0],-self.x[100],'k')
            plt.plot(self.y[0],-(self.x[100]+(self.x[0]-self.x[100])/4),'k--',linewidth=0.3)
            plt.plot(-self.y[:,0],-self.x[:,0],'k')
            plt.plot(-self.y[:,-1],-self.x[:,-1],'k')
            plt.plot(-self.y[0],-self.x[0],'k')
            plt.plot(-self.y[0],-self.x[100],'k')
            plt.plot(-self.y[0],-(self.x[100]+(self.x[0]-self.x[100])/4),'k--',linewidth=0.3)
            plt.xlabel('y')
            plt.ylabel('x')
            plt.axis('equal')
            # plt.axis('off')
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
                
        self.growthtype = 3
        self.growthrate = 1.2 #0.925
        self.periodicity = 2 #1
        
        if self.curvetype == "optimize":
            self.filename = "RA"+str(self.RA)+"_RT"+str(self.RT)+"_LambdaOptimize"
            self.CSVFile = self.folder+self.filename+".csv"
        elif self.curvetype == "defined":
            self.filename = "RA"+str(self.RA)+"_RT"+str(self.RT)+"_xDefined"
            self.CSVFile = self.folder+self.filename+".csv"
        else:
            self.filename = "RA"+str(self.RA)+"_RT"+str(self.RT)+"_Lambda"+str(self.Lambda)
            self.CSVFile = self.folder+self.filename+".csv"
            
        self.TEFile = self.folder+self.filename+"_TE.txt"