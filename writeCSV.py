# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:50:23 2021

@author: Bruno-USU
"""

import csv
import numpy as np

def writeCSV(wing):        

    if wing.writeGeom == 'y':
        with open(wing.CSVFile, 'w',newline='') as csvfile:
            writer = csv.writer(csvfile,delimiter=';')
            
            writer.writerow(['Aircraft','Model'])
            writer.writerow(['Parameter','WingRefArea',wing.S_ref])
            writer.writerow(['Parameter','MAC',wing.MAC])
            writer.writerow([])
            writer.writerow(['Component','Wing'])
            writer.writerow(['LiftingSurface','true'])
            writer.writerow(['Mesh',wing.n_airfoil,wing.n_span,wing.growthtype,wing.growthrate,wing.periodicity,'false'])
            writer.writerow(['Parameter','Mark_trailing_edges'])
            for i in range(wing.N_sections):
                writer.writerow(['CrossSection']+np.ndarray.tolist(wing.coords[:,i]))
                
        with open(wing.TEFile, 'w',newline='') as TEfile:
            writer = csv.writer(TEfile,delimiter=',')
            # writer.writerow(['#************************************************************************'])
            # writer.writerow(['#************** Trailing Edge vertices file *****************************'])
            # writer.writerow(['#************************************************************************'])
            # writer.writerow(['#'])
            writer.writerow([wing.N_sections])
            writer.writerow(['METER'])
            for i in range(wing.N_sections):
                writer.writerow([i+1,wing.x[0,i],wing.y[0,i],wing.z[0,i]])