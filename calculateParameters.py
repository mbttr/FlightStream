# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:32:47 2021

@author: bmoor
"""

import numpy as np

def calculateParameters(CL,CDi,Cm,wing,scene):
    CLa = 2*CL/np.deg2rad(2*scene.alpha)
    Cma = 2*Cm/np.deg2rad(2*scene.alpha)
    CLasec= 6.9207
    
    xAC_c = -Cma/CLa + 0.25/wing.MAC
    kappaD = CDi*np.pi*wing.RA/(CL**2)-1
    kappaL = CLasec/(CLa*(1+CLasec/(np.pi*wing.RA))) - 1
    
    return kappaD, xAC_c, kappaL