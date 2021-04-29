# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:46:02 2021

@author: bmoor
"""

def readFlightStreamOutput(outputFile):
    with open(outputFile, 'r') as file:
        output = file.readlines()
    
    data = output[29].split("\x00\x00,")
    CL = float(data[4])
    CDi = float(data[5])
    Cm = float(data[8])
    return CL, CDi, Cm