# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:06:40 2021

@author: bmoor
"""

import numpy as np
from NACA4 import node_NACA
import csv

class scene:
    def __init__(self, alpha, V_inf, outputFile, savedRunFile, simulationTemplate):
        self.alpha = alpha
        self.V_inf = V_inf
        self.outputFile = outputFile
        self.savedRunFile = savedRunFile
        self.simulationTemplate = simulationTemplate