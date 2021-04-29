#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 09:15:55 2018

@author: bruno
"""

import csv

def node_cosine(N_2,c):
	import numpy as np
	
	x = np.zeros(N_2)
	for i in range(N_2):
		dtheta = np.pi/(float(N_2) - .5)
		x[i] = .5*c*(1. - np.cos(float(i+1)*dtheta - .5*dtheta))
	
	return x

#This subroutine generates the nodes for a NACA XXXX airfoil
def node_NACA(c_max,c_loc,t_max,n):
	import numpy as np
	from scipy.optimize import newton
	import matplotlib.pyplot as plt
    
	x_flap = 0.7
	delta = 0
    
	c = 1

	N_2 = int(n/2)
	N = 2*N_2
	x = node_cosine(N_2,c)
	Node_x = np.zeros(N)
	Node_y = np.zeros(N)
	Node_c = np.zeros((2,N_2))
	flap = 'p'     
    
	def yc_flap(x,c,m,p):
			
		if (x <= p):
			yc = m*(2.*(x/(p)) - (x/(p))**2)
		else:
			yc = m*(2.*((c - x)/(c - p)) - ((c - x)/(c - p))**2)
		
		return yc
    
	y_flap = yc_flap(x_flap,c,c_max,c_loc)
	
	#Thickness of the airfoil
	def yt(x,c,t_max):
				
		#Closed TE		
		yt = .5*t_max*(2.980*np.sqrt(x/c) - 1.320*(x/c) - 3.286*((x/c)**2) + 2.441*((x/c)**3) - 0.815*((x/c)**4))
		
		#Open TE
# 		yt = .5*t_max*(2.969*np.sqrt(x/c) - 1.260*(x/c) - 3.516*((x/c)**2) + 2.843*((x/c)**3) - 1.015*((x/c)**4))
		
		return yt
	

	if flap.lower() == 'n':
		#y location of camber line as a function of x
		def yc(x,c,m,p):
				
			if (x <= p):
				yc = m*(2.*(x/(p)) - (x/(p))**2)
			else:
				yc = m*(2.*((c - x)/(c - p)) - ((c - x)/(c - p))**2)
			
			return yc
			
		#Derivative of the camber line
		def dyc(x,c,m,p):
			
			if (x <= p):
				dyc = (2.*m)*((1./p) - x/(p*p))
			else:
				dyc = (-2.*m)*(1./(c - p) - (c - x)/(c - p)**2)
			
			return dyc

			
		for i in range(N_2):
			
			upper = N_2 + i
			lower = N_2 - i - 1
			
			Node_c[:,lower] = [x[i],yc(x[i],c,c_max,c_loc)]
			
			Theta = np.arctan(dyc(x[i],c,c_max,c_loc))
			
			Node_x[upper] = x[i] - yt(x[i],c,t_max)*np.sin(Theta)
			Node_y[upper] = yc(x[i],c,c_max,c_loc) + yt(x[i],c,t_max)*np.cos(Theta)
			
			Node_x[lower] = x[i] + yt(x[i],c,t_max)*np.sin(Theta)
			Node_y[lower] = yc(x[i],c,c_max,c_loc) - yt(x[i],c,t_max)*np.cos(Theta)
		
	elif flap.lower() == 't':
		#y location of camber line as a function of x
		def xyc_t(x,c,m,p,delta):
			
			if (x > x_flap):
				
				if (x <= p):
					yc = m*(2.*(x/(p)) - (x/(p))**2)
				else:
					yc = m*(2.*((c - x)/(c - p)) - ((c - x)/(c - p))**2)
				
				r = np.sqrt((yc - y_flap)**2 + (x - x_flap)**2)
				theta = np.arctan2(yc - y_flap, x - x_flap)
				
				xc = x_flap + r*np.cos(delta - theta)
				yc = y_flap - r*np.sin(delta - theta)
				
			else:
				
				if (x <= p):
					yc = m*(2.*(x/(p)) - (x/(p))**2)
				else:
					yc = m*(2.*((c - x)/(c - p)) - ((c - x)/(c - p))**2)
				
				xc = x
			
			return xc, yc
			
			
		#Derivative of the camber line
		def dyc_t(x,c,m,p,delta):
			
			if x > x_flap:
				
				if (x <= p):
					dyc = (2.*m)*((1./p) - x/(p*p))
				else:
					dyc = (-2.*m)*(1./(c - p) - (c - x)/(c - p)**2)
				
				dyc = (dyc - np.tan(delta))/(1 + dyc*np.tan(delta))
				
			else:
				if (x <= p):
					dyc = (2.*m)*((1./p) - x/(p*p))
				else:
					dyc = (-2.*m)*(1./(c - p) - (c - x)/(c - p)**2)
			
			return dyc
		
		
		for i in range(N_2):
			
			upper = N_2 + i
			lower = N_2 - i - 1
			
			xc, yc = xyc_t(x[i],c,c_max,c_loc,delta)
			
			Node_c[:,lower] = [xc,yc]
			
			Theta = np.arctan(dyc_t(x[i],c,c_max,c_loc,delta))
			
			Node_x[upper] = xc - yt(x[i],c,t_max)*np.sin(Theta)
			Node_y[upper] = yc + yt(x[i],c,t_max)*np.cos(Theta)
			
			Node_x[lower] = xc + yt(x[i],c,t_max)*np.sin(Theta)
			Node_y[lower] = yc - yt(x[i],c,t_max)*np.cos(Theta)
			
		
	elif flap.lower() == 'p':
		#y location of camber line as a function of x
		def xyc_p(x,c,m,p,delta,x_flap,y_flap,R,l,phi,chi_TE):
			
			if (x > x_flap):
				
				if (x <= p):
					yc = m*(2.*(x/(p)) - (x/(p))**2)
				else:
					yc = m*(2.*((c - x)/(c - p)) - ((c - x)/(c - p))**2)
				
				chi_0 = (x - x_flap)*l/(c - x_flap)
				
				def chi_func(chi_p,l,R,chi_0):
					if delta == 0:
						return 0.5*chi_p*np.sqrt((chi_p*R*delta/l)**2 + 1) + chi_p/2 - chi_0
					else:
						return 0.5*chi_p*np.sqrt((chi_p*R*np.tan(delta)/l)**2 + 1) + l*np.arcsinh(chi_p*R*np.tan(delta)/l)/(2*R*np.tan(delta)) - chi_0
				chi_p = newton(chi_func,chi_0*chi_TE/l,args=(l,R,chi_0))
				eta_p = -chi_p**2 * np.tan(delta)/chi_TE
				
				xp = x_flap + chi_p*np.cos(phi) - eta_p*np.sin(phi)
				yp = y_flap + chi_p*np.sin(phi) + eta_p*np.cos(phi)
				
				yc_d = yc - y_flap*(1 - (x - x_flap)/(c - x_flap))
				
				xc = xp + yc_d*np.sin(np.arctan2(2*chi_p*np.tan(delta),chi_TE))
				yc = yp + yc_d*np.cos(np.arctan2(2*chi_p*np.tan(delta),chi_TE))
				
			else:
				
				if (x <= p):
					yc = m*(2.*(x/(p)) - (x/(p))**2)
				else:
					yc = m*(2.*((c - x)/(c - p)) - ((c - x)/(c - p))**2)
				
				xc = x
				chi_p = x
			
			return xc, yc, chi_p
			
			
		#Derivative of the camber line
		def dyc_p(x,c,m,p,delta,x_flap,chi_p,chi_TE):
			
			if x > x_flap:
				
				if (x <= p):
					dyc = (2.*m)*((1./p) - x/(p*p))
				else:
					dyc = (-2.*m)*(1./(c - p) - (c - x)/(c - p)**2)
				
				dyc = (dyc - 2*chi_p*np.tan(delta)/chi_TE)/(1 + dyc*2*chi_p*np.tan(delta)/chi_TE)
				
			else:
				if (x <= p):
					dyc = (2.*m)*((1./p) - x/(p*p))
				else:
					dyc = (-2.*m)*(1./(c - p) - (c - x)/(c - p)**2)
			
			return dyc
		
		#Calculate required constants
		if delta == 0:
			R = np.sqrt(4*delta**2 + 1) + 1
		else:
			R = np.sqrt(4*np.tan(delta)**2 + 1) + np.arcsinh(2*np.tan(delta))/(2*np.tan(delta))
		l = np.sqrt(y_flap**2 + (c - x_flap)**2)
		phi = np.arctan2(-y_flap,c-x_flap)
		chi_TE = 2*l/R
		
		for i in range(N_2):
			
			upper = N_2 + i
			lower = N_2 - i - 1
			
			xc, yc, chi_p = xyc_p(x[i],c,c_max,c_loc,delta,x_flap,y_flap,R,l,phi,chi_TE)
			
			Node_c[:,lower] = [xc,yc]
			
			Theta = np.arctan(dyc_p(x[i],c,c_max,c_loc,delta,x_flap,chi_p,chi_TE))
			
			Node_x[upper] = xc - yt(x[i],c,t_max)*np.sin(Theta)
			Node_y[upper] = yc + yt(x[i],c,t_max)*np.cos(Theta)
			
			Node_x[lower] = xc + yt(x[i],c,t_max)*np.sin(Theta)
			Node_y[lower] = yc - yt(x[i],c,t_max)*np.cos(Theta)
			
			
#	plt.plot(Node_x,Node_y)
#	plt.axis('equal')

# 	with open('airfoil.txt', 'w') as f:
# 		f.write(str(n+1) + '\n')
# 		writer = csv.writer(f, delimiter='\t')
# 		writer.writerows(zip(Node_x,Node_y))	
	
	return Node_x, Node_y, Node_c
	

#This subroutine translates and/or rotates an airfoil based on x, y, and alpha
def translate_rotate(N,c,dx,dy,alpha,Node_x,Node_y,Node_c):
    import numpy as np
	
	#Translate airfoil in x to place origin on quarter-chord
    Node_x = Node_x - c/4.
    Node_c[0,:] = Node_c[0,:]  - c/4
	
	#Rotate the airfoil about the origin by alpha
    x_temp = Node_x
    y_temp = Node_y
    xc_temp = Node_c[0,:]
    yc_temp = Node_c[1,:]
	
    Node_x = x_temp*np.cos(-alpha) - y_temp*np.sin(-alpha)
    Node_y = x_temp*np.sin(-alpha) + y_temp*np.cos(-alpha)
    Node_c[0,:] = xc_temp[:int(N/2)]*np.cos(-alpha) - yc_temp*np.sin(-alpha)
    Node_c[1,:] = xc_temp[:int(N/2)]*np.sin(-alpha) + yc_temp*np.cos(-alpha)
	
	#Translate airfoil to desired position
    Node_x = Node_x + dx
    Node_y = Node_y + dy
    Node_c[0,:] = Node_c[0,:] + dx
    Node_c[1,:] = Node_c[1,:] + dy
    
    
    return Node_x, Node_y, Node_c


def PMatrix(l,x,x1,xc,y,y1,yc,P):
	import numpy as np
	XY = np.zeros((2,2))
	xi = (1./l)*((x1-x)*(xc-x) + (y1-y)*(yc-y))
	eta = (1./l)*(-(y1-y)*(xc-x) + (x1-x)*(yc-y))

	Phi = np.arctan2((eta*l),(eta*eta + xi*xi - xi*l))
	Psi = .5*np.log((eta*eta + xi*xi)/(eta*eta + (xi - l)**2))

	XY[0,:] = [ (x1-x)/(2.*np.pi*l*l), -(y1-y)/(2.*np.pi*l*l) ]
	XY[1,:] = [ (y1-y)/(2.*np.pi*l*l), (x1-x)/(2.*np.pi*l*l) ]

	P[0,:] = [ (l - xi)*Phi + eta*Psi, xi*Phi - eta*Psi ]
	P[1,:] = [ eta*Phi - (l-xi)*Psi - l, -eta*Phi - xi*Psi + l ]	

	P = np.dot(XY,P)	
	
	return P