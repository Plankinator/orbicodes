# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:38:03 2026

@author: plank

This is a standalone script that simply calculates and plots an orbit based on
Lagrange coefficients in standard variables
"""
import matplotlib.pyplot as plt
import numpy as np
import math

def lagrangePropagator(R0, V0, dTh, mu=3.98600418E14):
    H = np.cross(R0,V0)
    h = np.linalg.norm(H)
    r0 = np.linalg.norm(R0)
    v0_r = np.dot(V0,R0) / r0
    r = (h**2 / mu) / (1 + (h**2/(mu*r0) - 1)*math.cos(dTh) - (h*v0_r/mu)*math.sin(dTh))
    f = 1 - (mu*r/(h**2)) * (1-math.cos(dTh))
    g = (r*r0/h)*math.sin(dTh)
    fdot = (mu/h) * math.tan(dTh/2) * ((mu/h**2)*(1-math.cos(dTh)) - (1/r0) - (1/r))
    gdot = 1 - (mu*r0/h**2)*(1-math.cos(dTh))
    R = f*R0 + g*V0
    V = fdot*R0 + gdot*V0
    return R, V

R0 = np.array([0, 1e7, 0]) # meters
V0 = np.array([6000, 0, 0]) # meters / sec

n = 100 # number of points
dTh = np.linspace(2*math.pi/n, 2*math.pi,n)
R = np.zeros((n,3))
V = np.zeros((n,3))
for i, th in enumerate(dTh):
    out = lagrangePropagator(R0, V0, th)
    R[i,:] = out[0]
    V[i,:] = out[1]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(R[:,0],R[:,1],R[:,2])
ax.scatter(0, 0, 0, marker='o', color='blue', label='Focus')
ax.scatter(R[0,0],R[0,1],R[0,2], marker='<', color='red', label='start')
plt.show()