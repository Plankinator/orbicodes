# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:47:41 2026

@author: plank
"""
import matplotlib.pyplot as plt
import numpy as np
import math

def lagrangePropagatorUniversal(R0, V0, dt, mu=3.98600418E14, tol = 10^-8):
    r0 = np.linalg.norm(R0)
    v0 = np.linalg.norm(V0)
    v0_r = np.dot(V0,R0) / r0
    al = (2/r0) - ((v0**2)/mu) # alpha, aka 1/semimajoraxis
    
    X, i, ratio = findUniversalAnomaly(dt,r0,v0_r,al)
    z = al*X**2
    
    f = 1 - (X**2) * CStumpff(z) / r0
    g = dt - X**3 * SStumpff(z) / math.sqrt(mu)
    R = f*R0 + g*V0
    r = np.linalg.norm(R)

    fdot = (math.sqrt(mu)/(r*r0)) * (al*(X**3)*SStumpff(z) - X)
    gdot = 1 - (X**2)*CStumpff(z)/r
    V = fdot*R0 + gdot*V0
    return R, V, i, ratio

def findUniversalAnomaly(dt,r0,v0_r,al,mu=3.98600418E14,tol = 1e-8):
    i = 0
    X0 = math.sqrt(mu)*abs(al)*dt
    Xi = X0
    z0 = al*(X0**2)
    f0 = (r0*v0_r/math.sqrt(mu))*(X0**2)*CStumpff(z0) + (1-al*r0)*(X0**3)*SStumpff(z0) + r0*X0 - math.sqrt(mu)*dt
    fp0 = (r0*v0_r/math.sqrt(mu))*(X0)*(1-z0*SStumpff(z0)) + (1-al*r0)*(X0**2)*CStumpff(z0) + r0
    ratio = f0/fp0
    while abs(ratio) > tol and i < 100:
        i = i+1
        Xi = Xi-ratio
        zi = al*(Xi**2)
        fi = (r0*v0_r/math.sqrt(mu))*(Xi**2)*CStumpff(zi) + (1-al*r0)*(Xi**3)*SStumpff(zi) + r0*Xi - math.sqrt(mu)*dt
        fpi = (r0*v0_r/math.sqrt(mu))*(Xi)*(1-zi*SStumpff(zi)) + (1-al*r0)*(Xi**2)*CStumpff(zi) + r0
        ratio = fi/fpi
    return Xi, i, ratio

def SStumpff(z):
    if z>0:
        S = (math.sqrt(z)-math.sin(math.sqrt(z))) / (math.sqrt(z)**3)
    elif z<0:
        S = (math.sinh(math.sqrt(-z)) - math.sqrt(-z)) / (math.sqrt(-z)**3)
    else:
        S = 1/6
    return S

def CStumpff(z):
    if z>0:
        C = (1 - math.cos(math.sqrt(z))) / z
    elif z<0:
        C = (math.cosh(math.sqrt(-z)) - 1) / (-z)
    else:
        C = 1/2
    return C

R0 = np.array([0, 1e7, 0]) # meters
V0 = np.array([6000, 0, 0]) # meters / sec
dt = 0 # seconds
R,V,i,ratio = lagrangePropagatorUniversal(R0, V0, dt)

n = 100 # number of points
stoptime = 3600 # sec
times = np.linspace(0,stoptime,n)
R = np.zeros((n,3))
V = np.zeros((n,3))
for i, t in enumerate(times):
    R[i,:], V[i,:], i, ratio = lagrangePropagatorUniversal(R0, V0, t)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(R[:,0],R[:,1],R[:,2])
ax.scatter(0, 0, 0, marker='o', color='blue', label='Focus')
ax.scatter(R[-1,0],R[-1,1],R[-1,2], marker='<', color='red', label='End')
plt.show()
