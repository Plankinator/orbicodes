# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 08:46:34 2026

@author: plank
"""

import matplotlib.pyplot as plt
import numpy as np
import orbfcns as orb


R0 = np.array([0, 8567328, 0]) # meters
V0 = np.array([7000, 100, 100]) # meters / sec
dV = np.array([300, 0, 480]) # meters / sec
T = orb.getPeriod(R0,V0)

n = 100 # number of points
if T>0:
    stoptime = T # sec
else:
    stoptime = 100000 # sec
times = np.linspace(0,stoptime,n)
R = np.zeros((n,3))
V = np.zeros((n,3))
for i, t in enumerate(times):
    R[i,:], V[i,:] = orb.lagrangePropagatorUniversal(R0, V0, t)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(R[:,0],R[:,1],R[:,2])
ax.scatter(R[-1,0],R[-1,1],R[-1,2], marker='<', color='red', label='End')

T2 = orb.getPeriod(R0,V0+dV)
n = 100 # number of points
if T2>0:
    stoptime = T2 # sec
else:
    stoptime = 100000 # sec
times = np.linspace(0,stoptime,n)
R2 = np.zeros((n,3))
V2 = np.zeros((n,3))
for i, t in enumerate(times):
    R2[i,:], V2[i,:] = orb.lagrangePropagatorUniversal(R0, V0+dV, t)
ax.plot(R2[:,0],R2[:,1],R2[:,2])
#ax.scatter(R[-1,0],R[-1,1],R[-1,2], marker='<', color='red', label='End')

# plot Earth as a sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
R_earth = 6371000
x = R_earth*np.outer(np.cos(u), np.sin(v))
y = R_earth*np.outer(np.sin(u), np.sin(v))
z = R_earth*np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(x, y, z, cmap='ocean', edgecolor='none', alpha=1)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
orb.setAxesEqual((ax))
plt.show()
