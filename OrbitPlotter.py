# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:47:41 2026

@author: plank

This script introduces two functions, one for plotting an orbit using orbfcns
lagrangePropagator(), and the other for plotting a sphere representing the
Earth. When run, it uses these functions to plot an orbit determined by R0 and 
V0.
"""
import matplotlib.pyplot as plt
import numpy as np
import orbfcns as orb

def plotLagrange (ax, R0, V0):
    """
    function description
    """
    T = orb.getPeriod(R0,V0)
    n = 100 # number of points
    if T>0: # closed orbit
        stoptime = T # sec
        times = np.linspace(0,stoptime,n)
    else: # parabolic / hyperbolic trajectory
        stoptime = 4*np.linalg.norm(R0)/np.linalg.norm(V0) # sec
        times = np.linspace(-stoptime,stoptime,n)
    R = np.zeros((n,3))
    V = np.zeros((n,3))
    for i, t in enumerate(times):
        R[i,:], V[i,:] = orb.lagrangePropagator(R0, V0, t)
    ax.plot(R[:,0],R[:,1],R[:,2],color='red')
    ax.scatter(R0[0],R0[1],R0[2], marker='<', color='orange', label='End')
    return ax

def plotEarth(ax):
    """
    plot Earth as a sphere
    """
    u = np.linspace(0, 2 * np.pi, 100) # longitude, from zero to 2 pi
    v = np.linspace(0, np.pi, 100) # latitude, from zero to pi
    R_earth = 6371000 # meters
    # x(lon,lat) = R*cos(lon)*sin(lat)
    x = R_earth*np.outer(np.cos(u), np.sin(v))
    # y(lon,lat) = R*sin(lon)*sin(lat)
    y = R_earth*np.outer(np.sin(u), np.sin(v))
    # z(lat)  R*cos(lat), times array of ones to get 100x100 dimensions
    z = R_earth*np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='#2E6B8A')
    return ax

R0 = np.array([0, 8567328, 0]) # meters
V0 = np.array([7000, 100, 100]) # meters / sec
fig = plt.figure()
ax = fig.add_subplot(projection='3d',computed_zorder=False)
ax = plotEarth(ax)
ax = plotLagrange(ax,R0,V0)
ax = orb.setAxesEqual(ax)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
"""
There is a problem with matplotlib here where the Earth and the Orbit can't
seem to dynamically overlap, either the Earth is always rendered closer to the
camera than the orbit or vice-versa. Need to look into alternative plotting
methods.
"""