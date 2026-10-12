# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:26:46 2026

@author: plank

orbfcns.py contains useful functions for basic orbital mechanics calculations
"""

import numpy as np
import math

def SStumpff(z):
    """
    Evaluates the S(z) Stumpff function, also known as the third order Stumpff
    function c_3(z).
    """
    if z>0:
        S = (math.sqrt(z)-math.sin(math.sqrt(z))) / (math.sqrt(z)**3)
    elif z<0:
        S = (math.sinh(math.sqrt(-z)) - math.sqrt(-z)) / (math.sqrt(-z)**3)
    else:
        S = 1/6
    return S

def CStumpff(z):
    """
    Evaluates the C(z) Stumpff function, also known as the second order Stumpff
    function c_2(z).
    """
    if z>0:
        C = (1 - math.cos(math.sqrt(z))) / z
    elif z<0:
        C = (math.cosh(math.sqrt(-z)) - 1) / (-z)
    else:
        C = 1/2
    return C

def lagrangePropagatorUniversal(R0, V0, dt, mu=3.98600418E14, tol = 1e-8):
    """
    Propagates an orbit given an initial position and velocity vector by amount
    of time dt, using lagrange coefficients with universal variables. Does not
    account for any perturbations.
    
    arguments:
        R0 = The initial position 3-vector in Earth-centered inertial coords.
            Units of [m]
        V0 = The initial velocity 3-vector in Earth-centered inertial coords.
            Units of [m/s].
        dt = The amount of time to propagate forwards through the orbit. Units
            of [s].
        mu = gravitational "GM" coefficient of the central body. Units of 
            [m3/s2]. Default value is for Earth, 3.98600418E14 m3/s2.
        tol = Maximum allowable absolute error on universal anomaly as solved
            by Newton's method iterations. Dimensionless. Default value 1e-8.
            
    returns:
        R = The final position 3-vector in Earth-centered inertial coords, dt 
            seconds after initial R0, V0 state. Units of [m].
        V = The final velocity 3-vector in Earth-centered inertial coords, dt 
            seconds after initial R0, V0 state. Units of [m/s].
    """
    r0 = np.linalg.norm(R0)
    v0 = np.linalg.norm(V0)
    v0_r = np.dot(V0,R0) / r0
    al = (2/r0) - ((v0**2)/mu) # alpha, aka 1/semimajoraxis
    # find change in Universal Anomaly, X, starting with guess X0
    X0 = math.sqrt(mu)*abs(al)*dt # initial guess for Universal Anomaly
    i = 0
    Xi = X0
    z0 = al*(X0**2) # define parameter z for convenience
    f0 = ( # error function
        (r0*v0_r/math.sqrt(mu))*(X0**2)*CStumpff(z0) 
        + (1-al*r0)*(X0**3)*SStumpff(z0) + r0*X0 - math.sqrt(mu)*dt
        )
    fp0 = ( # derivative of error function
        (r0*v0_r/math.sqrt(mu))*(X0)*(1-z0*SStumpff(z0)) 
        + (1-al*r0)*(X0**2)*CStumpff(z0) + r0
        )
    ratio = f0/fp0 # newton's method ratio, Xi+1 = Xi - fi/fpi
    while abs(ratio) > tol and i < 100: #iterate Xi up to 100x to meet tol
        i = i+1
        Xi = Xi-ratio
        zi = al*(Xi**2)
        fi = ( # error function
            (r0*v0_r/math.sqrt(mu))*(Xi**2)*CStumpff(zi) 
            + (1-al*r0)*(Xi**3)*SStumpff(zi) + r0*Xi - math.sqrt(mu)*dt
            )
        fpi = ( # derivative of error function
            (r0*v0_r/math.sqrt(mu))*(Xi)*(1-zi*SStumpff(zi)) 
            + (1-al*r0)*(Xi**2)*CStumpff(zi) + r0
            )
        ratio = fi/fpi # newton's method ratio, Xi+1 = Xi - fi/fpi
    # save final iteration values as simply X and z:
    X = Xi
    z = al*X**2
    # calculate lagrange coefficients f, g
    f = 1 - (X**2) * CStumpff(z) / r0
    g = dt - X**3 * SStumpff(z) / math.sqrt(mu)
    # propagate position vector R
    R = f*R0 + g*V0
    r = np.linalg.norm(R)
    # calculate lagrage coefficients fdot, gdot
    fdot = (math.sqrt(mu)/(r*r0)) * (al*(X**3)*SStumpff(z) - X)
    gdot = 1 - (X**2)*CStumpff(z)/r
    # propagate velocity vector V
    V = fdot*R0 + gdot*V0
    return R, V

def setAxesEqual(ax):
    """
    sets all three axes of a 3D plot to be of equal size to avoid aspect
    ratio distortion.
    argument:
        ax = a 3D Axes object, like the one created by:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
    """
    xmin, xmax = ax.get_xlim3d()
    ymin, ymax = ax.get_ylim3d()
    zmin, zmax = ax.get_zlim3d()
    axmax = max(xmax,ymax,zmax)
    axmin = min(xmin,ymin,zmin)
    range = [axmin,axmax]
    ax.set_xlim3d(range)
    ax.set_ylim3d(range)
    ax.set_zlim3d(range)
    ax.set_box_aspect([1, 1, 1])
    return

def getPeriod(R0, V0, mu=3.98600418E14):
    """
    Calculates the period of an orbit based on an initial position and
    velocity vector.
    
    arguments:
        R0 = The initial position 3-vector in Earth-centered inertial coords.
            Units of [m]
        V0 = The initial velocity 3-vector in Earth-centered inertial coords.
            Units of [m/s].
        mu = gravitational "GM" coefficient of the central body. Units of 
            [m3/s2]. Default value is for Earth, 3.98600418E14 m3/s2.    
    returns:
        T = The period of the elliptical orbit. Units of [s]. If the orbit is 
            parabolic or hyperbolic, this will return -1.
    """
    r0 = np.linalg.norm(R0)
    v0 = np.linalg.norm(V0)
    a = 1 / ((2/r0) - ((v0**2)/mu))
    if a>0:
        T = 2*math.pi*math.sqrt(a**3 / mu)
    else:
        T = -1
    return T
