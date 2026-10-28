# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:37:36 2026

@author: plank
"""
import numpy as np
import math
#state = np.array([-3178000,-4231000,4495000,5120,-5010,-1100])

class spacecraft:
    def __init__(self,state,epoch):
        """
        SUMMARY.

        Parameters
        ----------
        state : A 1x6 numpy array containing the spacecraft's state vector at 
        the time of epoch. State vector is [x, y, z, dx/dt, dy/dt, dz/dt].
        positions are in meters and velocities are in meters/second.
        epoch : The time at which the given state vector is true.

        Returns
        -------
        None
        """
        self.setState(state, epoch)
        
    def getPosition(self):
        return self.state[0:3]
    
    def getVelocity(self):
        return self.state[3:6]
    
    def setState(self,state,epoch):
        self.state = state
        self.epoch = epoch
        
    def getOrbitalElements(self,mu=3.98600418E14): # add a function for reporting orbital elements
        """
        SUMMARY.

        Calculates osculating orbital elements of the spacecraft based on 
        current state vector.
        -------
        returns
            h - specific angular momentum (m^2/sec)
            i - inclination (radians)
            om - right ascension of the ascending node (radians)
            e - eccentricity (dimensionless)
            w - argument of periapsis (radians)
            th - true anomaly (radians)
        """
        R = self.getPosition()
        V = self.getVelocity()
        r = np.linalg.norm(R)
        v = np.linalg.norm(V)
        vr = np.dot(R,V) / r
        H = np.cross(R,V)
        h = np.linalg.norm(H)
        i = np.arccos(H[2]/h)
        N = np.cross(np.array([0,0,1]),H)
        n = np.linalg.norm(N)
        if N[1] >= 0:
            om = np.arccos(N[0]/n)
        else:
            om = 2*math.pi - np.arccos(N[0]/n)
        E = (1/mu) * ((v**2 - mu/r)*R - r*vr*V)
        e = np.linalg.norm(E)
        if E[2] >= 0:
            w = np.arccos(np.dot(N,E)/(n*e))
        else:
            w = 2*math.pi - np.arccos(np.dot(N,E)/(n*e))
        if vr >= 0:
            th = np.arccos(np.dot(E,R)/(e*r))
        else:
            th = 2*math.pi - np.arccos(np.dot(E,R)/(e*r))
        OE = np.array([h, i, om, e, w, th])
        return OE
    
    # add options for propagating state maybe?
    # maybe add optional data for tracking remaining delta-V
    