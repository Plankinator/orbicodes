# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 09:38:20 2026

@author: plank
"""

import numpy as np
import math
import orbfcns as orb

def deltaTA(R1,R2,prograde):
    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)
    khat = np.array([0,0,1]) # Z-axis unit vector
    rxrz = np.dot(np.cross(R1,R2),khat)
    if prograde:
        if rxrz>=0:
            dth = math.acos(np.dot(R1,R2)/(r1*r2))
        else:
            dth = 2*math.pi - math.acos(np.dot(R1,R2)/(r1*r2)) 
    else:
        if rxrz<0:
            dth = math.acos(np.dot(R1,R2)/(r1*r2))
        else:
            dth = 2*math.pi - math.acos(np.dot(R1,R2)/(r1*r2)) 
    return dth

def calcA(R1,R2,dth):
    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)
    A = math.sin(dth) * math.sqrt(r1*r2/(1-math.cos(dth)))
    return A

def F_error(z,r1,r2,A,dt,mu=3.98600418E14):
    y = yy(z,A,r1,r2)
    F = (
        ((y/orb.CStumpff(z))**(3/2))*orb.SStumpff(z)
        + A*math.sqrt(y)
        - math.sqrt(mu)*dt
    )
    return F

def Fp_error(z,r1,r2,A,dt,mu=3.98600418E14):
    y = yy(z,A,r1,r2)
    if z!=0:
        Fp = (
            ((y/orb.CStumpff(z))**(3/2))
            * ( (orb.CStumpff(z)-(3*orb.SStumpff(z))/(2*orb.CStumpff(z)))/(2*z) + 
               (3*orb.SStumpff(z)**2)/(4*orb.CStumpff(z)))
            + (A/8)*(3*orb.SStumpff(z)/orb.CStumpff(z)*math.sqrt(y) +
                     A*math.sqrt(orb.CStumpff(z)/y))
            )
    else:
        Fp = (
            math.sqrt(2)/40 * y**(3/2) + (A/8) * (
                math.sqrt(y) + A*math.sqrt(1/(2*y))
                )
            )
    return Fp

def yy(z,A,r1,r2):
    y = r1 + r2 + A*(z*orb.SStumpff(z)-1)/(math.sqrt(orb.CStumpff(z)))
    return y

R1 = np.array([5000, 10000, 2100]) * 1000 # meters
R2 = np.array([-14600, 2500, 7000]) * 1000 # meters
dt = 3600 # sec
prograde = True

TA_rad = deltaTA(R1,R2,prograde)
TA_deg = TA_rad*180/math.pi

A = calcA(R1,R2,TA_rad)

i = 0
zi = 0;
r1 = np.linalg.norm(R1)
r2 = np.linalg.norm(R2)
Fi = F_error(zi, r1, r2, A, dt)
Fpi = Fp_error(zi, r1, r2, A, dt)
ratio = Fi/Fpi
while (abs(ratio) > 10e-8):
    i = i+1
    zi = zi - ratio
    Fi = F_error(zi, r1, r2, A, dt)
    Fpi = Fp_error(zi, r1, r2, A, dt)
    ratio = Fi/Fpi
z = zi
y = yy(z,A,r1,r2)

mu=3.98600418E14
f = 1 - y/r1
g = A*math.sqrt(y/mu)
gdot = 1 - y/r2

V1 = (1/g) * (R2 - f*R1)
V2 = (1/g) * (gdot*R2 - R1)
"""
I should format my Langrange propagator so that it is easily importable / 
callable as a function from other scripts like this one. I should also create a
similarly callable function to calculate the orbital elements based on a pair 
of position and velocity vectors.
"""