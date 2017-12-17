#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:19:17 2017

@author: manishdevana
"""


import numpy as np
import scipy.signal as sig
import scipy
import seawater as sw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import data_load
import gsw
import cmocean
import oceans as oc
import IW_functions as iw



default_params = {
        'kx' : 2*np.pi/300,
        'kz' : 2*np.pi/200,
        'U' : 0.1,
        'H' : 100,
        'omega' : 1/300,
        'z' : -4000,
        't' : 0,
        }

kx = 2*np.pi/300
kz = 2*np.pi/200
U = 0.1
H = 100
omega = 1/300
x0=0
t0=0
z0=3000



def u(x, z, kx, kz, U, H, omega, t):
    """
    Lee wave function for u (horizontally aligned u and v)
    """
    
    u = kz*U*H*np.sin(kx*x + kz*z + omega*t)
    
    return u

def rho_prime(x, z, kx, kz, U, H, omega, t):
    """ 
    lee wave function for density fluctuations
    """
    
    

                      


def u_lee_zx(mx, mz, U, H, omega, x0=0, t0=0, z0=-3000):
    """
    Solves lee wave equation for u(z,t)
    
    """
    kx = 2*np.pi/mx
    kz = 2*np.pi/mz
    x = x0
    t = t0
    z = z0
    total_time = np.arange(t, 3600*5, 30)
    u0 = u(x, z, kx, kz, U, H, omega, t)
    u_all = [u0]
    z = np.linspace(z0, -10, num=len(total_time))
    x = np.linspace(x0, 1e3, num=len(total_time))

    tgrid, zgrid = np.meshgrid(total_time, z)
    xgrid, zgrid = np.meshgrid(x, z)
    
    u_zx = u(xgrid, zgrid, kx, kz, U, H, omega, tgrid)
    
    
    return u_zx, xgrid, zgrid

    


















































def u_time_forward(kx, kz, U, H, omega, x0=0, t0=0, z0=-3000):
    
    x = x0
    t = t0
    z = z0
    dt = 30
    total_time = np.arange(t, 3600*5, dt)
    z = np.linspace(z0, -10, num=len(total_time))
    x = np.linspace(x0, 1e3, num=len(total_time))
    
    # simple sinusoidal hill
    phaseShift  = 100*np.pi*omega
    Hx = H*np.sin(2*np.pi*x/(2/omega) - phaseShift) 
    mask = np.logical_or(x>300+100, x<100) 
    Hx[mask] = 0
    
    


    tgrid, zgrid = np.meshgrid(total_time, z)
    xgrid, zgrid = np.meshgrid(x, z)
    tgrid = tgrid.T
    
    u0 = u(x[0], zgrid[0,:], kx, kz, U, Hx[0], omega, total_time[0])
    Uall = [u0]
    for i in range(1,len(total_time)):
        u_i = u0 + dt*u0
        Uall.append(u_i)
        u0 = u(xgrid[:,i], zgrid[:,i], kx, kz, U, Hx[i], omega, total_time[i])
        
        
        
    U2 = np.vstack(Uall)



class internalWave:
    """ 
    Object defining the internal wave
    """
    g = 9.8
    time_step = 5 
    


    
    def __init__(self, kx, kz, U, H, omega):
        """
        Generate initial values for variables given mean flows and wavenumbers
        
        Parameters
        ----------
        kx : horizontal wavenumber
        kz : vertical wavenumber
        U : mean flow speed
        H : Amplitude of simple sinusoidal topography
        omega : frequency
        
        Returns
        -------
        
        """
        self.kx = kx
        self.kz = kz
        self.U = U
        self.H = H
        self.omega = omega
        
    
    def u(self, x, z, t):
        """
        try to run the model forward in time
        """
        
        return self.kz*self.U*self.H*np.sin(self.kx*x + self.kz*z - self.omega*t)
        
        
    
    

    
    
    
    
    
    
    























