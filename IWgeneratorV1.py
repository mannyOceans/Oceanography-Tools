 c#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:19:24 2017

This is my attempt to study idealized forms of internal waves and try to make 
my own lee wave at the study site. 

PLAN:
    - Calculate Velocity Magnitude at several depth levels (mean of 100 meter 
    bins).
    - Use mean rho from each bin
    - generate idealized internal waves using values for Velocity and rho
    - generate idealized lee wave and integrate forward in time to track propogations
    

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


# load ctd, ladcp, and bathymetry data in if not already present
if 'ladcp_dict' not in locals(): 
    ladcp_dict, ctd_dict, bathy = data_load.load_data()
    

# NOTE: some of these parameters are not needed in here, I just pasted it from
    # another script
default_params = {
        'depth_max': 4000,
        'depth_step': 100,
        'bin_size':200,
        'overlap':100,
        'm_0': 1./300., # 1/lamda limits
        'm_c': 1./10., #  1/lamda limits
        'order': 10,
        'nfft':256,
        'plot_data': 'on',
        'transect_fig_size': (6,4),
        'reference pressure': 0,
        'plot_check_adiabatic': False,
        'plot_spectrums': False,
        }



def generateBathy2d(bathy, lat, lon, params=default_params, plot_fit=False):
    """
    Generates 2D bathymetry function h(x) based on bathymetry along a transect.
    It works preloaed with a bathymetry for this transect but option to try other
    places/combos of stuff
    """
    
    # X = distance along transect
    x = gsw.distance(lon, lat)
    x = np.cumsum(x)/1000
    x = np.append(0,x)
 
    # fit curve to x and bathy
    Poly = np.polyfit(x, bathy, params['order'])
    bathyrev = np.polyval(Poly, x)
    
    if plot_fit:
        # plot bathymetry curve and points to see how well it works
        fig = plt.figure()
        plt.scatter(x, bathy)
        plt.plot(x, bathyrev)
        plt.xlabel('Distance Along Transect')
        plt.ylabel('Depth')
        plt.title('bathymetry poly fit')
        plt.gca().invert_yaxis()
        
        
    
    
    
    
    