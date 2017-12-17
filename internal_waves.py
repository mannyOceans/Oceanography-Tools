#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:47:03 2017

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




# load ctd, ladcp, and bathymetry data in if not already present
if 'ladcp' not in locals(): 
    ladcp, ctd, bathy = data_load.load_data()
    
    
def internal_wave_energy(ctd):
    
    
    
    
    
    