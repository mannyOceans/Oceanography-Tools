#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 21:43:27 2017

@author: manishdevana
"""

import IW_analysis_rev1 as iw
#ladcp_dict, ctd_dict, bathy = data_load.load_data()

#    #---------------------------------------------------------------------
#    # Calculate Shear Spectrum and Integrated Power of Shear Spectrum
#    # U and V components are split into Real and Complex, respectively
#    shear = ladcp_dict['shear']
#    Ushear = np.real(shear)
#    Vshear = np.imag(shear)
#    
#    # Magnitude of shear (Why is this better than absolute function??)
#    Shear2 = (Ushear**2 + Vshear**2)
#    
#    # Bin Shear
#    Shear = bin_data(Shear2, z_ladcp)[0]
#    
#    # Dz
#    dz = np.mean(np.diff(np.squeeze(z_ladcp)))
#    
#    # Shear Spectrum and Integration, with wavenumber and wavelength axes
#    ShearInt, ShearSpec, mxShear, kxShear = PowerSpecAndInt(Shear, dz)
#    
    #---------------------------------------------------------------------
    


iw.measurementPlotting(ctd_dict, ladcp_dict)

