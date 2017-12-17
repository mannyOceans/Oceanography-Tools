#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 20:31:47 2017

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
import lee_waves as lw





# load ctd, ladcp, and bathymetry data in if not already present
if 'ladcp' not in locals(): 
    ladcp, ctd, bathy = data_load.load_data()
    U, V, p_ladcp = oc.loadLADCP(ladcp)
    S, T, p, lat, lon = oc.loadCTD(ctd)

bathy_file = 'bathy.nc'
    
default_params = {
        'depths': [1000, 2000, 2500, 3500]
        }

def profilePlot(data, z, new_fig=True, axis=None, params=default_params):
    """ 
    Quick function for plotting vertical profiles so I dont have to copy 
    and past a bunch of shit
    """
    z = np.squeeze(z)
    if new_fig:
        fig = plt.figure()
        for i in range(data.shape[1]):
            plt.plot(data[:,i], z)
        
        plt.gca().invert_yaxis()
        
        return fig
        
    else:
        for i in range(data.shape[1]):
            axis.plot(data[:,i], z)
        
        axis.invert_yaxis()
        
        
    
def bandpassFiltering(ladcp, ctd, bathy, stns=np.arange(5,13)):
    """
    Attempt at using band pass filtering to get out the signal that i want
    """
    U, V, p_ladcp = oc.loadLADCP(ladcp)
    
    U = U[:,stns]
    Umx, Ukx, Uspectra, Ubandpass,\
        UfiltShift, Upower, UmxShift, UpowFilt = oc.verticalBandPass(U, p_ladcp, 100, 250)
    
    fig = plt.figure()
    plt.plot(UmxShift, Upower[:,0], linewidth=1)
    
    fig1 = plt.figure()
    plt.plot(UmxShift, UpowFilt[:,0], linewidth=1)
    
    Ufiltered = scipy.fftpack.ifftshift(UfiltShift, axes=0)
    Ufiltrev = scipy.ifft(Ufiltered, axis=0)
    


def filtered_velocity_profiles(ladcp, ctd, bathy, stns=np.arange(5,13), plots=False):
    """
    Run filtering of various types on vertical velocity profiles to figure out 
    which filtering works best and try and identify characteristic lee wave
    features. stations indicates running function on only stations of interest
    """
    
    
    U, V, z = oc.loadLADCP(ladcp)
    
    
#    maxD = 4000
#    U, z = oc.depthTrim(U, p_ladcp, maxD)
#    V = oc.depthTrim(V, p_ladcp, maxD)[0]

    Upoly = []
    
    for cast in U.T:
        fitrev = oc.vert_polyFit(cast, z, 100)
        Upoly.append(fitrev)
        
    Upoly = np.vstack(Upoly).T
    Umasked = U - Upoly
    
    Vpoly = []
    
    for cast in V.T:
        fitrev = oc.vert_polyFit(cast, z, 100)
        Vpoly.append(fitrev)
        
    Vpoly = np.vstack(Vpoly).T
    Vmasked = V - Vpoly
    
    # apply smoothing to detrended data
    Ufinal = np.vstack([oc.verticalBoxFilter1(Ui, z) for Ui in Umasked.T]).T
    Vfinal = np.vstack([oc.verticalBoxFilter1(Vi, z) for Vi in Vmasked.T]).T
    M = np.sqrt(Ufinal**2 + Vfinal**2)
    
    maskD = np.squeeze(z <=4000)
    z = z[maskD]
    if plots:
        for i in range(U.shape[1]):
            fig = plt.figure()
            plt.plot(U[maskD,i], z, label='original', linewidth=1)
            plt.plot(Upoly[maskD,i], z, label='polyfit', linewidth=1)
            plt.plot(Umasked[maskD,i], z, label='residual', linewidth=1)
            plt.plot(Ufinal[maskD,i], z, label='detrend + smoothing', linewidth=1)
            plt.title('Smoothing Validation Plot (U) - Station ' + str(i))
            plt.xlabel('m/s')
            plt.ylabel('Pressure (dB)')
            plt.ylim(0,4000)
            plt.gca().set_aspect('auto')
            plt.gca().invert_yaxis()
            plt.legend()
            plt.savefig('figures/validations/velocity_smoothing/U_station_'\
                        + str(i) + '.png', bbox_inches='tight', dpi=400)
           
            
            fig = plt.figure()
            plt.plot(V[maskD,i], z, label='original', linewidth=1)
            plt.plot(Vpoly[maskD,i], z, label='polyfit', linewidth=1)
            plt.plot(Vmasked[maskD,i], z, label='residual', linewidth=1)
            plt.plot(Vfinal[maskD,i], z, label='detrend + smoothing', linewidth=1)
            plt.title('Smoothing Validation Plot (V) - Station ' + str(i))
            plt.xlabel('m/s')
            plt.ylabel('Pressure (dB)')
            plt.ylim(0,4000)
            plt.gca().invert_yaxis()
            plt.legend()
            plt.savefig('figures/validations/velocity_smoothing/V_station_'\
                        + str(i) + '.png', bbox_inches='tight', dpi=400)
            
            fig = plt.figure()
            plt.plot(M[maskD,i], z, label='Mag', linewidth=1)
            plt.title('Smoothing Validation Plot (Magnitude) - Station ' + str(i))
            plt.xlabel('m/s')
            plt.ylabel('Pressure (dB)')
            plt.ylim(0,4000)
            plt.gca().invert_yaxis()
            plt.legend()
            plt.savefig('figures/validations/velocity_smoothing/Mag_station_'\
                        + str(i) + '.png', bbox_inches='tight', dpi=400)
            
            plt.close('all')
    
    
    
    return Vmasked, Umasked, M

def filteredCTD(ladcp, ctd, bathy, stns=np.arange(5,13), plots=False):
    """ 
    Run polynomial then smoothing filter on ctd profiles to compare with
    velocity profiles
    """
    
    U, V, z = oc.loadLADCP(ladcp)
    S, T, p_ctd, lat, lon = oc.loadCTD(ctd)
    
    rho = oc.rhoFromCTD(S, T, p_ctd, lon, lat)
    
    rhoPoly = []
    for cast in rho.T:
        rho_i = oc.vert_polyFit2(cast, p_ctd[:,0], 100)
        rhoPoly.append(rho_i)
    
    rhoPoly = np.vstack(rhoPoly).T
    rhoFilt = rho - rhoPoly
    
    
    maskD = np.squeeze(p_ctd[:,0] <=4000)

    if plots:
        for i in range(rho.shape[1]):
            fig, (ax1, ax2) = plt.subplots(1,2)
            fig.suptitle('Smoothing Validation Plot (U) - Station ' + str(i))
            ax1.plot(rho[maskD,i], p_ctd[maskD,0], label='original', linewidth=1)
            ax1.plot(rhoPoly[maskD,i], p_ctd[maskD,0], label='polyfit', linewidth=1)
            ax1.legend()
            ax2.plot(rhoFilt[maskD,i], p_ctd[maskD,0], label='residual', linewidth=1)
            plt.legend()
            ax1.set_xlabel('kg/m^3')
            ax1.set_ylabel('Pressure (dB)')
            ax1.set_ylim(0,4000)
            ax1.invert_yaxis()
            ax2.set_xlabel('kg/m^3')
            ax2.set_ylim(0,4000)
            ax2.invert_yaxis()
            plt.legend()
            plt.savefig('figures/validations/ctd_smoothing/station_'\
                        + str(i) + '.png', bbox_inches='tight', dpi=400)
            plt.close()
            
           
            
    
    

    

def depthFlowPlots(ladcp, ctd, bathy, bathy_file, dRange=2500):
    """
    Bathymetry flow vector plots for bottom 500 meters average
    """
    
    
    U, V, z = oc.loadLADCP(ladcp)
    S, T, p, lat, lon = oc.loadCTD(ctd)
    dz = np.nanmean(np.gradient(np.squeeze(z)))
    
    
        
    dist = gsw.distance(lon, lat)
    dist = np.cumsum(dist)/1000
    dist = np.append(0,dist)
 
    
    
    Umean, Vmean = oc.depthAvgFlow(U, V, dz, depthRange=dRange)
    Magmean = np.sqrt(Umean**2 + Vmean**2)
    np.savetxt('processed_data/Umean_'+str(dRange)+'m.txt', Umean)
    np.savetxt('processed_data/Vmean_'+str(dRange)+'m.txt', Vmean)
    np.savetxt('processed_data/Magmean_'+str(dRange)+'m.txt', Magmean)
    
    Ufig = oc.transect_flow_vector(Umean, Vmean, lat,\
                                   lon, bathy_file,\
                                   title='avg. flow bottom ' + str(dRange) + ' meters' )
    plt.savefig('figures/depth_averaged_flow_vector_bottom_' +str(dRange) + 'm.png',\
                                    bbox_inches='tight', dpi=400)
    plt.close()
    
    
    fig1 = plt.figure()
    plt.plot(dist, Umean, label= 'U')
    plt.plot(dist, Vmean, label= 'V')
    plt.plot(dist, Magmean, label= 'Mag')
    plt.xlabel('Distance along Transect')
    plt.ylabel(' m/s' )
    plt.savefig('figures/depth_averaged_flow_bottom_' +str(dRange) + 'm.png',\
                                    bbox_inches='tight', dpi=400)
    plt.close()
    


def idealVsMeasured(ladcp, ctd, bathy):
    
    U, V, z = oc.loadLADCP(ladcp)
    S, T, p, lat, lon = oc.loadCTD(ctd)
    dz = np.nanmean(np.gradient(np.squeeze(z)))
    
    
        
    dist = gsw.distance(lon, lat)
    dist = np.cumsum(dist)/1000
    dist = np.append(0,dist)
    
    Ufilt, Vfilt, MagFilt = filtered_velocity_profiles(ladcp,\
                                    ctd, bathy,\
                                    stns=np.arange(5,13),\
                                    plots=False)
    np.savetxt('processed_data/Ufiltered.txt', Ufilt)
    np.savetxt('processed_data/Vfiltered.txt', Vfilt)
    np.savetxt('processed_data/Magfiltered.txt', MagFilt)
   
    mx = 150
    mz = 200
    omega = 1/150
    H = 150
    U = 0.03
    
    
    u_lee, Ux, Uz = lw.u_lee_zx(mx, mz, U, H, omega, z0=-4000)
    

    
    # choose distances that match the profiles distance away from the 
    # theoretical lee wave generation site and plot the vertical profiles of 
    # velocity magnitude to compare -- because of the waves periodicity, it 
    # wil make more sense to choose a range of X values that contain the full 
    # and plot the  profiles through the period
    
    dx = np.nanmean(np.gradient(Ux, axis=1))
    x_window = 200
    steps = int(np.ceil(x_window/dx))
    mask = range(steps,2*steps)
    # extract a portion of the grid with x width = window but start at least
    # "steps" to catch a full period
 
    

    diffs = np.empty(0)
    for i in range(MagFilt.shape[1]):
        lwP = scipy.interpolate.interp1d(Uz[:,steps], U_lee_rev[:,0], fill_value='extrapolate')
        lwRev = lwP(-1*p_ladcp)
        Mag_rev = MagFilt[:,i] - np.nanmean(MagFilt[:,i])
        diffs = np.append(diffs,np.nansum(np.abs(Mag_rev - lwRev)))
    



    

  
   
