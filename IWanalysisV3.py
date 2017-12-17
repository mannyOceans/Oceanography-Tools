#!/usr/bin/env python3
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
import oceans as oc
import IW_functions as iw




# load ctd, ladcp, and bathymetry data in if not already present
if 'ladcp' not in locals(): 
    ladcp, ctd, bathy = data_load.load_data()
    
#depths = [1000, 2000, 2500, 3000, 3500]

# NOTE: some of these parameters are not needed in here, I just pasted it from
    # another script
default_params = {
        'depth_max': 4000,
        'depth_step': 100,
        'bin_size':200,
        'overlap':100,
        'm_0': 1./300., # 1/lamda limits
        'm_c': 1./10., #  1/lamda limits
        'order': 1,
        'nfft':256,
        'plot_data': 'on',
        'transect_fig_size': (6,4),
        'reference pressure': 0,
        'plot_check_adiabatic': False,
        'plot_spectrums': False,
        }



def velocityInLine(U, V, lat, lon):
    """
    Returns velocity in line with transect
    """
    
    # Angle between velocity vector and transect
#    theta = np.arctan((lat[-1]-lat[0])/(lon[-1]-lon[0])) - \
#                    np.arctan(U/V)
    
    theta = np.full_like(U, np.nan)                
    for i in range(len(lat)):
        if i != 0:
            theta[:,i] = np.arctan((lat[i]-lat[i-1])/(lon[i]-lon[i-1])) - \
                                np.arctan(U[:,i]/V[:,i])
        else:
            theta[:,i] = np.arctan((lat[i]-lat[i-1])/(lon[i]-lon[i-1])) - \
                                np.arctan(U[:,i]/V[:,i])
    
    # Velocity Magnitude
    Umag = np.sqrt(U**2 + V**2)
    
    # Velocity in direction of transect
    X = Umag*np.cos(theta)
    
    return X
    
    
    

def velocityAnalysis(params=default_params):
    """
    Generate idealised internal waves with data
    """
     
    # load data
    U = ladcp_dict['u']
    V = ladcp_dict['v']
    W = np.zeros_like(U)
    lat = ladcp_dict['lat']
    lon = ladcp_dict['lon']
    dist = gsw.distance(lon.T, lat.T)
    dist = np.cumsum(dist)/1000
    dist = np.append(0,dist)
    z = ladcp_dict['p_grid']
    
    Ubin, binidx = bin_data(U, z)
    Vbin = bin_data(V, z)[0]
    
    # Bin mean velocities
    Umean = np.vstack([np.nanmean(station, axis=1) for station in Ubin]).T
    Vmean = np.vstack([np.nanmean(station, axis=1) for station in Vbin]).T
    
    # Bin mean mags
    Xmean = velocityInLine(Umean, Vmean, lat, lon)
    Vx = np.zeros_like(Xmean) # empty V matrix for plotting (no v since all in line with transect)
    
    # depth grid for plotting
    z2 = np.nanmean(bin_data(z, z)[0][0], axis=1)
    
    # Plot check X mean velocities
    fig1 = plt.figure(figsize=(12,10))
    
    
    plt.pcolor(dist, z2, Xmean, cmap=cmocean.cm.balance)
    plt.colorbar(label='$m/s$')
    plt.quiver(dist, z2, Xmean, Vx, scale=5, color='lime')
    plt.ylim(0, 4000)
    plt.fill_between(dist, bathy, 4000, color = '#B4B4B4')
    plt.gca().invert_yaxis()
    plt.title('Velocity in line with transect')
    plt.savefig('figures/velocity_inline.png', bbox_inches='tight', dpi=300)
    
    
    
    
    
def transectContour(axis, data, dist, z, bathy, title, vmin=-.5, vmax=.5, colorbar=False):
    """ 
    Creates transect plot with bathymetry of any data grid
    """
    
    c = axis.contourf(dist, z, data)
    axis.set_ylim(0, 4000)
    if colorbar:
        plt.colorbar(c, ax=axis)
    axis.fill_between(dist, bathy, 4000, color = '#B4B4B4')
    axis.invert_yaxis()
    axis.set_xlabel('Distance along transect (km)')
    axis.set_ylabel('Pressure(db)')
    axis.set_title(title)
    for i, distIn in enumerate(dist):
        axis.annotate(str(i+1), (distIn, 200))
        
    
    return c
    





def velPlot1(params=default_params):
    """
    Calculate the mean velocity at different points in the direction of the
    transect. The goal is to isolate the flow in the direction of hypothesized 
    lee wave propogation. 
    """
    
    # load data
    U = ladcp_dict['u']
    V = ladcp_dict['v']
    W = np.zeros_like(U)
    lat = ladcp_dict['lat']
    lon = ladcp_dict['lon']
    dist = gsw.distance(lon.T, lat.T)
    dist = np.cumsum(dist)/1000
    dist = np.append(0,dist)
    z = ladcp_dict['p_grid']
    
    U_depthMean = np.nanmean(U, axis=0)
    V_depthMean = np.nanmean(V, axis=0)
    
    # Test directions of flow 
    angles = np.arctan(U/V)
    
    # Create depth intervals
    levels = np.arange(100, params['depth_max'], params['depth_step'])
    
    # Velocity magnitude
    Umag= np.sqrt(U**2 + V**2)
    
    # Theta in radians
    theta = np.arctan((lat[-1]-lat[0])/(lon[-1]-lon[0])) - \
                    np.arctan(U/V)
            
    # X = velocity in direction of transect
    X = Umag*np.cos(theta)
    Vx = np.zeros_like(X)

    fig1 = plt.figure(figsize=(12,10))
    
    plt.quiver(dist, z, X, Vx, scale=5)
    plt.ylim(0, 4000)
    plt.fill_between(dist, bathy, 4000, color = '#B4B4B4')
    plt.gca().invert_yaxis()
    
    fig3 = plt.figure(figsize=(12,10))
    
    plt.quiver(dist, z, U, V, scale=4, color='pink')
    plt.ylim(0, 4000)
    plt.fill_between(dist, bathy, 4000, color = '#B4B4B4')
    plt.gca().invert_yaxis()
    
    
    
    
    fig2 = plt.figure()
    plt.scatter(lon, lat, c='pink', marker='x')
    plt.quiver(lon, lat, U_depthMean, V_depthMean, scale=1)
    plt.title('Depth averaged velocities')
    
   
    fig4 = plt.figure()
    plt.pcolor(dist, z, U)
    plt.ylim(0, 4000)
    plt.colorbar()
    plt.fill_between(dist, bathy, 4000, color = '#B4B4B4')
    plt.gca().invert_yaxis()
    plt.title('U')
    
    fig5 = plt.figure()
    plt.pcolor(dist, z, V)
    plt.ylim(0, 4000)
    plt.colorbar()
    plt.fill_between(dist, bathy, 4000, color = '#B4B4B4')
    plt.gca().invert_yaxis()
    plt.title('V')
    
    fig6 = plt.figure()
    plt.pcolor(dist, z, X)
    plt.ylim(0, 4000)
    plt.colorbar()
    plt.fill_between(dist, bathy, 4000, color = '#B4B4B4')
    plt.gca().invert_yaxis()
    plt.title('Velocity in line with transect (sorta)')
    
    
    fig7 = plt.figure()
    ax = fig7.gca(projection='3d')
    plt.quiver(np.squeeze(lon), np.squeeze(lat), z, U, V, W, length=0.1)
    plt.gca().invert_zaxis()

   

def velocityMinusGeostrophic(ladcp, ctd, depths,\
                             bathy_file='bathy.nc', plots=True):
    """
    Remove geostrophic flow from velocity to see residuals
    """
    
    
    U, V, p_ladcp = oc.loadLADCP(ladcp)
    S, T, p_ctd, lat, lon = oc.loadCTD(ctd)
    
    # For transect plots
    dist = gsw.distance(lon, lat)
    dist = np.cumsum(dist)/1000
    dist = np.append(0,dist)
    
    # Calculate geostrophic flow perpendicular to the transect
    Ugeo, Vgeo, geoMag, distGeo = oc.geoFlow(S, T, p_ctd, lat, lon)
    
    # now interpolate the geostrophic flow and ladcp measured flow onto the 
    # same pressure grid
    Ugeo = oc.matchGrids(Ugeo, p_ctd[:,0], p_ladcp)
    Vgeo = oc.matchGrids(Vgeo, p_ctd[:,0], p_ladcp)
    
    # subtract geostrophic components from measured to get residuals
    Ures = U[:,1:] - Ugeo
    Vres = V[:,1:] - Vgeo
    
    
#    # Create vector plots of residuals along transect
#    for depth in depths:
#        oc.VectorOverheadPlot(Ures, Vres, lat,\
#                              lon, p_ladcp[:,0],\
#                              depth, bathy_file)
    if plots:
        dataIn = [Ugeo, Vgeo, U, V]
        distIn = [dist[1:], dist[1:], dist, dist]
        bathyIn = [bathy[1:], bathy[1:], bathy, bathy]
        title = ['U geostrophic', 'V geostrophic', 'U ladcp', 'V ladcp']
        
        fig, ax = plt.subplots(2,2, figsize=(13.5,7.1))
        for i in range(len(dataIn)):
            c = transectContour(ax.ravel()[i], dataIn[i],\
                            distIn[i], np.squeeze(p_ladcp),\
                            bathyIn[i],\
                            title[i], colorbar=True)
            
        plt.tight_layout()
        plt.savefig('figures/geostrophic_flow/geostrophyVsmeasured.png',\
                    bbox_inches='tight', dpi=400)
        plt.close()
        
        fig, ax = plt.subplots(1, 2, figsize=(13.5,7.1))
        transectContour(ax.ravel()[0],\
                        Ures, dist[1:],
                        np.squeeze(p_ladcp),\
                        bathy[1:],
                        'U residual Flow',
                        colorbar=True)
        transectContour(ax.ravel()[1],\
                        Vres, dist[1:],
                        np.squeeze(p_ladcp),\
                        bathy[1:],
                        'V residual Flow',
                        colorbar=True)
        
        plt.savefig('figures/geostrophic_flow/residuals.png',\
                    bbox_inches='tight', dpi=400)
        plt.close()

    
    
    
        for depth in depths:
            oc.VectorOverheadPlot_compare(Ures, Vres, U, V, lat,\
                                  lon, p_ladcp[:,0],\
                                  depth, bathy_file,\
                                  average_window=200)
            plt.savefig('figures/geostrophic_flow/vectors_at_'\
                        + str(depth) + 'm.png',\
                        bbox_inches='tight',\
                        dpi=400)
        plt.close('all')
    
    return Ures, Vres, Ugeo, Vgeo

    

def velProfiles_v2(U, V, z, lat, lon, bathy, stns=np.arange(5,13)):
    """
    Revised velocity profiles targeting stations of interest
    """
    
    dist = iw.distance(lat, lon)
    dist = dist[stns]
    
    U = U[:,stns]
    V = V[:,stns]
    fig, ax = plt.subplots(2,2, figsize=(12,8))
    
    
    for i in range(len(stns)):
        ax[0,0].plot(U[:,i], z)
    ax[0,0].set_ylim(0, 4000)
    ax[0,0].set_xlabel('m/s')
    ax[0,0].set_ylabel('Pressure (db)')
    ax[0,0].set_title('U')
    ax[0,0].invert_yaxis()
    
    for i in range(len(stns)):
        ax[0,1].plot(V[:,i], z, label='station ' + str(stns[i]))
    ax[0,1].legend()
    ax[0,1].set_ylim(0, 4000)
    ax[0,1].set_xlabel('m/s')
    ax[0,1].set_title('V')
    ax[0,1].invert_yaxis()
    ax[0,1].yaxis.tick_right()
        
        
    # Get flow minus geostrophic flow
    Ures, Vres, Ugeo, Vgeo = velocityMinusGeostrophic(ladcp, ctd,\
                                                      depths, plots=False)
    
    Ures = Ures[:,stns]
    Vres = Vres[:,stns]
    

    
    
    for i in range(len(stns)):
        ax[1,0].plot(Ures[:,i], z)
    ax[1,0].set_ylim(0, 4000)
    ax[1,0].set_xlabel('m/s')
    ax[1,0].set_ylabel('Pressure (db)')
    ax[1,0].set_title('U - minus Geostrophic Flow')
    ax[1,0].invert_yaxis()
    
    for i in range(len(stns)):
        ax[1,1].plot(Vres[:,i], z, label='station ' + str(stns[i]))
    ax[1,1].legend()
    ax[1,1].set_ylim(0, 4000)
    ax[1,1].set_xlabel('m/s')
    ax[1,1].set_title('V - minus Geostrophic Flow')
    ax[1,1].invert_yaxis()
    ax[1,1].yaxis.tick_right()
    
    plt.tight_layout()
    plt.savefig('figures/velprofiles/interesting_stations.png',\
                bbox_inches='tight',\
                dpi=400)
    

    
    
    