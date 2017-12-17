an#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:32:48 2017

@author: manishdevana
"""

import IW_functions as iw

from netCDF4 import Dataset
import numpy as np
import scipy.signal as sig
import scipy
import seawater as sw
import matplotlib.pyplot as plt
from matplotlib import ticker
import data_load
import gsw
import cmocean

default_params = {
        'depths': [1000, 2000, 2500, 3500]
        }

if 'ladcp_dict' not in globals():
    ladcp_dict, ctd_dict, bathy = data_load.load_data()



# Change if you want a different bathymetry file
fname = 'bathy.nc'

def bathyLoadNc(fname_bathy, lat, lon, add_buffer=True, buffer=.1):
    """
    Loads bathymetric data from Netcdf file and extracts box around transect
    """
    file = Dataset(fname)

    bathyDict = {key:file.variables[key][:] for key in file.variables.keys()}
    file.close()
    lat2 = bathyDict['lat']
    lon2 = bathyDict['lon']
    if add_buffer:
        # go back and actually add the buffer in
        latidx = np.where(np.logical_and(lat2 < np.nanmax(lat)+buffer\
                                , lat2 > np.nanmin(lat)-buffer))[0]

        lonidx = np.where(np.logical_and(lon2 < np.nanmax(lon)+buffer\
                                , lon2 > np.nanmin(lon)-buffer))[0]
    else:
        latidx = np.where(np.logical_and(lat2 < np.nanmax(lat)\
                                , lat2 > np.nanmin(lat)))[0]

        lonidx = np.where(np.logical_and(lon2 < np.nanmax(lon)\
                                , lon2 > np.nanmin(lon)))[0]

    bathyrev = bathyDict['elevation'][latidx,:]
    bathyrev = bathyrev[:,lonidx]
    longrid, latgrid = np.meshgrid(lon2[lonidx], lat2[latidx])



    return bathyrev, longrid, latgrid

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
    
    # U and V breakdown of X in direction of transect line
    phi = np.full_like(U, np.nan)   
    for i in range(len(lat)):
        phi[:,i]  = np.arctan((lat[i]-lat[i-1])/(lon[i]-lon[i-1]))
    
    Ux = np.cos(phi)*Umag
    Vx = np.sin(phi)*Umag
    
    return X, Ux, Vx
    
    

def speedAtZ(U, V, z, depth=3000, bin_width=100):
    """ 
    Extracts mean velocity at depth with average of bin width
    """
    
    z_upper = np.nanargmin(np.abs(z - (depth+.5*bin_width)))
    z_lower = np.nanargmin(np.abs(z - (depth-.5*bin_width)))
    zmean = np.nanmean(z[z_lower:z_upper])
    
    Urev = U[z_lower:z_upper,:]
    Urev = np.nanmean(Urev, axis=0)    
    Vrev = V[z_lower:z_upper,:]
    Vrev = np.nanmean(Vrev, axis=0)
    
    return Urev, Vrev, zmean
    

def bathyplotsV1(bathy_file=fname, ladcp=ladcp_dict, params=default_params):
    """
    plots bathymetry data with various velocity components on top
    """
    z = ladcp['p_grid']
    lat = np.squeeze(ladcp['lat'])
    lon = np.squeeze(ladcp['lon'])
    
    # get velocites and velocity in line with transect
    U = ladcp_dict['u']
    V = ladcp_dict['v']

    
    # target Depths
    depths = params['depths']

    # mean flow at depths
    meanFl = [speedAtZ(U, V, z, depth=Z) for Z in depths]
    

    
    # Box of Gebco bathymetry data around transect
    bathy, longrid, latgrid = bathyLoadNc(fname,\
                                          lat,\
                                          lon,\
                                          add_buffer=True)
    

    longridb, latgridb, zgrid = np.meshgrid(lon, lat, z)

    # Surface plot of bathymetry
    X = .3
    Y = .85
    for i, Z in enumerate(depths):
        
        fig = plt.figure()
    
        plt.pcolormesh(longrid, latgrid,\
                       bathy, cmap=cmocean.cm.deep_r,\
                       shading='gouraud')
        plt.colorbar(label='Depth (meters)')
        plt.plot(ladcp['lon'], ladcp['lat'])
        q1 = plt.quiver(lon, lat,\
                        meanFl[i][0],\
                        meanFl[i][1],\
                        scale=2,\
                        color='red',\
                        width=.004)
        plt.quiverkey(q1, X, Y, .5, "1 m/s", labelpos='W')
        plt.title("Flow at " + str(Z) + "m")
#        Y -= .1
        plt.savefig('figures/vel_at_' + str(Z) + '_meters.png',\
                    bbox_inches='tight',\
                    dp=400)
        plt.close(fig)
    
        
    #    plt.contour(longrid, latgrid,\
#                bathy, colors='k',\
#                linestyles='solid',\
#                linewidths=1)
    
def bathyplotsV2(bathy_file=fname, ladcp=ladcp_dict, params=default_params):
    """
    Plot flow at targeted depths along with flow in line of transect
    """
    z = ladcp['p_grid']
    lat = np.squeeze(ladcp['lat'])
    lon = np.squeeze(ladcp['lon'])
    
    # get velocites and velocity in line with transect
    U = ladcp['u']
    V = ladcp['v']

    # Extract for specified depth
    
    # target Depths
    depths = params['depths']

    # Mean flow at targeted depths
    meanFl = [speedAtZ(U, V, z, depth=Z) for Z in depths]
    
    # mean flow at depths of flow in line with transect
    X, Ux, Vx = velocityInLine(U, V, lat, lon)
    meanFlx = [speedAtZ(Ux, Vx, z, depth=Z) for Z in depths]
    
    # Plot data
    
    # Box of Gebco bathymetry data around transect
    bathy, longrid, latgrid = bathyLoadNc(fname,\
                                          ladcp['lat'],\
                                          ladcp['lon'], add_buffer=True)
    

    longridb, latgridb, zgrid = np.meshgrid(lon, lat, z)

    # Surface plot of bathymetry
    X = .3
    Y = .85
    scales = [2, 2, 2, 2]
    for i, Z in enumerate(depths):
        
        fig = plt.figure()
    
        plt.pcolormesh(longrid, latgrid,\
                       bathy, cmap=cmocean.cm.deep_r,\
                       shading='gouraud')
        plt.colorbar(label='Depth (meters)')
        plt.scatter(ladcp['lon'], ladcp['lat'],\
                    c='k',\
                    marker='+')
#        q1 = plt.quiver(lon, lat,\
#                        meanFl[i][0],\
#                        meanFl[i][1],\
#                        scale=scales[i],\
#                        color='red')
        q2 = plt.quiver(lon, lat,\
                        meanFlx[i][0],\
                        meanFlx[i][1],\
                        scale=scales[i],\
                        color='lime',\
                        width=.004)
#        plt.quiverkey(q1, X, Y, .5, "1 m/s", labelpos='W')
        plt.quiverkey(q2, X, Y, .5, "1 m/s", labelpos='W')
        
        plt.title("Flow at " + str(Z) + "m (in parallel to transect direction)")
#        Y -= .1
        plt.savefig('figures/vel_at_' + str(Z) + '_meters_inline_comps.png',\
                    bbox_inches='tight',\
                    dpi=400)
        plt.close(fig)
    
def spectraPlots(ladcp=ladcp_dict, ctd=ctd_dict, params=default_params):
    """
    Function for plotting Spectra of shear and strain .. (will add more spectra
    plots later!)
    """
    
    # Calculate Shear, Strain, and RW along with spectrums and what not
    Rw, ShearInt, StrainInt, ShearSpec,\
        StrainSpec, StrainPow, ShearPow, N2,\
        mxShear, kxShear, mxStrain, kxStrain,\
        Rho, shear, zbinPlot\
        = iw.analyzeShearStrainRw(ctd, ladcp, params=iw.default_params)
    
    # Bin N2
    N2b = iw.bin_data(N2, ctd['p'][:,0])[0]
    N2bmean = np.vstack([np.nanmean(station, axis=1) for station in N2b]).T
        
    # Shear and Strain Spectra Plots
    fig1 = plt.figure()
    for station in StrainPow:
        for level in station:
            plt.loglog(kxStrain, level, color='k')
    
    plt.xlabel('k - wavenumber')
    plt.ylabel('Strain')
    
    #shear normalized by Buoyancy Frequency
    fig2 = plt.figure()
    for i in range(len(ShearPow)):
        for m in range(len(ShearPow[i])):
            plt.loglog(kxShear, ShearPow[i][m]/N2bmean[m,i], color='k')
    
    plt.xlabel('k - wavenumber')
    plt.ylabel('Shear')
    
    
    
    
  
    
   
def vectorPlots(bathy_file=fname, ladcp=ladcp_dict, params=default_params):
    """ 
    Runs and saves plots for flow at depths with and without component of 
    inline flow breakdowns
    """
    
    bathyplotsV2(bathy_file=bathy_file, ladcp=ladcp)
    bathyplotsV1(bathy_file=bathy_file, ladcp=ladcp)
    
    

    
def velProfiles(U, V, z, lat, lon, bathy):
    """
    Function for plotting the veloctity profiles and searching for reversals
    which may indicate the presence of wavelike motions
    """
    
    dist = iw.distance(lat, lon)
    Mag = np.sqrt(U**2 + V**2)
    
    
    for i in range(U.shape[1]):
        
        fig1 = plt.figure()
        plt.plot(U[:,i], z)
        plt.ylim(0, 4000)
        plt.gca().invert_yaxis()
        plt.xlabel('Velocity (U) m/s')
        plt.ylabel('pressure (dB)')
        plt.title('U station No. ' + str(i))
        ax1 = fig1.add_axes([0.72, 0.72, 0.2, 0.2])
        c1 = ax1.contourf(dist, np.squeeze(z), U)
        cb = plt.colorbar(c1, location='top')
        tick_locator = ticker.MaxNLocator(nbins=2)
        cb.locator = tick_locator
        cb.update_ticks()
        ax1.fill_between(dist, bathy, 4000, color = '#B4B4B4')
        ax1.plot(np.tile(dist[i], z.shape), z, color='red')
        ax1.yaxis.tick_right()
        plt.ylim(0, 4000)
        plt.gca().invert_yaxis()

        plt.savefig('figures/velprofiles/U_station No. ' + str(i) + '.png',\
                    bbox_inches='tight', dpi=450)
        plt.close()
        
    for i in range(U.shape[1]):
        
        fig1 = plt.figure()
        plt.plot(V[:,i], z)
        plt.ylim(0, 4000)
        plt.gca().invert_yaxis()
        plt.xlabel('Velocity  (V) m/s')
        plt.ylabel('pressure (dB)')
        plt.title('V station No. ' + str(i))
        ax1 = fig1.add_axes([0.72, 0.72, 0.2, 0.2])
        c1 = ax1.contourf(dist, np.squeeze(z), U)
        cb = plt.colorbar(c1, location='top')
        tick_locator = ticker.MaxNLocator(nbins=2)
        cb.locator = tick_locator
        cb.update_ticks()
        ax1.fill_between(dist, bathy, 4000, color = '#B4B4B4')
        ax1.plot(np.tile(dist[i], z.shape), z, color='red')
        ax1.yaxis.tick_right()
        plt.ylim(0, 4000)
        plt.gca().invert_yaxis()

        plt.savefig('figures/velprofiles/V_station No. ' + str(i) + '.png',\
                    bbox_inches='tight', dpi=450)
        plt.close()
        
    for i in range(U.shape[1]):
        
        fig1 = plt.figure()
        plt.plot(Mag[:,i], z)
        plt.ylim(0, 4000)
        plt.gca().invert_yaxis()
        plt.xlabel('Mag Velocity  (Mag) m/s')
        plt.ylabel('pressure (dB)')
        plt.title('station No. ' + str(i))
        ax1 = fig1.add_axes([0.72, 0.72, 0.2, 0.2])
        c1 = ax1.contourf(dist, np.squeeze(z), U)
        cb = plt.colorbar(c1, location='top')
        tick_locator = ticker.MaxNLocator(nbins=2)
        cb.locator = tick_locator
        cb.update_ticks()
        ax1.fill_between(dist, bathy, 4000, color = '#B4B4B4')
        ax1.plot(np.tile(dist[i], z.shape), z, color='red')
        ax1.yaxis.tick_right()
        plt.ylim(0, 4000)
        plt.gca().invert_yaxis()

        plt.savefig('figures/velprofiles/Mag_station No. ' + str(i) + '.png',\
                    bbox_inches='tight', dpi=450)
        plt.close()
        



        
    
   
    