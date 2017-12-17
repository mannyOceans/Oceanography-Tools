 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

NOTES:
    - I think I got the shear and strain stuff to work?!?!?!
    - Make sure youve got the right units on the total internal energy calcs
    - figure out why to use Hanning Window
    - make sure the wavelengths of integration are correct
    - BUILD PLOTTING FUNCTIONS!!








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


# ONLY RUN THIS LINE WHEN YOURE ACTUALLY PROCESSING THE DATA OTHERWISE
# IT TAKES FOREVER TO RUN THE SCRIPT

# load ctd, ladcp, and bathymetry data in if not already present
#if 'ladcp_dict' not in globals(): 
#    ladcp_dict, ctd_dict, bathy = data_load.load_data()

default_periodogram_params = {
        'window': 'hanning',
        'nfft': 256,
        'detrend': 'linear',
        'scaling': 'density',
        }

default_params = {
        'bin_size':200,
        'overlap':100,
        'm_0': 1./500., # 1/lamda limits
        'm_c': 1./10., #  1/lamda limits
        'order': 1,
        'nfft':256,
        'periodogram_params': default_periodogram_params,
        'plot_data': 'on',
        'transect_fig_size': (6,4),
        'reference pressure': 0,
        'plot_check_adiabatic': False,
        'plot_spectrums': False,
        }

def bin_data(data, z, params=default_params):
    """ Function for binning a cast with corresping depth grid """
    bin_size = params['bin_size']
    overlap = params['overlap']
    step = bin_size - overlap
    bins = np.arange(z[0], max(z), step)
    binIdx = np.digitize(z, bins)
    binned_data = []
    for cast in data.T:
        binned = [cast[np.squeeze(np.array(np.logical_or(binIdx == i, binIdx == i+1)))] \
                 for i in np.unique(binIdx)[0:-1]]
        binned = np.vstack(binned)
        binned_data.append(binned)

    return binned_data, binIdx-1

def bfrq(T, S, z, lat, lon):
    """ 
    Calculate Bouyancy frequency from practical salinity and temperature.
    Converts to potential temperature and absolute salinity to calculate density
    """
    
    
    SA = gsw.conversions.SA_from_SP(S, z, lon, lat)
    g = gsw.grav(lat, z)
    pdens = gsw.rho_t_exact(SA, T, z)
    dpdens = np.gradient(gsw.rho_t_exact(SA, T, z), axis=0)
    dz = np.gradient(z, axis=0)
    N2 = (g/pdens)*(dpdens/dz)

    return N2


def SpectrumGenerator(data, dz, params=default_params):
    """
    Function for getting the spectrum and associated wavenumber and wavelength
    axes
    """

    nfft = params['nfft']

    # Sampling frequency
    fs = 1./dz
    Nc = .5*fs

    mx = Nc*(np.arange(0, nfft))/(nfft/2)

    spectrum = []
    #Assumes binned data
    for station in data:
        spectrum.append(scipy.fftpack.fft(station, n=nfft, axis=1))
        

    # Convert 1/lambda to k (wavenumber)
    kx = 2*np.pi*mx

    return spectrum, mx, kx

def integratePowerSpec_rev(spectrum, mx, kx, params=default_params):

    # set wavelength limits for integration
    m_0 = params['m_0']
    m_c = params['m_c']
    idx = (mx < m_c) & (mx > m_0)
    mxInt = mx[idx]
    kxInt = kx[idx]
    IntSpec = []

    for station in spectrum:
        # integrate each bin within integration limits defined above
        IntSpec.append(np.trapz((1/len(station))*\
                                np.absolute(station[:,idx]), x=mxInt, axis=1))

    specPower = [(1/len(station[0]))*np.abs(station[:,idx]) for station in spectrum]
    spectrum = [station[:,idx] for station in spectrum]

    return IntSpec, mxInt, kxInt, spectrum, specPower



def PowerSpecAndInt(data, dz, params=default_params):
    """"
    Combines Spectrum generation and integrates each bin spectrum
    
    INPUTS:
        - Data Matrix
        - dz (delta z)  
    
    OUTPUTS:
        - Integrated Spectrum (between wavelengths specified in parameters)
        - Spectrum at specified wavenumbers
        - mx -> grid of 1/wavelengths 
        - kx -> wavenumber grid
        - specPower -> Power of spectrum

    """

    # Generate Spectrum and Wavelength/Wavenumber Axes
    spectrum, mx, kx = SpectrumGenerator(data, dz, params=params)

    # Integrate power spectrum
    IntSpec, mx, kx, specRev, specPower = integratePowerSpec_rev(spectrum, mx, kx, params=params)

    IntSpec = np.vstack(IntSpec).T
    return IntSpec, specRev, mx, kx, specPower


def adiabatic_level(ctd_dict, params=default_params):
    """
    Adiabatic Leveling from Bray and Fofonoff (1981) - or at least my best
    attempt at this procedure.

    -------------
    Data should be loaded in bins - might try to add a way to check that later

    -------------
    CTD and ladcp data are on two different grids so this function uses the finer
    CTD data to generate the regressions which are then used on the pressure grid


    """
    order = params['order']
    pref = params['reference pressure']

    # Checks for data being in right formats

    PS = ctd_dict['s']
    PT = ctd_dict['t']
    z = ctd_dict['p']
    lat = ctd_dict['lat']
    lon = ctd_dict['lon']
    idx = bin_data(PS, np.array(ctd_dict['p'][:,0]))[1]
    idxUn = np.unique(idx)

    # Drop last bin becuase its only half a bin
    idxUn = idxUn[:-1]
    # Calculate Specific Volume
    SA = gsw.SA_from_SP(PS, z, lon, lat)
    rho = gsw.pot_rho_t_exact(SA, PT, z, pref)
    SV = 1./rho
    N2 = np.full((len(idxUn), SV.shape[1]), np.nan)
    for k, idx1 in enumerate(idxUn):
        idx2 = np.where(np.logical_or(idx == idx1, idx == idx1+1))[0]

        rhobar = np.nanmean(rho[idx2,:], axis=0)
        for i in range(SV.shape[1]):
            test = idx2[np.isfinite(SV[idx2,i])]
            if len(test) >= 0.50*len(idx2):
                poly = np.polyfit(z[test,i], SV[test,i], order)
                g = gsw.grav(lat.T[i], np.nanmean(z[:,0][idx2]))
                N2[k,i] = -1e-4*(g**2)*(rhobar[i]**2)*poly[1]
            else:
                N2[k,i] = np.nan

    if params['plot_check_adiabatic']:
        fig1  = plt.figure()
        plt.contourf(N2)
        plt.colorbar()
        plt.gca().invert_yaxis()
        ax = plt.gca()  # get the current axes
        # make sure all the data fits
        ax.autoscale()


    return N2, PS, PT, z, lat, lon


def strainFromCtd(ctd, params=default_params):
    """
    Calculate Strain Using Adiabatic Leveling Method
    """

    # Get Reference N2 from Adiabatic Leveling
    N2Ref, PS, PT, z_ctd, lat, lon = adiabatic_level(ctd, params=default_params)

    # Calculate N2 from CTD data
    latgrid = np.meshgrid(lat, z_ctd[:,0])[0]
    longrid = np.meshgrid(lon, z_ctd[:,0])[0]
    SA = gsw.conversions.SA_from_SP(PS, z_ctd, lon, lat)
    g = gsw.grav(lat, z_ctd)
    pdens = gsw.rho_t_exact(SA, PT, z_ctd)
    dpdens = np.gradient(gsw.rho_t_exact(SA, PT, z_ctd), axis=0)
    dz = np.gradient(z_ctd, axis=0)
    N2 = (g/pdens)*(dpdens/dz)

    # Binned N2 data
    N2binned, binidx = bin_data(N2, z_ctd[:,0])

    # Strain
    Strain = []
    strain = np.full(N2binned[0].shape, np.nan)
    for i, station in enumerate(N2binned):
        for k, binIn in enumerate(station):
            strain[k,:] = (binIn - N2Ref[k,i])/N2Ref[k,i]
        Strain.append(strain)


    return Strain, N2, N2binned, pdens



def shearFromUV(U, V, z, params=default_params):
    """
    Calculate shear (vertical) from U and Z to compare with given shear values

    Use data from before binning to make it quicker
    """
    dU = np.gradient(U, axis=0)
    dV = np.gradient(V, axis=0)
    dz = np.gradient(z, axis=0)

    dzgrid = np.tile(dz, dU.shape[1])

    dUdz = dU/dzgrid
    dVdz = dV/dzgrid

    shear = np.sqrt(dUdz**2 + dVdz**2)

    return shear, dUdz, dVdz

def etaCalc(neutral_dens, z, params=default_params):
    """
    Function for calculating Eta Using Neutral Density (NOT POTENTIAL)
    """
    # Bin neutral densities and depth data
    nd_binned, binIdx = bin_data(neutral_dens, z[:,0])
    zbin = bin_data(z, z[:,0])[0]
    
    # Need dens ref ()
    

    # Calculate Eta
    order = params['order']
    densFit = []
    densCoeff = []
    linFit = []
    densRef = []
    for station, z1 in zip(nd_binned, zbin) :
        for binIn, z2 in zip(station, z1):
            idx = np.isfinite(binIn)
            if sum(idx) < 5:
                # if there arent many points don't waste time
                linFit.append(np.full((100,), np.nan))
                
            else:
                # fits 1st order poly to neutral density 
                # and calculates reference Density
                linCoeff = np.polyfit(z2[idx], binIn[idx], order)
                linFit.append(np.polyval(linCoeff, z2))
        linFit = np.vstack(linFit)
        densRef.append(linFit)
        # resets matrix
        linFit = []
    
    # eta = (nDens - nDensRef)/(dnDensRef/dz)
    eta = []
    for stnNum, station in enumerate(nd_binned):
        dRefdz = np.gradient(densRef[stnNum], axis=1)\
            /np.gradient(zbin[stnNum], axis=1)
        eta.append((station - densRef[stnNum])/dRefdz)
    
    return eta

    
    


def PEcalc(neutral_dens, z, N2, params=default_params):
    """
    Calculate Potential Energy
    """
    
    # Calculate Eta (Isopycnal displacements)
    eta = etaCalc(neutral_dens, z)
    
    # Return Spectrum/Power of eta
    dz = np.nanmean(np.gradient(z, axis=0))
    etaInt, etaSpec, kxEta, mxEta, etaPower = PowerSpecAndInt(eta, dz)
    
    # Calculate PE

    




def totalInternalEnergy(ctd_dict, ladcp_dict, params=default_params):
    """
    Calculate total internal energy and spectra
    """
    T = ctd_dict['t']
    S = ctd_dict['s']
    lat = ctd_dict['lat']
    lon = ctd_dict['lon']
    z = ctd_dict['p']
    # Load Neutral density (calculated using a Matlab Routine )
    neutral_dens =  np.genfromtxt('neutral_densities.csv', delimiter=',')
    
    # Bouyancy Frequency 
    N2 = bfrq(T, S, z, lat ,lon)
    
    # Calculate Eta
    
    
    
    
    
def velocityAnalysis(ladcp_dict, params=default_params):
    """ 
    Random velocity feature analyses
    """
    
    
    # Get direction of flow
    U = ladcp_dict['uupgrid']
    V = ladcp_dict['vupgrid']
    theta = np.arctan(U/V)
    Umean = np.nanmean(U, axis=1)
    Vmean = np.nanmean(V, axis=1)
    W = np.zeros_like(U)
    
    
    lat = ctd_dict['lat']
    lon = ctd_dict['lon']
    dist = gsw.distance(lon, lat)
    dist = np.cumsum(dist)/1000
    dist = np.append(0,dist)
    
    dlat = lat[:,-1] - lat[:,0]
    dlon = lon[:,-1] - lon[:,0]
    pathAngle = np.arctan(dlat/dlon)
    
    thetaRev = theta - pathAngle
    y = np.zeros_like(dist)
    z = ladcp_dict['p_grid']
    fig1  = plt.figure()
    ax = fig1.gca(projection='3d')
    plt.quiver(dist, y, z, U, V, W)
    plt.gca().invert_zaxis()
    plt.show()
    for i in range(len(dist)):
        
        plt.quiver(dist, y, z, U, V, W)
        
        
       
    






def analyzeShearStrainRw(ctd_dict, ladcp_dict, params=default_params):
    """
    Calculate Shear, Strain, and Shear to Strain ratio (Rw) (N2 normalized)
        - Uses CTD dictionary loaded at the top of the script
        - Turn plotting on in default parameters dictionary
    """
    #----------------------------------------------------------------------
    # Manual Shear Calculation
    U = ladcp_dict['uupgrid']
    V = ladcp_dict['vupgrid']
    z_ladcp = ladcp_dict['p_grid']

    # Calculate Shear from U & V
    shearMag, shearU, shearV = shearFromUV(U, V, z_ladcp)

    shearMag = shearMag**2
    # Bin and dz
    shearMagbin = bin_data(shearMag, z_ladcp)[0]
    dz_shear = np.mean(np.diff(np.squeeze(z_ladcp)))

    # Shear^2 spectrum
    ShearInt, ShearSpec, mxShear, kxShear,ShearPow = PowerSpecAndInt(shearMagbin, dz_shear)

    ShearInt = ShearInt*60

    # Strain Calculations

    # Load Necessary Data
    z_ctd = bin_data(ctd_dict['p'], np.array(ctd_dict['p'][:,0]))[0]
    lat = np.squeeze(np.array(ctd_dict['lat']))
    lon = np.squeeze(np.array(ctd_dict['lon']))

    # Practical Salinity and Temperature
    PS = bin_data(ctd_dict['s'], np.array(ctd_dict['p'][:,0]))[0]
    PT = bin_data(ctd_dict['t'], np.array(ctd_dict['p'][:,0]))[0]

    # Get Strain from ctd measurements (Returns strain squared?)
    Strain, N2, N2b, rho = strainFromCtd(ctd_dict)
    Strain2 = [strainIn**2 for strainIn in Strain]


    # Power Spectrum/ integration of Strain
    dz_ctd = dz = np.mean(np.diff(np.squeeze(z_ctd)))
    StrainInt, StrainSpec, mxStrain, kxStrain, StrainPow = PowerSpecAndInt(Strain2, dz_ctd)

    # N2 Bar Calculations
    N2bar = [np.nanmean(cast, axis=1) for cast in N2b]
    N2bar = np.vstack(N2bar).T


    # Cutoff Integrated  Shear Spectrums because of depth grid differences
    maxDepth = 6000
    zb_ladcp = bin_data(z_ladcp, z_ladcp)[0]
    zb_ladcp = zb_ladcp[0]
    idx = zb_ladcp[:,-1] <= maxDepth
    ShearInt = ShearInt[idx,:]
    zbinPlot = np.nanmean(zb_ladcp[idx,:], axis=1)
    ShearPow = [shearIn[idx,:] for shearIn in ShearPow]

    # Calculate Polarization Ratio (Shear to Strain Variance Ration - N2 Normalized)
    Rw = ShearInt/(N2bar*StrainInt)

    return Rw, ShearInt, StrainInt, ShearSpec,\
                    StrainSpec, StrainPow, ShearPow,\
                    N2, mxShear, kxShear, mxStrain, \
                    kxStrain, rho, shearMag, zbinPlot


def transectPlots(params=default_params):
    """
    This function plots data (will be built up)
    """
    # main ctd data
    T = ctd_dict['t']
    S = ctd_dict['s']
    lat = ctd_dict['lat']
    lon = ctd_dict['lon']
    z_ctd = ctd_dict['p']

    # Data cleanups / gsw conversions
    SA = gsw.conversions.SA_from_SP(S, z_ctd,\
                                    np.meshgrid(lon, z_ctd[:,0])[0],\
                                    np.meshgrid(lat, z_ctd[:,0])[0])
    CT = gsw.conversions.CT_from_t(SA, T, z_ctd)
    PT = gsw.conversions.pt0_from_t(SA, T, z_ctd)
    sigma0 = gsw.density.sigma0(SA, CT)



    # main ladcp_data
    U = ladcp_dict['u']
    V = ladcp_dict['v']
    Vel = np.sqrt(U**2 + V**2)
    z_ladcp = ladcp_dict['p_grid']

    # plot x-axis as distance travelled
    dist = gsw.distance(lon, lat)
    dist = np.cumsum(dist)/1000
    dist = np.append(0,dist)

    # grids for plotting
    distGrid, depth_ctd = np.meshgrid(dist, z_ctd[:,0])
    distGridLadcp, depth_ladcp = np.meshgrid(dist, z_ladcp)

    # Calculate Shear, Strain, and RW along with spectrums and what not
    Rw, ShearInt, StrainInt, ShearSpec, StrainSpec, StrainPow, ShearPow, N2,\
                mxShear, kxShear, mxStrain, kxStrain, Rho, shear, zbinPlot\
                = analyzeShearStrainRw(ctd_dict, ladcp_dict)

    yaxes = [0, 4000]
    xlabels = 'Distance Travelled Along Transect(km)'
    ylabels = 'Depth (m)'
    # CTD plots
    figSizeTrans = params['transect_fig_size']
    # Temperature
    levels = np.arange(np.nanmin(PT), np.nanmax(PT), 0.1)
    fig1 = plt.figure(figsize=figSizeTrans)
    plt.contourf(distGrid, depth_ctd, PT,\
                 cmap=cmocean.cm.thermal)
    plt.colorbar(label='Temperature ($^\circ$C)')
    plt.contour(distGrid, depth_ctd, PT, levels=levels,\
                colors='k', linewidths=.25)
    plt.ylim(yaxes)
    plt.gca().invert_yaxis()
    plt.fill_between(dist, bathy, 4000, color = '#B4B4B4')
    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.title('Potential Temperature')
    plt.savefig('figures/Potential_Temperature.png',bbox_inches='tight', dpi=1000)

    # Salinity
    levels = np.arange(np.nanmin(SA), np.nanmax(SA), 0.1)
    fig1 = plt.figure(figsize=figSizeTrans)
    plt.contourf(distGrid, depth_ctd, SA,\
                 cmap=cmocean.cm.haline)
    plt.ylim(yaxes)
    plt.gca().invert_yaxis()
    plt.colorbar(label='PSU')
    plt.contour(distGrid, depth_ctd, SA, levels=levels,\
                colors='k', linewidths=.25)
    plt.fill_between(dist, bathy, 4000, color = '#B4B4B4')
    plt.title('Absolute Salinity')
    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.savefig('figures/Absolute_Salinty.png',bbox_inches='tight', dpi=1000)

    # N2
    levels = np.arange(np.nanmin(np.log10(N2)), np.nanmax(np.log10(N2)), 0.1)
    fig1 = plt.figure()
    N2plot1 = plt.contourf(distGrid, depth_ctd, np.log10(N2),\
                           cmap=cmocean.cm.phase)
    plt.ylim(yaxes)
    plt.gca().invert_yaxis()
    plt.colorbar(label='$log_{10} N^2&$')
#    plt.contour(distGrid, depth_ctd, np.log10(N2), levels=levels,\
#                colors='k', linewidths=.25)
    plt.fill_between(dist, bathy, 4000, color = '#B4B4B4')
    plt.title('Buoyancy Frequency (N2)')
    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.savefig('figures/N2.png',bbox_inches='tight', dpi=400)


    # Pdens
    levels = np.arange(np.nanmin(sigma0), np.nanmax(sigma0), 0.005)
    fig1 = plt.figure()
    plt.contourf(distGrid, depth_ctd, sigma0,\
                 cmap=cmocean.cm.tempo)
    plt.ylim(yaxes)
    plt.ylabel(ylabels)
    plt.xlabel(xlabels)
    plt.gca().invert_yaxis()
    plt.colorbar(label='$kg/m^3$')
    plt.contour(distGrid, depth_ctd, sigma0, levels=levels,\
                colors='pink', alpha=.7, linewidths=.25)
    plt.fill_between(dist, bathy, 4000, color = '#B4B4B4')
    plt.title('Potential Density')

    plt.savefig('figures/Potential_density.png',bbox_inches='tight', dpi=400)


    # LADCP PLOTS
    velPlot = (Vel.T - np.nanmean(Vel, axis=1)).T
    levels = np.arange(np.nanmin(Vel), np.nanmax(Vel), 0.1)
    fig1 = plt.figure(figsize=figSizeTrans)
    plt.contourf(distGridLadcp, depth_ladcp, Vel,\
                 cmap=cmocean.cm.speed)
    plt.colorbar(label='$m/s$')
#    plt.contour(distGridLadcp, depth_ladcp, Vel, levels=levels,\
#                colors='k', linewidths=.25)
    plt.ylim(yaxes)
    plt.gca().invert_yaxis()
    plt.fill_between(dist, bathy, 4000, color = '#B4B4B4')
    plt.title('Velocity Magnitude')
    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.savefig('figures/Velocity.png',bbox_inches='tight', dpi=1000)


    # Shear Plot
    shearLog = np.log10(shear)
    shearTrim = shearLog > -6.5
    shear2 = shearLog
    shear2[~shearTrim] = np.nan
    levels = np.arange(np.nanmin(shear), np.nanmax(shear), 0.1)
    fig1 = plt.figure(figsize=figSizeTrans)
    plt.contourf(distGridLadcp, depth_ladcp, shear2,\
                 cmap='jet')

    plt.colorbar(label='$m/s$')
#    plt.contour(distGridLadcp, depth_ladcp, Vel, levels=levels,\
#                colors='k', linewidths=.25)
    plt.ylim(yaxes)
    plt.gca().invert_yaxis()
    plt.fill_between(dist, bathy, 4000, color = '#B4B4B4')
    plt.title('Shear')
    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.savefig('figures/shear.png',bbox_inches='tight', dpi=1000)
    
    plt.close('all')



def distance(lat, lon):
    """
    Takes lat and long of a transect and converts to km travelled between each 
    station of the transect
    """
    dist = gsw.distance(lon, lat)
    dist = np.cumsum(dist)/1000
    dist = np.append(0,dist)
    
    return dist

def SpectrumAnalysis(params=default_params):
    
    
    lat = ctd_dict['lat']
    lon = ctd_dict['lon']
    dist = gsw.distance(lon, lat)
    dist = np.cumsum(dist)/1000
    dist = np.append(0,dist)
    z = np.squeeze(ladcp_dict['p_grid'])
    
    

    # Load Spectrum Caclulations
    Rw, ShearInt, StrainInt, ShearSpec,\
                    StrainSpec, StrainPow, ShearPow,\
                    N2, mxShear, kxShear, mxStrain, \
                    kxStrain, rho, shearMag, zplots\
                    = analyzeShearStrainRw(ctd_dict, ladcp_dict)
    
    # shear trimming of low values
    idxshear = np.log10(shearMag) < -6.5
    shearMag[idxshear] = np.nan
    
    # plot spectra vs wavenumbers
    fig1 = plt.figure()
    for station in StrainPow:
        for level in station:
            plt.loglog(kxStrain, np.abs(level), color='k')
    
    plt.title('Strain Spectra Power')
    plt.xlabel('k - wavenumber')
    plt.ylabel('strain')
    
    
#    fig2 = plt.figure()
#    for station in ShearPow:
#        for level in station:
#            plt.loglog(kxShear, np.abs(level), color='k')
#    
#    plt.title('Shear Spectra Power')
#    plt.xlabel('k - wavenumber')
#    plt.ylabel('strain')
#    
    
    # plot Rw by bins
    yaxes = [0, 4000]
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
    
    scf = ax1.contourf(dist, z, np.log10(shearMag) \
                       - np.nanmean(np.log10(shearMag)),\
                       cmap='seismic')
    ax1.set_ylim(yaxes)
    plt.colorbar(scf, ax=ax1, label='$log_{10} shear$')
    ax1.invert_yaxis()
    ax1.fill_between(dist, bathy, 4000, color = '#B4B4B4')
    ax1.set_title('Vertical Shear (DU/dz)')
    ax1.set_xlabel('Distance along Transect')
    ax1.set_ylabel('Pressure (dB)')
    
    
    rcf = ax2.contourf(dist, zplots, Rw, cmap=cmocean.cm.curl)
    ax2.set_ylim(yaxes)
    plt.colorbar(rcf, ax=ax2, label='Rw')
    ax2.invert_yaxis()
    ax2.fill_between(dist, bathy, 4000, color = '#B4B4B4')
    ax2.set_title('Rw - Shear (N2 normalized) to Strain Ratio')
    ax2.set_xlabel('Distance along Transect')
    ax2.set_ylabel('Pressure (dB)')
    
    plt.tight_layout()
    
    plt.savefig('figures/Rw_and_shear.png', bbox_inches='tight', dpi=400)
    plt.close('all')
    
                     
    
