#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 11:07:12 2017

NOTES:
    - Switch Power Spec Integration to using fft instead of periodogram
    - Rw is Wayyyy to high? maybe it is becauase I havent normalized but stil
        seems wrong
    - Understand units /  magnitude of values for strain to make sure they
            are the right values
    - Understand N2 equation for adiabatic leveling 
    - figure out a proper equation for calculating N2 in general
    - Go back and fix switch seawater package to gsw package
    - Make sure youve got the right units on the total internal energy calcs
    - figure out why to use Hanning Window
    - make sure the wavelengths of integration are correct
    - BUILD PLOTTING FUNCTIONS!!
    
    
    

CURRENT STATUS: Calculating strain but the values are HUGE so its probably way
way off. Must  be a units thing???




@author: manishdevana
"""

import numpy as np
import scipy.signal as sig
import scipy
import seawater as sw
import matplotlib.pyplot as plt
import data_load
import gsw
import cmocean


# ONLY RUN THIS LINE WHEN YOURE ACTUALLY PROCESSING THE DATA OTHERWISE 
# IT TAKES FOREVER TO RUN THE SCRIPT
# ladcp_dict, ctd_dict = data_load.load_data()

default_periodogram_params = {
        'window': 'hanning',
        'nfft': 256,
        'detrend': 'linear',
        'scaling': 'density',
        }

default_params = {
        'bin_size':200,
        'overlap':100,
        'm_0': 1./150.,
        'm_c': 1./15.,
        'order': 1,
        'nfft':256,
        'periodogram_params': default_periodogram_params,
        'plot_data': 'on',
        'reference pressure': 0
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
        
    return binned_data, binIdx

def powerSpecUV(U_binned, V_binned, dz, params=default_params):
    """ Computes power Spectrum for binned data:
        Currently only deals with U and V spectrum 
        Shear and Strain (Rw) calculations come later
                                                
    """
    fs = 1./dz # sampling frequency (m^-1)
    m_all = []
    KE_bins = []
    KE_spec = []
    m_bin = []
    count = 0
    count2 = 0
    for stationU, stationV in zip(U_binned, V_binned):
        count += 1
        for Ubin, Vbin in zip(stationU, stationV):
            count2 +=1
            if np.all(np.isfinite(Ubin)) and np.all(np.isfinite(Vbin)):
                # use periodogram to estimate power spectral density with 
                # frequency grid set by sampling rate (dz)
                m, PV = sig.periodogram(Vbin, fs=fs, **params['periodogram_params'])
                PU = sig.periodogram(Ubin, fs=fs, **params['periodogram_params'])[1]
                m_bin.append(m)
                KE = (PU + PV)/2.
                KE_bins.append(KE)
            else:
                # If there is nans in the bin, the spectral density will not be
                # counted and the bin will be filled with NaN. However, to fill
                # the bin, it is assumed that the first bin was filled so the 
                # size can be extracted. Probably should find a better way 
                # around this!
                m_bin.append(np.full((1, KE_bins[1].shape[0]), np.nan))
                KE_bins.append(np.full((1, KE_bins[1].shape[0]), np.nan))
            
        KE_bins = np.vstack(KE_bins) # stack into matrix form
        m_bin = np.vstack(m_bin)
        KE_spec.append(KE_bins) # list with matrix of KE for each station
        m_all.append(m_bin)
        KE_bins = [] # Reset KE matrix
        m_bin = [] # Reset m matrix
    
    return m_all, KE_spec
        
        

def powerSpecEta(eta, dz, params=default_params):
    """
    Function for taking the power spectral density of Eta
    """
    fs = 1./dz
    m_eta = []
    m1 = []
    Peta = []
    p1 = []
    for station in eta:
        for binIn in station:
            if np.all(np.isfinite(binIn)):
                m, P = sig.periodogram(binIn, fs=fs\
                            , **params['periodogram_params'])
            elif np.sum(np.isfinite(binIn)) > 0.75*(np.size(binIn)):
                idx = np.isfinite(binIn)
                m, P = sig.periodogram(binIn[idx], fs=fs\
                            , **params['periodogram_params'])
            else:
                m = np.nan
                P = np.nan
            
            m1.append(m)
            p1.append(P)
        m_eta.append(m1)
        Peta.append(p1)
        m1 = []
        p1 = []
        
            
    return m_eta, Peta


def SpectrumGenerator(data, dz, params=default_params):
    """
    Function for getting the spectrum and associated wavenumber and wavelength 
    axes
    """
    
    nfft = params['nfft']
    
    # Sampling frequency
    fs = 1./dz
    Nc = .5*fs
    
    mx = Nc*(np.arange(0, 256))/(256/2)
    
    spectrum = []
    #Assumes binned data
    for station in data:
        spectrum.append(scipy.fftpack.fft(station, n=nfft, axis=1))
    
    kx = 2*np.pi*mx
    
    return spectrum, mx, kx
    

def PowerSpecAndInt(data, dz, params=default_params):
    """"
    Combines Spectrum generation and integrates each bin spectrum
    
    """
    
    # Generate Spectrum and Wavelength/Wavenumber Axes
    spectrum, mx, kx = SpectrumGenerator(data, dz, params=params)
    
    # Integrate power spectrum
    IntSpec, mx, kx = integratePowerSpec_rev(spectrum, mx, kx, params=params)
    
    return IntSpec, spectrum, mx, kx


def PowerSpecDensity(data, dz, params=default_params):
    """
    More general function for taking power spectral density That integrates 
    directly in the function
    """
    
    
    
    # Lists to store frequency (wavelength) grid and PSD
    M = []
    P = []
    Ms1 = []
    Ps1 = []
    # Assumes data has been binned
    Pint2 = []
    PInt = []
    for station in data:
        for binIn in station:
            idx = np.isfinite(binIn)
            
            if np.sum(idx) >= 0.75*len(binIn):
            
                Ms, Ps = sig.periodogram(binIn[idx], fs=fs,\
                                     **params['periodogram_params'])
                
            else:
                Ms, Ps = np.nan, np.nan
            
            Ms1.append(Ms)
            Ps1.append(Ps)
            
        M.append(Ms1)
        P.append(Ps1)
        
        Ms1 = []
        Ps1 = []
    
    # Integrate Power Spectra
    PI1 = []
    PInt = []
    for mIn, Pin, in zip(M, P):
        PI1 = integratePowerSpec(Pin, mIn)
        PI1 = np.vstack(PI1)
        PInt.append(PI1)
        PI1 = []
    
    # Stack Into matrix
    PInt = np.hstack(PInt)
    
    # Return Integrated and Spectrum
    return P, M, PInt



def integratePowerSpec(Pdata, mx, kx, params=default_params):
    
    # set wavelength limits for integration
    m_0 = params['m_0'] 
    m_c = params['m_c']
    mxInt = mx[(mx < m_c) & (mx > m_0)]
    IntSpectra = []
    for mIn, Pin in zip(m, Pdata):
        if np.size(Pin) == 1:
            IntSpectra.append(np.nan)
        else:
            idx = (mIn < m_c) & (mIn > m_0)
            I = np.trapz(Pin[idx], x=mIn[idx])
            IntSpectra.append(I)
    return IntSpectra

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
        IntSpec.append(np.trapz(np.absolute(station[:,idx]), x=mxInt, axis=1))
        
        
            
    return IntSpec, mxInt, kxInt
    
    


def IWkineticEnergy(U, V, z, params=default_params):
    
    U, binIdx = bin_data(U, z)
    V = bin_data(V, z)[0]
    # Depth Bins (done separately but should go back and put into data binning function)
    zb_adcp = [z[np.squeeze(np.array(np.logical_or(binIdx == i, binIdx == i+1)))] \
                 for i in np.unique(binIdx)[0:-1]]
    zb_adcp = np.vstack(zb_adcp)
    
    
    dz = int(np.mean(np.diff(z)))
    m, KE_spec = powerSpecUV(U, V, dz)
    count = 0
    KE = []
    for m1, KE1 in zip(m, KE_spec):
        count += 1
        KE_int = integratePowerSpec(KE1, m1)
        KE_int = np.vstack(KE_int)
        KE.append(KE_int)
        KE_int = []
    KE = np.hstack(KE)
    return KE, m, U, V, zb_adcp



def etaCalc(neutral_dens, p_bins, params=default_params):
    """
    Calculate eta (neutral density isopycnal displacement)
    Dgamma_ref is calculcated using 1st order linear fit as default but should 
    check if theres a better way.
    
    """
    order = params['order']
    densFit = []
    densCoeff = []
    linFit = []
    densRef = []
    for station, p1 in zip(neutral_dens, p_bins) :
        for binIn, p2 in zip(station, p1):
            idx = np.isfinite(binIn)
            if sum(idx) < 5:
                # if there arent many points don't waste time
                linFit.append(np.full((100,), np.nan))
                
            else:
                # fits 1st order poly to neutral density 
                # and calculates reference Density
                linCoeff = np.polyfit(p2[idx], binIn[idx], order)
                linFit.append(np.polyval(linCoeff, p2))
        linFit = np.vstack(linFit)
        densRef.append(linFit)
        # resets matrix
        linFit = []
    
    # eta = (nDens - nDensRef)/(dnDensRef/dz)
    eta = []
    for stnNum, station in enumerate(neutral_dens):
        dRefdz = np.gradient(densRef[stnNum], axis=1)\
            /np.gradient(p_bins[stnNum], axis=1)
        eta.append((station - densRef[stnNum])/dRefdz)
    
    return eta




            
                
    
    
            
    



def potentialEnergy(ctd, z, params=default_params):
    """ Function for calculating Potential Energy Spectrum...
    NOTE: Neutral Density must be imported from MATLAB becuase I can't find 
    Neutral Density routine for pythong (will build one later)
    """
    neutral_dens =  np.genfromtxt('neutral_densities.csv', delimiter=',')
    neutral_dens, ctd_binsIdx = bin_data(neutral_dens, z)
    ctd_binned = {key: bin_data(ctd[key], z)[0] for key in ctd}
    
    # Calculate Isopycnal Displacements
    eta = etaCalc(neutral_dens, ctd_binned['p'])
    
    # Power Spectrum of Eta
    dz = np.mean(np.mean(np.gradient(ctd_binned['p'][0][0,:])))
    m_eta, Peta = powerSpecEta(eta, dz)
    
    # Integrating Eta (Not quite PE yet but too lazy to change it right now)
    PE = []
    for m, P in zip(m_eta, Peta):
        PE_int = integratePowerSpec(P, m)
        PE_int = np.vstack(PE_int)
        PE.append(PE_int)
    PE_pre = np.hstack(PE)
    
    # calculate N2 (and segment mean N2)
    # The factor 1e-4 is needed for conversion from dbar to Pa.
    # FROM JESSES CODE : N2 = -1e-4*rhobar**2*g**2*p[order-1, :]
    
    #                -g      d(pdens)
    #        N2 =  ----- x --------
    #               pdens     d(z)

    g = 9.8
    pden = sw.eos80.pden(ctd['s'], ctd['t'], ctd['p'])
    pden = bin_data(pden, z)[0]
    
    N2 = []
    N2bar = [] # Segment Mean N2
    # I think the values are right but should probably check with Alberto
    for station, pIn in zip(pden, ctd_binned['p']):
            dpdensdz = np.gradient(station, axis=1)\
                            /np.gradient(pIn, axis=1)
                            
            N2.append((1e-4)*(g / np.gradient(station, axis=1)) * dpdensdz)
            N2bar.append(np.nanmean((1e-4)*(g / np.gradient(station, axis=1))\
                                 * dpdensdz, axis=1))
            
            
    # Final PE calculation        
    PE = []
    
    # Make it similar matrix to PE_pre
    N2bar = np.vstack(N2bar)
    N2bar = N2bar.T
    
    PE = 0.5*N2bar*PE_pre
    
    zb_ctd = ctd_binned['p'][0]
       
    return PE, N2, N2bar, zb_ctd    
        
        
    






def total_internal_energy_calc(ladcp_dict, ctd_dict, params=default_params):
    
    U = np.array(ladcp_dict['u'])
    V = np.array(ladcp_dict['v'])
    z_ladcp = np.squeeze(np.array(ladcp_dict['p_grid']))
    dz_ladcp = int(np.mean(np.diff(z_ladcp)))
    
    # Kinetic Energy integrated in Bins
    # Go back and make so that bins with no data / Nan are filled in to
    # make sure that there is one single matrix
    KE, m, U, V, zb_ladcp = IWkineticEnergy(U, V, z_ladcp)
    
    # Potential Energy from eta (isopycnal displacements)
    z_ctd = np.array(ctd_dict['p'][:,0])
    
    # separate data grids from CTD dictionary
    data_needed = ['p', 's', 't', 'pt']
    ctd = {key: ctd_dict[key] for key in data_needed}
    
    # Call Potential Energy Calculations
    PE, N2, N2bar, zb_ctd = potentialEnergy(ctd, z_ctd)
    
    # Trim Kinetic Energy since its pressure grid goes deeper
    # maxDepth = np.min([np.max(z_ctd), np.max(z_ladcp)])
    
    # set max depth manually because I keep getting one bin off
    maxDepth = 6000
    idx_ladcp = zb_ladcp[:,-1] <= maxDepth
    KErev = KE[idx_ladcp,:]
    
    
    idx_ctd = zb_ctd[:,-1] <= maxDepth
    PErev = PE[idx_ctd,:]
    # Total Internal Wave Energy (E_total = E_potential + E_kinetic)
    ET = PErev + KErev
    
    logET = np.log10(ET)
    
    # Make midpoints grid for plotting final grid
    z_plot = np.median(zb_ladcp, axis=1)
    z_plot = z_plot[idx_ladcp]
    
    
    return logET, zb_ladcp, z_plot, zb_ctd, N2, N2bar
    


def adiabatic_level(sal, temp, z_ctd, z_ladcp, lat, lon, params=default_params):
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
    
   
    
    
    # Calculate Specific Volume
    SV = []
    RHO = []
    for i in range(len(sal)):
        
        SA = gsw.SA_from_SP(sal[i], z_ctd[i], lon[i], lat[i])
        rho = gsw.pot_rho_t_exact(SA, temp[i], z_ctd[i], pref)
        sv = 1./rho
        rhobar = np.nanmean(rho, axis=1)
        SV.append(sv)
        RHO.append(rhobar)
        sv = []
        
        
    # Linear Regression with x = Pressure, and Y = Specific Volume (CTD Pres)
    P = []
    for m, station in enumerate(SV):
        Poly = np.full((station.shape[0], 2), np.nan)
        for i in range(np.shape(station)[0]):
            idx = np.isfinite(z_ctd[m][i]) & np.isfinite(station[i])
            if np.sum(idx) >= .50*len(idx):
                Poly[i, :] = (np.polyfit(z_ctd[m][i][idx], station[i][idx], order))
                
                # N2 Reference Calculation (WHERE DOES THIS EQUATION COME FROM?!)
                
                
                
                # Inverse Coefficients so you evaluate for Pressure
                Poly[i, :] = Poly[i, ::-1]
            
        P.append(Poly)
    
    # calculate N2 (and segment mean N2)
    # The factor 1e-4 is needed for conversion from dbar to Pa.
    # FROM JESSES CODE : N2 = -1e-4*rhobar**2*g**2*p[order-1, :]
    
    #                -g      d(pdens)
    #        N2 =  ----- x --------
    #               pdens     d(z)
    z1 = bin_data(z_ladcp, z_ladcp)
    z1 = bin_data(z_ladcp, z_ladcp)[0]
    z1 = z1[0]
    
    pr1 = []
    N2ref = []
    N2BnF = []
    
    for i in range(len(P)):
        for binIn in range(len(P[i])):
            g = gsw.grav(lat[i], np.nanmean(z1[binIn]))
            
            # Bray and Fofonoff 1981 N2 Equation?
            N2ref.append(-1e-4*g**2*RHO[i][binIn]**2*P[i][binIn,1])
            
            
        
        N2ref = np.vstack(N2ref)
        N2BnF.append(N2ref)
        N2ref = []
    
    
 
    # Returns N2 Reference for each bin
    return N2BnF



def analyzeShearStrainRw(ctd_dict, ladcp_dict, params=default_params):
    """ 
    Calculate Shear, Strain, and Shear to Strain ratio (Rw) (N2 normalized)
        - Uses CTD dictionary loaded at the top of the script
        - Turn plotting on in default parameters dictionary
    """
    

    z_ctd = bin_data(ctd_dict['p'], np.array(ctd_dict['p'][:,0]))[0]
    z_grid_ctd  = np.array(ctd_dict['p'][:,0])
    
    sal = bin_data(ctd_dict['s'], z_grid_ctd)[0]
    temp = bin_data(ctd_dict['t'], z_grid_ctd)[0]
    lat = np.squeeze(np.array(ctd_dict['lat']))
    lon = np.squeeze(np.array(ctd_dict['lon']))
    z_ladcp = ladcp_dict['p_grid']
    
    # Dz
    dz = np.mean(np.diff(np.squeeze(z_ladcp)))
    # Calculate N2 ref using adiabatic leveling method 
    N2ref = adiabatic_level(sal, temp, z_ctd, z_ladcp, lat, lon)
    
    # Strain Calculations
    N2 = []
    # Calculcate N2 (Check equations - Im using the most simple one)
    for i in range(len(sal)):
         
        #             -g     d(pdens)
        #     N2 =  ----- x --------
        #           pdens     d(z)
        SA = gsw.conversions.SA_from_SP(sal[i], z_ctd[i], lon[i], lat[i])
        g = gsw.grav(lat[i], z_ctd[i])
        pdens = gsw.rho_t_exact(SA, temp[i], z_ctd[i])
        dpdens = np.gradient(gsw.rho_t_exact(SA, temp[i], z_ctd[i]), axis=1)
        dz = np.gradient(z_ctd, axis=1)
        
        # Calculate N2
        N2.append(1e-4*(g/pdens)*(dpdens/dpdens))

    # Calculate Strain (find something to compare the values to)
    # Strain = (N2-N2ref)/ N2ref
    strain = []
    for i in range(len(sal)):
        strain.append((N2[i] - np.tile(N2ref[i], N2[i].shape[1]))\
                      /np.tile(N2ref[i], N2[i].shape[1]))
    
    # Take spectrum of Strain Squared (MAYBE?)
    for i in range(len(strain)):
        strain[i] = strain[i]**2
    # Dz
    dz = np.mean(np.diff(np.squeeze(z_ladcp)))
    
    # Power Spec and Integration of strain
    Pstrain, Mstrain, StrainInt = PowerSpecDensity(strain, dz)
    # RW - Shear to strain polarization ratios
    
    # U and V components are split into Real and Complex, respectively
    shear = ladcp_dict['shear']
    Ushear = np.real(shear)
    Vshear = np.imag(shear)
    
    # Magnitude of shear
    Shear = np.sqrt(Ushear**2 + Vshear**2)
    
    # Bin Shear
    Shear = bin_data(Shear, z_ladcp)[0]
    
    # Dz
    dz = np.mean(np.diff(np.squeeze(z_ladcp)))
    
    # Power Spectrum of Shear
    Pshear, Mshear, ShearInt = PowerSpecDensity(Shear, dz)
    
    # Cut Off deeper sections so arrays match size (Use same method as in
    #    Internal wave calcs)
    
    maxDepth = 6000
    # binning depth data returns a list for some reason???
    zb_ladcp = bin_data(z_ladcp, z_ladcp)[0]
    zb_ladcp = zb_ladcp[0]
    
    idx_ladcp = zb_ladcp[:,-1] <= maxDepth
    ShearInt = ShearInt[idx_ladcp,:]
    
    # Bin Mean N2
    N2bar = np.empty(ShearInt.shape)
    for i in range(len(N2)):
        N2bar[:,i] = np.nanmean(N2[i], axis=1)
    
    # Rw calculations ---THESE VALUES ARE WAY TOO HIGH
    Rw = (ShearInt**2/N2bar)/StrainInt**2
    
    lat = ctd_dict['lat']
    lon = ctd_dict['lon']
    
    M = Mstrain[0][3]
    return Rw, ShearInt, Pshear, StrainInt, Pstrain, lat, lon, M
   
    
def Analysis1(params=default_params):
    
    # Internal Energy Calculations
    logET, zb_ladcp, z_plot, zb_ctd, N2, N2bar =\
        total_internal_energy_calc(ladcp_dict, ctd_dict)
    
    Rw, ShearInt, Pshear, StrainInt, Pstrain, lat, lon, M =\
        analyzeShearStrainRw(ctd_dict, ladcp_dict)
        
    
    
    # Spectrum K axis
    m_0 = params['m_0'] 
    m_c = params['m_c']
    
    idx = (M < m_c) & (M > m_0)
    m = M[idx]
    
    # Wavenumbers
    k = 2*np.pi*m
    
    
    
    

    
    if params['plot_data'] == 'on':
        
        # Make X-axis grid (Distance Traveled)
        dist = gsw.distance(lon, lat)
        dist = np.cumsum(dist)/1000
        dist = np.append(0,dist)
        
        # grids
        dist, depths = np.meshgrid(dist, z_plot)
        
        # Internal Energy Plot
        fig1 = plt.figure()
        plt.contourf(dist, depths, logET, cmap=cmocean.cm.thermal)
        plt.ylim([0, 4000])
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title('Log Total Internal Energy')
        plt.savefig('InteralEnergyPlot.png')
        
        # Rw Plot (not normalized to GM spectrum)
        fig2 = plt.figure()
        plt.contourf(dist, depths, np.log10(Rw/7), cmap=cmocean.cm.balance)
        plt.ylim([0, 4000])
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title('Log10(Rw/7)')
        plt.savefig('RwPlot.png')
        
        
        # Spectrum of strain plots
        
        
        
    
    
    return logET, zb_ladcp, zb_ctd, N2, N2bar
    
        
    
    

    
    