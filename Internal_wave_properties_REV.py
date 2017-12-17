#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:44:35 2017

@author: manishdevana
This toolbox calculates internal wave properties

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
import pandas as pd




# load ctd, ladcp, and bathymetry data in if not already present
#if 'ladcp' not in globals(): 
#    ladcp, ctd, bathy = data_load.load_data()
#    rho_neutral =  np.genfromtxt('neutral_rho.csv', delimiter=',')
#    strain = np.genfromtxt('strain.csv', delimiter=',')
#    wl_max=350
#    wl_min=100
#    

def reset_test():
    
    ladcp, ctd, bathy = data_load.load_data()
    rho_neutral =  np.genfromtxt('neutral_rho.csv', delimiter=',')
    strain = np.genfromtxt('strain.csv', delimiter=',')
    wl_max=350
    wl_min=100
    
    
def specGrid(data, sp, nfft):
    
    

    # Sampling frequency
    fs = 1./sp
    fmin = 0
    fi = 1/nfft
    fmax = fs-fi
    mx = np.arange(fmin, fmax, fi)

    # Convert 1/lambda to k (wavenumber)
    kx = 2*np.pi*mx
    
    return mx, kx


def power_spec(spectrum):
    
    N = len(spectrum)
    power = (1/N)*np.abs(spectrum)
    
    return power

def SpectrumGen(data, bin_size, fudge=True):
    """
    Function for getting the spectrum and associated wavenumber and wavelength
    axes
    """
    nfft = bin_size
    mask = np.isfinite(data)
    if fudge:
        data = data-np.nanmean(data)
        
    

    spectrum = scipy.fftpack.fft(data[mask], n=nfft)
        

    return spectrum

def integrate_power(power, mx, wl_min, wl_max):
    
    
    m_0 = 1/wl_min
    m_1 = 1/wl_max
    idx = (mx < m_0) & (mx > m_1)
    
    variance = np.trapz(power[idx], mx[idx])
    
    return variance

def power_int_smart(power, mx, wl_min, wl_max):
    
    idx0 = np.nanargmax(power)
    idx1 = [idx0-1, idx0, idx0+1]
    
    variance = np.trapz(power[idx0-1:idx0+1], 2*np.pi*mx[idx0-1:idx0+1])
    
    
    return variance
    


def SpecPowerInt_binned(data, sp, bin_idx, wl_min, wl_max):
    
    
    data_spec = []
    data_pow = []
    for dataIn in data.T:
        data_i = np.vstack([SpectrumGen(dataIn[binIn], bin_size)\
                              for binIn in bin_idx])
        data_spec.append(data_i)
        data_pow.append(power_spec(data_i))
    
    data_mx, data_kx = specGrid(data[bin_idx[0,:],0], bin_size)
        
    data_int = []
    for station in data_pow:
        data_i = np.vstack([integrate_power(binIn,\
                    data_mx, wl_min, wl_max) for binIn in station])
        data_int.append(data_i)
        data_i = []
    
    data_int = np.hstack(data_int)
    
    return data_int, data_pow, data_spec, data_mx, data_kx


    

def internal_wave_PE(rho_neutral, ref_rho, N2, z, bin_idx, wl_min, wl_max, strain, bin_size, fudge=True):
    """ 
    Calculate internal wave potential energy based on isopycnal displacements
    and using neutral densities. (optional to do this) The function should work
    the same with normal density and an accompanying reference density profile.
    
    """
    
    # filter rho neutral
    
    drho = np.diff(ref_rho, axis=0);
    dz = np.diff(z, axis=0)
    zeros = np.full((1,rho_neutral.shape[1]), np.nan)
    drho = np.append(drho, zeros, axis=0)
    dz = np.append(dz, zeros, axis=0)
    drhodz = drho/dz
    drhodz2 = np.vstack([oc.verticalBoxFilter1(cast, zIn, box=400)\
                   for cast, zIn in zip(drhodz.T, z[:-1,:].T)]).T
    
    if fudge:
        eta = strain*dz
    
    
    eta = eta**2;
    sp = np.nanmean(np.gradient(z[:,0]))
    
    eta_mx, eta_kx = specGrid(eta[bin_idx[0,:],0], sp, bin_size)
    
    eta_i = []
    power_i = []
    Eta_all = []
    eta_power = []

    for cast in eta.T:
        
        for bins in bin_idx:
            spectrum = SpectrumGen(cast[bins], bin_size)
            eta_i.append(spectrum)
            power_i.append(power_spec(spectrum))
            
        
        eta_i = np.vstack(eta_i)
        power_i = np.vstack(power_i)
        Eta_all.append(eta_i)
        eta_power.append(power_i)
        eta_i = []
        power_i = []
        
    eta_int = []
    for station in eta_power:
        eta_i = np.vstack([power_int_smart(binIn,\
                    eta_mx, wl_min, wl_max) for binIn in station])
        eta_int.append(eta_i)
        eta_i = []
    
    eta_int = np.hstack(eta_int)
    
    # Compute segment mean N2 values
    N2mean = []
    for binIn in bin_idx:
        N2mean.append(np.nanmean(N2[binIn,:], axis=0))
    
    N2mean = np.vstack(N2mean)
    
    Ep = 0.5*N2mean*eta_int
    
    return Ep, eta_power, eta_kx, N2mean, Eta_all


def internal_wave_KE(U, V, z, bin_idx, wl_min, wl_max, bin_size):
    """
    Calculates internal wave kinetic energy
    """
    
    
    Uspeci = []
    Vspeci = []
    Uspec = []
    Vspec  = []
    Upowi = []
    Vpowi = []
    Upower = []
    Vpower = []
    U = U**2
    V = V**2
    
    sp = np.nanmean(np.gradient(z, axis=0))
    
    U_mx, U_kx = specGrid(U[bin_idx[0,:],0], sp, bin_size)
    
    for Ui, Vi in zip(U.T, V.T):
        
        for binIn in bin_idx:
            Uspec1 = SpectrumGen(Ui[binIn], bin_size, fudge=True)
            Upowi.append(power_spec(Uspec1))
            Uspeci.append(Uspec1)
            Vspec1 = SpectrumGen(Vi[binIn], bin_size, fudge=True)
            Vpowi.append(power_spec(Vspec1))
            Vspeci.append(Vspec1)
        
        Uspeci = np.vstack(Uspeci)
        Vspeci = np.vstack(Vspeci)
        Upowi = np.vstack(Upowi)
        Vpowi = np.vstack(Vpowi)
        
        Uspec.append(Uspeci)
        Vspec.append(Vspeci)
        Upower.append(Upowi)
        Vpower.append(Vpowi)
        Uspeci = []
        Vspeci = []
        Upowi = []
        Vpowi = []
    
    # integrate Power Spec of U and V between chosen vertical wavelengths
    Uint = []
    Vint = []
    
    for Us, Vs in zip(Upower, Vpower):
        Ui = np.vstack([power_int_smart(binIn,\
                    U_mx, wl_min, wl_max) for binIn in Us])
        Vi = np.vstack([power_int_smart(binIn,\
                    U_mx, wl_min, wl_max) for binIn in Vs])
        Uint.append(Ui)
        Vint.append(Vi)
        
        Ui = []
        Vi = []
        
    
    Uint = np.hstack(Uint)
    Vint = np.hstack(Vint)
    
    Ek = 0.5*(Uint + Vint)
    
    return Ek, Upower, Vpower, U_kx, Uspec, Vspec
    
    
def cleanCTD(ctd):
    
    S, T, p_ctd, lat, lon = oc.loadCTD(ctd)
    
    Spoly = []
    
    for cast in S.T:
        fitrev = oc.vert_polyFit(cast, p_ctd[:,0], 100, deg=1)
        Spoly.append(fitrev)
        
    Spoly = np.vstack(Spoly).T
    S = S - Spoly
    
    Tpoly = []
    
    for cast in T.T:
        fitrev = oc.vert_polyFit(cast, p_ctd[:,0], 100, deg=1)
        Tpoly.append(fitrev)
        
    Tpoly = np.vstack(Tpoly).T
    T = T - Tpoly
    
    np.savetxt('Trev.csv',T)
    np.savetxt('Srev.csv',S)
   
    

def strain_ctd(N2, S, T, p_ctd, lat, lon, adiabatic=False):
    
    z = gsw.z_from_p(p_ctd,lat)
    
    if adiabatic:
        N2ref = oc.adiabatic_level(S, T, z, lon, lat)
    else:
        N2ref = []
        
        for cast in N2.T:
            fitrev = oc.vert_polyFit(cast, p_ctd[:,0], 100)
            N2ref.append(fitrev)
        
        N2ref = np.vstack(N2ref).T
    
    N2 = N2 / (2*np.pi)**2
    Nref = N2ref / (2*np.pi)**2
    
    strain = (N2 - N2ref)/N2ref
    
    return strain
    
  
                    


def internal_wave_energy(ctd, ladcp,  rho_neutral, bathy, strain,\
                         ctd_bin_size=512, ladcp_bin_size=512,\
                         wl_min=100, wl_max=300, plots=False):
    
    # Load Hydrographic Data
    U, V, p_ladcp = oc.loadLADCP(ladcp)
    S, T, p_ctd, lat, lon = oc.loadCTD(ctd)
    
    Upoly = []
    
    for cast in U.T:
        fitrev = oc.vert_polyFit(cast, p_ladcp, 100, deg=1)
        Upoly.append(fitrev)
        
    Upoly = np.vstack(Upoly).T
    U = U - Upoly
    
    Vpoly = []
    
    for cast in V.T:
        fitrev = oc.vert_polyFit(cast, p_ladcp, 100, deg=1)
        Vpoly.append(fitrev)
        
    Vpoly = np.vstack(Vpoly).T
    V = V - Vpoly
    
    strainpoly = []
    for cast in strain.T:
        fitrev = oc.vert_polyFit(cast, p_ctd[:,1], 100, deg=1)
        strainpoly.append(fitrev)
        
    strainpoly = np.vstack(strainpoly).T
    strain = strain - strainpoly
    
    SA = gsw.SA_from_SP(S, p_ctd, lon, lat)
    CT = gsw.CT_from_t(SA, T, p_ctd)
    N2, dump = gsw.stability.Nsquared(SA, CT, p_ctd, lat)
    N2 = np.abs(N2)
    
    maxDepth = 4000
    idx_ladcp = p_ladcp[:,-1] <= maxDepth
    idx_ctd = p_ctd[:,-1] <= maxDepth
    
    rho_neutral = rho_neutral[idx_ctd, :]
    strain = strain[idx_ctd, :]
    S = S[idx_ctd,:]
    p_ctd = p_ctd[idx_ctd,:]
    U = U[idx_ladcp, :]
    V = V[idx_ladcp, :]
    p_ladcp = p_ladcp[idx_ladcp,:]
    # Bin CTD data
    ctd_bins = oc.binData(S, p_ctd[:,0], ctd_bin_size)
    
    depths = np.vstack([np.nanmean(p_ctd[binIn]) for binIn in ctd_bins])
    dist = gsw.distance(lon, lat)
    dist = np.cumsum(dist)/1000
    dist = np.append(0,dist)
    
    
    # Bin Ladcp Data
    ladcp_bins = oc.binData(U, p_ladcp[:,0], ladcp_bin_size)
    
    # generate reference profiles for neutral densities
    ref_rho = []
    counter = 0
    for cast in rho_neutral.T:
        rho_i = oc.vert_polyFit2(cast, p_ctd[:,0], 100)
        ref_rho.append(rho_i)
        counter += 1
    
    ref_rho = np.vstack(ref_rho).T
    rho_rev = rho_neutral-ref_rho
    
    

    
    z = -1*gsw.z_from_p(p_ctd, lat)
    # Calculate Potential Energy
    Ep, eta_power, eta_kx, N2mean, etaSpec = internal_wave_PE(rho_rev,\
                         ref_rho, N2, z, ctd_bins,  wl_min, wl_max, strain, ctd_bin_size)
    
    # Calculate Kinetic Energy
    Ek, Upow, Vpow, UVkx, Uspec, Vspec = internal_wave_KE(U, V,\
                                                p_ladcp, ladcp_bins, wl_min, wl_max, ladcp_bin_size)
    
    # Total Kinetic Energy
    Etotal = Ek + Ep
    
    if plots:
        fig = plt.figure()
        plt.contourf(dist, np.squeeze(depths), np.log10(Etotal))
        plt.colorbar()
        plt.fill_between(dist, bathy, 4000, color = '#B4B4B4')
        plt.ylim(0, 4000)
        plt.gca().invert_yaxis()
        plt.title('Log 10 Internal Wave Energy')
        plt.savefig('figures/internal_energy.png', dpi=400, bbox_inches='tight')
    
    return Ek, Ep, Etotal, eta_power, Upow,\
                 Vpow, UVkx, eta_kx, N2mean,\
                 wl_min, wl_max, dist, depths,\
                 U, V, p_ladcp, Uspec, Vspec, etaSpec





def frequencyEstimator(ctd, ladcp, bathy, rho_neutral, strain,\
                       wl_min=100, wl_max=500, full_set=False):
    """ 
    Function for calculating the intrinsic frequency of the internal waves
    in bin setups
    """
    
    U, V, p_ladcp = oc.loadLADCP(ladcp)
    S, T, p_ctd, lat, lon = oc.loadCTD(ctd)
    
    
    Ek, Ep, Etotal, eta_power,\
        Upow, Vpow, UVkx, eta_kx,\
        N2mean, wl_min, wl_max,\
        dist, depths, U, V, p_ladcp,\
        Uspec, Vspec, etaSpec =\
        internal_wave_energy(ctd, ladcp,\
        rho_neutral,\
        bathy, strain, wl_min=wl_min, wl_max=wl_max)
        
         
    # Use ratios to solve for internal frequncys
    f = np.nanmean(gsw.f(lat))
    
    omega = f*np.sqrt(Etotal/(Ek-Ep))

    m = np.mean((wl_min, wl_max))
    m = (2*np.pi)/300
    kh = (m/np.sqrt(np.abs(N2mean)))*(np.sqrt(omega**2 - f**2))
    mask = kh == 0
    kh[mask]= np.nan
    lambdaH = 1e-3*(2*np.pi)/kh
    
    
    k = omega
        
    eta_power_export = np.vstack(eta_power)
    eta_kx_export = np.vstack(eta_kx)
    Up_export = np.vstack(Upow)
    Vp_export = np.vstack(Vpow)
    UVkx_export = np.vstack(UVkx)
    

    np.savetxt('eta_power.csv',eta_power_export)
    np.savetxt('eta_kx.csv',eta_kx_export)
    np.savetxt('Upow.csv',Up_export)
    np.savetxt('Vpow.csv',Vp_export)
    np.savetxt('UVkx.csv',UVkx_export)


        
        
    # look for wavenumber maxes
    
   
    # get mean spectra\
    
    eta_mean = []
    for station in eta_power:
        eta_mean.append(np.nanmean(station, axis=0))
    
    eta_mean = np.vstack(eta_mean).T
    
    
    aspect = kh/m 
    
    file2save = pd.DataFrame(lambdaH)
    file2save.index = np.squeeze(depths)
    file2save.to_excel('lambdaH.xlsx')
    file2save = pd.DataFrame(kh)
    file2save.index = np.squeeze(depths)
    file2save.to_excel('Kh.xlsx')
    file2save = pd.DataFrame(omega)
    file2save.index = np.squeeze(depths)
    file2save.to_excel('omega.xlsx')
    file2save = pd.DataFrame(aspect)
    file2save.index = np.squeeze(depths)
    file2save.to_excel('aspect.xlsx')
    
    np.savetxt('eta_mean.csv', eta_mean)
        
    
    np.savetxt('kh.csv', kh)
    np.savetxt('lamdah.csv', lambdaH)
    np.savetxt('omega.csv', omega)
    
    if full_set:
        return lambdaH, kh, omega, N2mean,\
                dist, depths, U, V, p_ladcp,\
                Uspec, Vspec, etaSpec, aspect,\
                Ek, Ep, Etotal
                
    else:
        return lambdaH, kh, omega, N2mean


def shear2strain(ctd, ladcp, ctd_bin_size=512, ladcp_bin_size=512,\
                         wl_min=100, wl_max=450):
    
     # Load Hydrographic Data
    U, V, p_ladcp = oc.loadLADCP(ladcp)
    S, T, p_ctd, lat, lon = oc.loadCTD(ctd)
    
    maxDepth = 4000
    idx_ladcp = p_ladcp[:,-1] <= maxDepth
    idx_ctd = p_ctd[:,-1] <= maxDepth
    
    S = S[idx_ctd,:]
    T = T[idx_ctd,:]
    p_ctd = p_ctd[idx_ctd,:]
    U = U[idx_ladcp, :]
    V = V[idx_ladcp, :]
    p_ladcp = p_ladcp[idx_ladcp,:]
    # Bin CTD data
    ctd_bins = oc.binData(S, p_ctd[:,0], ctd_bin_size)
    
#    Upoly = []
#    
#    for cast in U.T:
#        fitrev = oc.vert_polyFit(cast, p_ladcp, 100)
#        Upoly.append(fitrev)
#        
#    Upoly = np.vstack(Upoly).T
#    U = U - Upoly
#    
#    Vpoly = []
#    
#    for cast in V.T:
#        fitrev = oc.vert_polyFit(cast, p_ladcp, 100)
#        Vpoly.append(fitrev)
#        
#    Vpoly = np.vstack(Vpoly).T
#    V = V - Vpoly
    
    SA = gsw.SA_from_SP(S, p_ctd, lon, lat)
    CT = gsw.CT_from_t(SA, T, p_ctd)
    N2, dump = gsw.stability.Nsquared(SA, CT, p_ctd, lat)
    N2 = np.abs(N2)
    
    dz = np.diff(p_ladcp, axis=0)
    dz = np.tile(dz, 21)
    
    shear = np.diff(np.sqrt(U**2 + V**2), axis=0)/dz
#    shearU = np.diff(np.srqr, axis=0)/dz
#    shearV = np.diff(V, axis=0)/dz
#    
#    shear = np.sqrt(shearU**2 + shearV**2)
    shear = shear**2
    
#    strain = strain_ctd(N2, S, T, p_ctd, lat, lon)
#    strain = strain**2
    strain = np.genfromtxt('strain.csv', delimiter=',')
    strain = strain[idx_ctd,:]**2
    ctd_bins = oc.binData(S, p_ctd[:,0], ctd_bin_size)
    ladcp_bins = oc.binData(U, p_ladcp[:,0], ladcp_bin_size)  
    
    
    sp = np.nanmean(np.gradient(p_ctd[:,0]))
    strain_int, strain_pow, \
        strain_spec, strain_mx, strain_kx =\
        SpecPowerInt_binned(strain, sp, ctd_bins, wl_min, wl_max)
        
          
    sp = np.nanmean(np.gradient(p_ladcp[:,0]))
    shear_int, shear_pow, \
        shear_spec, shear_mx, shear_kx =\
        SpecPowerInt_binned(shear, sp, ladcp_bins, wl_min, wl_max)
        
    
    N2mean = []
    for binIn in ctd_bins:
        N2mean.append(np.nanmean(N2[binIn,:], axis=0))
    
    N2mean = np.vstack(N2mean)
    
    RW = shear_int/(N2mean*strain_int)
    clean = ~np.isfinite(RW)
    RW[clean] = np.nan
    
    

def thorpe_scales(p, S, T, lat, testing=False):
    """
    Computes thorpe scales displacements for density profile
    """
    rho = oc.rhoFromCTD(S, T, p, lon, lat)
    z = gsw.z_from_p(p,lat)
    
        
    if testing:
        plt.figure()
        plt.plot(rho,z)
        
        plt.figure()
        for i, cast in enumerate(test1.T):
            plt.plot(rho[cast,i]-rho[:,i],z[cast,i])
       
    

def spectral_power_plots(data, kgrid):
    
    
    fig = plt.figure()
    for U1 in data:
        for U2 in U1:
            plt.loglog(kgrid, U2, color='k', alpha=.1)
    
    
    
    
    