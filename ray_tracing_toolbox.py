ot#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:40:35 2017


Tools for running ray racing on internal waves in a general sense



@author: manishdevana
"""


# Load necessary modules
import numpy as np
import scipy.signal as sig
import scipy
import seawater as sw
import matplotlib.pyplot as plt
import data_load
import gsw
import cmocean
import oceans as oc
import Internal_wave_properties as iwp
import lee_wave_analysis1 as lw
import pandas as pd


# Load primary equations as functions
def initial_conditions():
    """
    Initial conditions for test runs
    """
    ladcp, ctd, bathy = data_load.load_data()
    U, V, z_grid = oc.loadLADCP(ladcp)
    S, T, p, lat, lon = oc.loadCTD(ctd)
    
    # Calculate N2
#    N2 = oc.gswN2(S, T, p, lat, lon)
#    N2 = N2[:,13]
#    N2 = oc.verticalBoxFilter1(N2, p[:,1])
    N2  = np.genfromtxt('ref_N2.csv', delimiter=',')
    bottom_row = np.full((1,21), np.nan)
    N2 = np.vstack((N2, bottom_row))
    N2 = N2[:,13]
    U = U[:,13]
    V = V[:,13]
    U = oc.verticalBoxFilter1(U, z_grid)
    V = oc.verticalBoxFilter1(V, z_grid)
    k0 = (2*np.pi)/5500
    m0 = (2*np.pi)/300
    z0 = 600
    x0 = 0
    omega = -0.00013426659784432059
    tstep = 1 # in seconds
    runtime = 12 # In Hours
    lat = -53.15857422
    
    return U, V, z_grid, k0, m0, z0, x0, lat, tstep, runtime, omega, N2, p[:,1]



def flow_alignment(U, V):
    """
    Returns flow magnitude with binary direction (positive in direction of mean
    flow and negative in opposition to mean flow )
    This is to allow for simplificaiton of ray tracing. 
    
    Assumption only works if flow is mostly directionally uniform.
    """
    
    angles = np.arctan2(U,V)
    mean_angle = np.nanmean(angles)
    reversed_flow = np.sign(angles) != np.sign(mean_angle)  
    mag = np.sqrt(U**2 + V**2)
    mag[reversed_flow] *= -1
    
    return mag

    
def f_of_z(data, z, z_grid):
    """
    General function for getting the value in vertical profile at specified 
    depth
    """
    target = np.nanargmin(np.abs(z_grid-z))
    
    fz = data[target]
    
    return fz


def meanFlow(flow, z, z_grid):
    """
    Mean flow as a function of depth
    """
    
    target = np.nanargmin(np.abs(z_grid-z))
    
    U = flow[target]
    
    return U

def Nz(z):
    """
    Buoyancy Frequency as function of depth
    """
    
    
def U0(U, z, box=100):
    """
    Clean up U (take out perturbations)
    """
    
    U0 = oc.verticalBoxFilter1(U, z, box=box)
    
    
    return U0


    
    
    
    
    
    









def RT_backtrace_simplified(z0, k0, m0, U, V, omega,\
                            z_grid, lat, lon, x0=0, tstep=5,\
                            test=False, runtime=5, plots=False):
    """
    Run ray tracing backwards in time
    using first order finite differencing (i think thats what im doing)
    """
    if test:
        U, V, z_grid, k0, m0, z0, x0, lat, tstep, runtime, omega = initial_conditions()
        
    runtime = runtime*60*60 #Convert into seconds from hours
    x = x0
    z = -z0
    k = k0
    m = m0
    
    # dkdt = 0 # Not actually necessary to declare this but useful reminder
    # dldt = 0 # Same as above
    flow = flow_alignment(U, V)
    
    # Doppler Shifting Frequency
    omega_int = omega - k*meanFlow(flow, z, z_grid)
    
    f = gsw.f(lat)
    # set intital conditions
    
    # create time vectors
    time = np.arange(0, runtime, tstep)
    m_all = np.full_like(time, np.nan, dtype=float)
    x_all = np.full_like(time, np.nan, dtype=float)
    z_all = np.full_like(time, np.nan, dtype=float)
    omega_all = []
    DUDZ = np.gradient(flow)
    # run wavenmber derivative first
    for i, tIn in enumerate(time):
        dudz = meanFlow(DUDZ, -z, z_grid)
        m_step = -1*dudz*k*tstep
        z_step = (((omega_int**2 - f**2)/omega_int)*(-m/(k**2+m**2))) * tstep
        x_step = (((omega_int**2 - f**2)/omega_int)*\
                  (-((m**2)/k)/(k**2+m**2)) + dudz) * tstep
#        z_step = -m*omega_int/k**2 + w0
        m -= m_step
        x -= x_step
        z -= z_step
        
        m_all[i] = m
        x_all[i] = x
        z_all[i] = z
        omega_all.append(omega_int)
            
        omega_int = omega - k*meanFlow(flow, -z, z_grid)
        
    x_all = x_all*1e-3
    omega_all = np.array(omega_all)
    if plots:
        plt.figure()
        plt.plot(x_all,z_all)
        plt.gca().invert_yaxis()
        plt.xlabel('Horizontal Distance (km)')
        plt.ylabel('depth (m)')
    
    return x_all, omega_all, z_all, time, m_all


def invert_depth_grid(bottom, z_grid):
    """
    Inverts Depth Grid so that pressure (dB) depths convert to height above
    seafloor
    """
    
    if bottom < 0:
        bottom = -bottom
    
    bottom = float(bottom)
    
    # Find where depth grid meet seafloor
    zrev = bottom-z_grid
    below_ground = zrev <= 0
    
    # remove depths below seafloor
    zrev[below_ground] = np.nan
    
    return zrev

def ray_trace_jones_top_down(U, V, z_grid, N2,\
                             p_grid, k0, m0, z0, omega, lat,\
                             x0=0, tstep=5, runtime=5, plots=True):

    """
    Ray tracing using equations from Jones 1969 with z=0 at surface (top down)
    """
    
    runtime = runtime*60*60 #Convert into seconds from hours
    x = x0
    z = z0 # z has to be negative of 0 at surface (I THINK?)
    k = k0
#    if m0 > 0:
#        m0 = -m0
    m = -m0
    
    
    U0 = flow_alignment(U, V)
    
    # Doppler Shifting Frequency
    omega_int = omega - k*meanFlow(U0, z, z_grid)
    
    f = gsw.f(lat)
    # set intital conditions
    
    # create time vectors
    time = np.arange(0, runtime, tstep)
    m_all = np.full_like(time, np.nan, dtype=float)
    x_all = np.full_like(time, np.nan, dtype=float)
    z_all = np.full_like(time, np.nan, dtype=float)
    omega_all = np.full_like(time, np.nan, dtype=float)
    DUDZ = np.gradient(U0)
    for i, dump in enumerate(time):
        dudz = f_of_z(DUDZ, z, z_grid)
        n2 = f_of_z(N2, z, p_grid)
        Uz = f_of_z(U0, z, z_grid)
        Om = omega - k*Uz
        if np.abs(Om) < np.abs(f):
            break
        
        xstep = ((n2 - Om**2)/(Om*(k**2+m**2)))*k + Uz
        if np.isfinite(x) and np.isfinite(xstep): 
            xstep = xstep*tstep
            x -= xstep
            
            zstep = (-m*Om)/(k**2+m**2)
            zstep = zstep*tstep
            z -= zstep
    
            mstep = -k*dudz
            mstep = mstep*tstep
            m -= mstep
        
        
        omega_all[i] = Om
        z_all[i] = z
        x_all[i] = x
        m_all[i] = m
        
        
    x_all = x_all*1e-3
    
    return x_all, z_all, omega_all, m_all



def seafloor_grid(depths, lat, lon):
    """
    Function for reinterpolating seafloor grid onto a finer grid for 
    ray tracing
    """
    
    

def ray_trace_jones_bottom_up_test():
    """
    Ray tracing using equations from Jones 1969 with z=o at seafloor
    (testing whether it makes a difference compared to top down ray tracting)
    """
    
    
    


def ray_trace_jones_top_down_test():
    """
    Ray tracing using equations from Jones 1969 with z=0 at surface
    (testing whether it makes a difference compared to bottom up ray tracting)
    """
    
    # Load Initial Conditions for testing
     
    U, V, z_grid, k0, m0, z0, x0,\
        lat, tstep, runtime, omega, N2, p_grid = initial_conditions()
    
    # Set initial parameters
    runtime = runtime*60*60 #Convert into seconds from hours
    x = x0
    z = z0 # z has to be negative of 0 at surface (I THINK?)
    k = k0
    m = -m0
    
    U0 = flow_alignment(U, V)
    
    # Doppler Shifting Frequency
    omega_int = omega - k*meanFlow(U0, z, z_grid)
    
    f = gsw.f(lat)
    # set intital conditions
    
    # create time vectors
    time = np.arange(0, runtime, tstep)
    m_all = np.full_like(time, np.nan, dtype=float)
    x_all = np.full_like(time, np.nan, dtype=float)
    z_all = np.full_like(time, np.nan, dtype=float)
    omega_all = np.full_like(time, np.nan, dtype=float)
    DUDZ = np.gradient(U0)
    count = 0
    for i, dump in enumerate(time):
        dudz = f_of_z(DUDZ, z, z_grid)
        n2 = f_of_z(N2, z, p_grid)
        Uz = f_of_z(U0, z, z_grid)
        Om = omega - k*Uz
        if np.abs(Om) < np.abs(f):
            break
        
        xstep = ((n2 - Om**2)/(Om*(k**2+m**2)))*k + Uz
        if np.isfinite(x) and np.isfinite(xstep): 
            xstep = xstep*tstep
            x -= xstep
            
            zstep = (-m*Om)/(k**2+m**2)
            zstep = zstep*tstep
            z -= zstep
    
            mstep = -k*dudz
            mstep = mstep*tstep
            m -= mstep
        
        
        omega_all[i] = Om
        z_all[i] = z
        x_all[i] = x
        m_all[i] = m
        
        count += 1
        
    x_all = x_all*1e-3
        
    if plots:
        plt.figure()
        plt.plot(x_all,z_all)
        plt.gca().invert_yaxis()
        plt.xlabel('Horizontal Distance (km)')
        plt.ylabel('depth (m)')
        
        plt.figure()
        plt.plot(x_all, m_all/(2*np.pi))
        plt.xlabel('Horizontal Distance (km)')
        plt.ylabel('vertical wavelength')
        
        fgrid = np.full_like(x_all, f)
        n2_grid = np.full_like(x_all, np.nanmean(np.sqrt(N2)))
        plt.figure()
        plt.plot(x_all,omega_all,'g')
        plt.plot(x_all, np.abs(fgrid), 'r')
        plt.plot(x_all , n2_grid, 'b')
        plt.xlabel('Horizontal Distance (km)')
        plt.ylabel('frequency')


        
        
    

    
    

def raytrace_tester(plots=True):
    
    
    U, V, z_grid, k0, m0, z0, x0, lat, tstep, runtime, omega, N2, dump = initial_conditions()
    
    runtime = runtime*60*60 #Convert into seconds from hours
    x = x0
    z = -z0
    k = k0
    m = -m0
    
    # dkdt = 0 # Not actually necessary to declare this but useful reminder
    # dldt = 0 # Same as above
    flow = flow_alignment(U, V)
    
    # Doppler Shifting Frequency
    omega_int = omega - k*meanFlow(flow, z, z_grid)
    
    f = gsw.f(lat)
    # set intital conditions
    
    # create time vectors
    time = np.arange(0, runtime, tstep)
    m_all = np.full_like(time, np.nan, dtype=float)
    x_all = np.full_like(time, np.nan, dtype=float)
    z_all = np.full_like(time, np.nan, dtype=float)
    omega_all = []
    DUDZ = np.gradient(flow)
    
    # run wavenmber derivative first
    for i, tIn in enumerate(time):
        dudz = meanFlow(DUDZ, -z, z_grid)
        m_step = -1*dudz*k*tstep
        z_step = (((omega_int**2 - f**2)/omega_int)*(-m/(k**2+m**2))) * tstep
        x_step = (((omega_int**2 - f**2)/omega_int)*\
                  (-((m**2)/k)/(k**2+m**2)) + dudz) * tstep
#        z_step = -m*omega_int/k**2 + w0
        m -= m_step
        x -= x_step
        z -= z_step
        m_all[i] = m
        x_all[i] = x
        z_all[i] = z
        omega_all.append(omega_int)

        
    x_all = x_all*1e-3 # convert to kilometers
    
    # Plot path in x-z axis
    if plots:
        plt.figure()
        plt.plot(x_all,z_all)
        plt.gca().invert_yaxis()
        plt.xlabel('Horizontal Distance (km)')
        plt.ylabel('depth (m)')
        
        plt.figure()
        plt.plot(x_all, m_all/(2*np.pi))
        plt.xlabel('Horizontal Distance (km)')
        plt.ylabel('vertical wavelength')
        
        fgrid = np.full_like(x_all, f)
        plt.figure()
        plt.plot(x_all,omega_all)
        plot(x_all, fgrid)
        plt.xlabel('Horizontal Distance (km)')
        plt.ylabel('frequency')
    
    all_data = {'x':x_all, 'z': z_all}
    
    return all_data





def ray_trace_full_set_jones(tstep=5, runtime=24, m0=(2*np.pi)/250, plots=True):
    """
    ray trace all of the data using top down Jones equations WILL TAKE FOREVER 
    SO BE CAREFULL
    """
    ladcp, ctd, bathy = data_load.load_data()
    U, V, z_grid = oc.loadLADCP(ladcp)
    S, T, p, lat, lon = oc.loadCTD(ctd)
    N2 = oc.gswN2(S, T, p, lat, lon)
    for i, cast in enumerate(N2.T):
        N2[:,i] = oc.verticalBoxFilter1(cast, p[:,i])
    # Load Data
    lambdaH = pd.read_excel('lambdaH.xlsx')
    kh = pd.read_excel('Kh_masked.xlsx')
    omega = pd.read_excel('omega_masked.xlsx')
    # Depth grid stored as index in pandas dataframes
    depths = np.array(omega.index)
    X = pd.DataFrame(index=depths, columns=np.arange(0,21))
    Z = pd.DataFrame(index=depths, columns=np.arange(0,21))
    OM = pd.DataFrame(index=depths, columns=np.arange(0,21))
    m = pd.DataFrame(index=depths, columns=np.arange(0,21))
#    time = np.arange(0, runtime, tstep)
    x_all = []
    z_all = []
    Om_all = []
    starts = []
    count=0
    for i in range(kh.shape[1]):
        
        
        for k in range(kh.shape[0]):
            depth = depths[k]
            if np.isfinite(kh.loc[depth][i]) and np.isfinite(omega.loc[depth][i]):
                X.loc[depth][i], Z.loc[depth][i],\
                    OM.loc[depth][i], m.loc[depth][i]\
                        = ray_trace_jones_top_down(U[:,i], V[:,i], z_grid,\
                                N2[:,i], p[:,i], kh.loc[depth][i],\
                                    m0, depth, omega.loc[depth][i], lat[:,i],\
                                    tstep=tstep, runtime=runtime)
                starts.append([i+1, depth])
                x_all.append(X.loc[depth][i])
                z_all.append(Z.loc[depth][i])
                Om_all.append(OM.loc[depth][i])
                
            else:
                 X.loc[depth][i] = np.nan
                 OM.loc[depth][i] = np.nan
                 Z.loc[depth][i] = np.nan
                 m.loc[depth][i] = np.nan
        count +=1
        print(count)
    
    
    x_all = np.vstack(x_all)
    z_all = np.vstack(z_all)
    Om_all = np.vstack(Om_all)
    starts = np.vstack(starts)
    
    np.savetxt('x_ray_trace.csv', x_all)
    np.savetxt('z_ray_trace.csv', z_all)
    np.savetxt('Om_ray_trace.csv', Om_all)
    np.savetxt('starts_ray_trace.csv', starts)
    
    
    
    
    # Plotting Data
    if plots:
        fig = plt.figure()
        for i in range(kh.shape[1]):
            for k in range(kh.shape[0]):
                depth = depths[k]
                idx = i
                plt.plot(X.loc[depth][idx],Z.loc[depth][idx])
                plt.xlabel('Horizontal Distance (km)')
                plt.ylabel('depth (m)')
        plt.gca().invert_yaxis()
        
                
        fig = plt.figure()
        for i in range(kh.shape[1]):
            for k in range(kh.shape[0]):
                depth = depths[k]
                idx = i
                plt.plot(X.loc[depth][idx],m.loc[depth][idx]/(2*np.pi))
                plt.xlabel('Horizontal Distance (km)')
                plt.ylabel('vertical wavenumber')
        
        
        fig = plt.figure()
        for i in range(kh.shape[1]):
            for k in range(kh.shape[0]):
                depth = depths[k]
                idx = i
                plt.plot(X.loc[depth][idx],OM.loc[depth][idx])
                plt.xlabel('Horizontal Distance (km)')
                plt.ylabel('frequency ')
        
        
        
    
    
    return x_all, z_all, Om_all, starts



def ray_trace_full_set(tstep=5, runtime=5, m0=(2*np.pi)/250):
    """
    DO NOT RUN UNLESS YOU HAVE FREE TIME THIS TAKES FOREVER!
    """
    ladcp, ctd, bathy = data_load.load_data()
    U, V, z_grid = oc.loadLADCP(ladcp)
    S, T, p, lat, lon = oc.loadCTD(ctd)
    # Load Data
    lambdaH = pd.read_excel('lambdaH.xlsx')
    kh = pd.read_excel('Kh_masked.xlsx')
    omega = pd.read_excel('omega_masked.xlsx')
    # Depth grid stored as index in pandas dataframes
    depths = np.array(omega.index)
    X = pd.DataFrame(index=depths, columns=np.arange(1,22))
    Z = pd.DataFrame(index=depths, columns=np.arange(1,22))
    M = pd.DataFrame(index=depths, columns=np.arange(1,22))
    m = pd.DataFrame(index=depths, columns=np.arange(1,22))
    time = np.arange(0, runtime, tstep)
    count = 0
    for i in range(kh.shape[1]):
        
        
        for k in range(kh.shape[0]):
            depth = depths[k]
            if np.isfinite(kh.loc[depth][i]) and np.isfinite(omega.loc[depth][i]):
                X.loc[depth][i], M.loc[depth][i],\
                    Z.loc[depth][i], time, m.loc[depth][i] = RT_backtrace_simplified(\
                             depth, kh.loc[depth][i], m0,\
                             U[:,i], V[:,i], omega.loc[depth][i],\
                             z_grid, lat[:,i], lon[:,i])
            else:
                 X.loc[depth][i] = np.nan
                 M.loc[depth][i] = np.nan
                 Z.loc[depth][i] = np.nan
                 m.loc[depth][i] = np.nan
        count +=1
        print(count)
#        
#    store = pd.HDFStore('ray_traced_paths.h5')
#    store['x']= X
#    store['z']= Z
    
    fig = plt.figure()
    for i in range(kh.shape[1]):
        for k in range(kh.shape[0]):
            depth = depths[k]
            idx = i+1
            plt.plot(X.loc[depth][idx],Z.loc[depth][idx])
            plt.xlabel('Horizontal Distance (km)')
            plt.ylabel('depth (m)')
    plt.gca().invert_yaxis()
    
            
    fig = plt.figure()
    for i in range(kh.shape[1]):
        for k in range(kh.shape[0]):
            depth = depths[k]
            idx = i+1
            plt.plot(X.loc[depth][idx],m.loc[depth][idx]/(2*np.pi))
            plt.xlabel('vertical_wavenumber')
            plt.ylabel('depth (m)')
    
    
    fig = plt.figure()
    for i in range(kh.shape[1]):
        for k in range(kh.shape[0]):
            depth = depths[k]
            idx = i+1
            plt.plot(X.loc[depth][idx],M.loc[depth][idx])
            plt.xlabel('frequency')
            plt.ylabel('depth (m)')
    
    
    
    
    
    
    
    