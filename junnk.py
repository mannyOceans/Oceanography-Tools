#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:29:35 2017

@author: manishdevana
"""

fig = plt.figure()
plt.plot(U[:,0], z, label='original')
plt.plot(Upoly[:,0], z, label='polyfit')
plt.plot(Umasked[:,0], z, label='residual')

plt.gca().invert_yaxis()
plt.legend()


fig = plt.figure()
plt.plot(V[:,0], z, label='original')
plt.plot(Vpoly[:,0], z, label='polyfit')
plt.plot(Vmasked[:,0], z, label='residual')

plt.gca().invert_yaxis()
plt.legend()


fig = plt.figure()
for cast, z in zip(N2.T, p_ctd.T):
    plt.plot(np.log10(cast), z, label='original')
plt.gca().invert_yaxis()
fig1 = plt.figure()
for cast, z in zip(N2ref.T, p_ctd.T):
    plt.plot(np.log10(cast), z, label='original')
plt.gca().invert_yaxis()
    
plt.plot(Vpoly[:,0], z, label='polyfit')
plt.plot(Vmasked[:,0], z, label='residual')

plt.gca().invert_yaxis()
plt.legend()


test = scipy.fftpack.fft2(bathyrev)
test2 = np.abs(test)/128/89

fig2 = plt.figure()
plt.contourf(dist, z, rho_neutral)
plt.title('reference')
plt.colorbar()
plt.gca().invert_yaxis()

fig = plt.figure()
plt.contourf(dist, np.squeeze(depths), np.abs(omega))
plt.colorbar()
plt.gca().invert_yaxis()


fig = plt.figure()
plt.contourf(xgrid, zgrid, U2)
plt.colorbar()

for station in lambdaH.T: 
    plt.hexbin(lambdaH[:,1], depths2[:,1])

sns.jointplot(x=lambdaH, y=depths2, kind="hex", color="k");

bathy, longrid, latgrid = oc.bathyLoadNc(bathy_file,\
                                          lat,\
                                          lon,\
                                          add_buffer=True,\
                                          buffer=.5)


ground = []
for loni, lati in zip(np.squeeze(lon), np.squeeze(lat)):
    latidx = np.nanargmin(np.abs(lati-latgrid[:,0]))
    lonidx = np.nanargmin(np.abs(loni -longrid[0,:]))
    ground.append(bathy[lonidx,latidx])
    
ground = np.hstack(ground)
    
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.squeeze(lon), np.squeeze(lat), ground)
ax.plot_surface(longrid, latgrid, bathy,\
               cmap=cmocean.cm.deep_r,\
                linewidth=0,\
                antialiased=True)

ax.set_xlabel('lon')
ax.set_ylabel('lat')
ax.set_zlabel('depth')


for depth in depths:
    
    Urev, Vrev = oc.speedAtZ(U, V, p_ladcp, depth, bin_width=350)
    np.savetxt('processed_data/Umean_at_'+str(depth)+'meters.txt', Urev)
    np.savetxt('processed_data/Vmean_at_'+str(depth)+'meters.txt', Vrev)
    


rho_neutral =rho_neutral[:,0]
z = p_ctd[:,0]
bin_idx = ctd_bins
ref_rho = ref_rho[:,0]


fig = plt.figure()

for grid, stn in zip(specGrid, PS):
    
    plt.scatter(grid[1:], stn[1:])




