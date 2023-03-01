import h5py
import numpy as np
#Copied from Dr. Qiang Zhu at UC Irvine from their issue post
#https://github.com/openmm/openmm/issues/3388
f           = h5py.File('traj.h5','r')
velocities  = np.array(f['velocities'])
coordinates = np.array(f['coordinates'])
print('Velocities')
print(velocities[0][:10])

print('Coordinates')
print(coordinates[0][:10])
