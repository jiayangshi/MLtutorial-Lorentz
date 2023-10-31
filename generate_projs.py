import foam_ct_phantom
import numpy as np
import os
import h5py
import astra
import tomopy
from pcfv import add_possion_noise, cal_attenuation_factor, absorption
from tqdm import tqdm
from tifffile import imwrite


phantom = foam_ct_phantom.FoamPhantom('train_phantom.h5')

num_angles_high = 1024
factor = 2
num_angles_low = num_angles_high//factor

pg_high = foam_ct_phantom.ParallelGeometry(
    nx=256,
    ny=256,
    angles=np.linspace(0, np.pi, num_angles_high, False),
    pixsize=2/256
)

phantom.generate_projections('projections_high.h5', pg_high)

projs_high = foam_ct_phantom.load_projections('projections_high.h5')
os.remove('projections_high.h5')
sinogram_high = projs_high.swapaxes(0,1).copy()
sinogram_low = sinogram_high.copy()[:, ::factor, :]

attenuation_factor = 1.4632 #cal_attenuation_factor(sinogram_low, 50, 0.5) # corresponds to absorption of 50%
sinogram_low *= attenuation_factor
print(f"Absorption: {absorption(sinogram_low)*100:0.0f}%")
sinogram_low = add_possion_noise(sinogram_low, 80)
sinogram_low /= attenuation_factor

# f = h5py.File('projections_low.h5', 'w')
# f.create_dataset("data", data=sinogram_low)
# f = h5py.File('projections_high.h5', 'w')
# f.create_dataset("data", data=sinogram_high)

# np.save('projections_high.npy', sinogram_high)
# np.save('projections_low.npy', sinogram_low)

angles_high = np.linspace(0, np.pi, num_angles_high, endpoint=False)
vg_high = astra.create_vol_geom(256, 256)
pg_high = astra.create_proj_geom('parallel', 1, 256, angles_high)
proj_id_high = astra.create_projector('cuda', pg_high, vg_high)
W_high = astra.OpTomo(proj_id_high)

angles_low = np.linspace(0, np.pi, num_angles_low, endpoint=False)
vg_low = astra.create_vol_geom(256, 256)
pg_low = astra.create_proj_geom('parallel', 1, 256, angles_low)
proj_id_low = astra.create_projector('cuda', pg_low, vg_low)
W_low = astra.OpTomo(proj_id_low)

recon_high = np.zeros((256, 256, 256), dtype=np.float32)
recon_low = np.zeros((256, 256, 256), dtype=np.float32)

for i in tqdm(range(256)):
    recon_high[i] = W_high.reconstruct('FBP_CUDA',sinogram_high[i])
    recon_low[i] = W_low.reconstruct('FBP_CUDA',sinogram_low[i])

recon_high = tomopy.circ_mask(recon_high, axis=0, ratio=1)
recon_low = tomopy.circ_mask(recon_low, axis=0, ratio=1)

# normalize value range for easy use and demonstration purpose
vmax = recon_high.max()
recon_high[recon_high<0] = 0
recon_high = recon_high/vmax

recon_low[recon_low<0] = 0
recon_low = recon_low/vmax
recon_low[recon_low>1] = 1

np.save('recon_high.npy', recon_high)
np.save('recon_low.npy', recon_low)
imwrite('recon_high.tif', recon_high)
imwrite('recon_low.tif', recon_low)