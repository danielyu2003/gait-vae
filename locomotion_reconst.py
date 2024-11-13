import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from physvae import utils
from physvae.locomotion.model import VAE
from torchdiffeq import odeint

datadir = './data/locomotion/'
dataname = 'test'
data_test = sio.loadmat('{}/data_{}.mat'.format(datadir, dataname))['data'].astype(np.float32)
_, dim_x, dim_t = data_test.shape

dim_y = 3
modeldir = './out_locomotion/'

with open('{}/args.json'.format(modeldir), 'r') as f:
    args_tr_dict = json.load(f)

# set model
device = "cpu"
model = VAE(args_tr_dict).to(device)

# load model
model.load_state_dict(torch.load('{}/model.pt'.format(modeldir), map_location=device))
model.eval()
print('model loaded')

data_test_tensor = torch.Tensor(data_test).to(device).contiguous()

# reg
z_phy_stat, z_aux2_stat, init_yy = model.encode(data_test_tensor)
z_phy, z_aux2 = model.draw(z_phy_stat, z_aux2_stat, hard_z=False)
x_PB, x_P, _, _ = model.decode(z_phy, z_aux2, init_yy, full=True)


def ODEfunc(t: torch.Tensor, yy: torch.Tensor):
    return model.physics_model(z_phy, yy)


yy_seq = odeint(ODEfunc, init_yy, model.t_intg, method='dopri5')

idx = 0
dat = data_test[idx].T

# reg
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(dat) # true
# nn+phys+reg
plt.plot(x_PB[idx].detach().cpu().numpy().T, 'k--')
plt.subplot(2, 1, 2)
plt.plot(dat) # true
# nn+phys
plt.plot(x_P[idx].detach().cpu().numpy().T, 'k--')

plt.show()
