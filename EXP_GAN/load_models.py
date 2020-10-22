import random
from scipy.special import lambertw
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils

import models


# CIFAR10
# image_size = 64
# nc = 3
# nz = 128
# dataset_name = 'cifar10'

# CELEBA
dataset_name = 'celeba'
image_size = 64
nc = 3
nz = 128


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available(), device)

manual_seed = 49
random_ = False
R=1
batch_size = 8000


num_random_samples = 600
reg = 1

epsilon = reg
hidden_dim = nz

# Fix the seed
np.random.seed(seed=manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
cudnn.benchmark = True


def compute_constants(reg,device,nz,R=1,num_random_samples=100,seed=49):
    q = (1/2) +  (R**2) / reg
    y = R ** 2 / ( reg * nz )
    q = np.real( (1/2) * np.exp( lambertw(y) ) )

    C = (2 * q) ** (nz/4)

    np.random.seed(seed)
    var = ( q * reg ) / 4
    U = np.random.multivariate_normal(np.zeros(nz), var * np.eye(nz), num_random_samples)
    U = torch.from_numpy(U)

    U_init = U.to(device)
    C_init = torch.DoubleTensor([C]).to(device)
    q_init = torch.DoubleTensor([q]).to(device)

    return q_init, C_init, U_init


q, C, U_init = compute_constants(reg,device,nz,R=R,num_random_samples=num_random_samples,seed=manual_seed)
q, C, U_init = q.to(device), C.to(device), U_init.to(device)

G_generator = models.Generator(image_size,nc, k=nz, ngf=64)
D_embedding = models.Embedding(image_size,nc,reg,device,q,C,U_init,k=hidden_dim,
        num_random_samples=num_random_samples,R=R,seed=manual_seed,ndf=64,random=random_)

netG = models.NetG(G_generator)
path_model_G = 'netG_celebA_600_1.pth'
netG.load_state_dict(torch.load(path_model_G, map_location="cpu")) # If on cluster comment map_location
netG.to(device)

netE = models.NetE(D_embedding)
path_model_E = 'netE_cifar_max_600_1.pth'
netE.load_state_dict(torch.load(path_model_E, map_location="cpu")) # If on cluster comment map_location
netE.to(device)


# Choose a random seed to sample a random image from the generator
manual_seed = 123
np.random.seed(seed=manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
cudnn.benchmark = True


batch_size_noise = 32
fixed_noise = torch.DoubleTensor(batch_size_noise, nz, 1, 1).normal_(0, 1).to(device)
fixed_noise = fixed_noise.float()
y_fixed = netG(fixed_noise) # between -1 and 1

# A sample from the trained model
y_fixed = y_fixed.mul(0.5).add(0.5)
vutils.save_image(y_fixed,'celebA_image_vf_600.png')
