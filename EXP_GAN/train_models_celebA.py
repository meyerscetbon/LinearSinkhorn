import random
from scipy.special import lambertw
import numpy as np
import torch
import timeit
import os

import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils


import models
import data_loading
import torch_lin_sinkhorn


def compute_constants(reg, device, nz, R=1, num_random_samples=100, seed=49):
    q = (1 / 2) + (R ** 2) / reg
    y = R ** 2 / (reg * nz)
    q = np.real((1 / 2) * np.exp(lambertw(y)))

    C = (2 * q) ** (nz / 4)

    np.random.seed(seed)
    var = (q * reg) / 4
    U = np.random.multivariate_normal(
        np.zeros(nz), var * np.eye(nz), num_random_samples
    )
    U = torch.from_numpy(U)

    U_init = U.to(device)
    C_init = torch.DoubleTensor([C]).to(device)
    q_init = torch.DoubleTensor([q]).to(device)

    return q_init, C_init, U_init


def training_func(
    num_random_samples,
    reg,
    batch_size,
    niter_sin,
    image_size,
    nc,
    nz,
    dataset_name,
    device,
    manual_seed,
    lr,
    max_iter,
    data_root,
    R,
    random_,
):
    name_dir = "sampled_images_celebA" + "_" + str(num_random_samples) + "_" + str(reg)
    if os.path.exists(name_dir) == 0:
        os.mkdir(name_dir)

    epsilon = reg
    hidden_dim = nz

    # Create an output file
    file_to_print = open(
        "results_training_celebA"
        + "_"
        + str(num_random_samples)
        + "_"
        + str(reg)
        + ".csv",
        "w",
    )
    file_to_print.write(str(device) + "\n")
    file_to_print.flush()

    # Fix the seed
    np.random.seed(seed=manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    cudnn.benchmark = True

    # Initialisation of weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find("Linear") != -1:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0)

    trn_dataset = data_loading.get_data(
        image_size, dataset_name, data_root, train_flag=True
    )
    trn_loader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )

    # construct Generator and Embedding
    q, C, U_init = compute_constants(
        reg, device, nz, R=R, num_random_samples=num_random_samples, seed=manual_seed
    )
    G_generator = models.Generator(image_size, nc, k=nz, ngf=64)
    D_embedding = models.Embedding(
        image_size,
        nc,
        reg,
        device,
        q,
        C,
        U_init,
        k=hidden_dim,
        num_random_samples=num_random_samples,
        R=R,
        seed=manual_seed,
        ndf=64,
        random=random_,
    )

    netG = models.NetG(G_generator)
    netE = models.NetE(D_embedding)

    netG.apply(weights_init)
    netE.apply(weights_init)

    netG.to(device)
    netE.to(device)

    lin_Sinkhorn_AD = torch_lin_sinkhorn.Lin_Sinkhorn_AD.apply
    fixed_noise = torch.DoubleTensor(64, nz, 1, 1).normal_(0, 1).to(device)
    one = torch.tensor(1, dtype=torch.float).double()
    mone = one * -1

    # setup optimizer
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lr)
    optimizerE = torch.optim.RMSprop(netE.parameters(), lr=lr)

    time = timeit.default_timer()
    gen_iterations = 0

    for t in range(max_iter):
        data_iter = iter(trn_loader)
        i = 0
        while i < len(trn_loader):
            # ---------------------------
            #        Optimize over NetE
            # ---------------------------
            for p in netE.parameters():
                p.requires_grad = True

            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 10  # 10
                Giters = 1
            else:
                Diters = 1  # 5
                Giters = 1

            for j in range(Diters):
                if i == len(trn_loader):
                    break

                for p in netE.parameters():
                    p.data.clamp_(-0.01, 0.01)  # clamp parameters of NetE to a cube

                data = data_iter.next()
                i += 1
                netE.zero_grad()

                x_cpu, _ = data
                x = x_cpu.to(device)
                x_emb = netE(x)

                noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1).to(device)
                with torch.no_grad():
                    y = netG(noise)

                y_emb = netE(y)

                # Compute the loss
                sink_E = (
                    2 * lin_Sinkhorn_AD(x_emb, y_emb, epsilon, niter_sin)
                    - lin_Sinkhorn_AD(y_emb, y_emb, epsilon, niter_sin)
                    - lin_Sinkhorn_AD(x_emb, x_emb, epsilon, niter_sin)
                )

                sink_E.backward(mone)
                optimizerE.step()

            # ---------------------------
            #        Optimize over NetG
            # ---------------------------
            for p in netE.parameters():
                p.requires_grad = False

            for j in range(Giters):
                if i == len(trn_loader):
                    break

                data = data_iter.next()
                i += 1
                netG.zero_grad()

                x_cpu, _ = data
                x = x_cpu.to(device)
                x_emb = netE(x)

                noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1).to(device)
                y = netG(noise)
                y_emb = netE(y)

                # Compute the loss
                sink_G = (
                    2 * lin_Sinkhorn_AD(x_emb, y_emb, epsilon, niter_sin)
                    - lin_Sinkhorn_AD(y_emb, y_emb, epsilon, niter_sin)
                    - lin_Sinkhorn_AD(x_emb, x_emb, epsilon, niter_sin)
                )

                sink_G.backward(one)
                optimizerG.step()

                gen_iterations += 1

            run_time = (timeit.default_timer() - time) / 60.0

            s = "[%3d / %3d] [%3d / %3d] [%5d] (%.2f m) loss_E: %.6f loss_G: %.6f" % (
                t,
                max_iter,
                i * batch_size,
                batch_size * len(trn_loader),
                gen_iterations,
                run_time,
                sink_E,
                sink_G,
            )

            s = s + "\n"
            file_to_print.write(s)
            file_to_print.flush()

            if gen_iterations % 100 == 0:
                with torch.no_grad():
                    fixed_noise = fixed_noise.float()
                    y_fixed = netG(fixed_noise)
                    y_fixed = y_fixed.mul(0.5).add(0.5)
                    vutils.save_image(
                        y_fixed,
                        "{0}/fake_samples_{1}.png".format(name_dir, gen_iterations),
                    )

        if t % 10 == 0:
            torch.save(
                netG.state_dict(),
                "netG_celebA" + "_" + str(num_random_samples) + "_" + str(reg) + ".pth",
            )
            torch.save(
                netE.state_dict(),
                "netE_celebA" + "_" + str(num_random_samples) + "_" + str(reg) + ".pth",
            )


# Dataset
image_size = 64
nc = 3
nz = 128
dataset_name = "celeba"
data_root = "./data"

# Parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_ = False
manual_seed = 49
lr = 5 * 1e-5
R = 1
max_iter = 10000
niter_sin = 1000
batch_size = 8000


num_random_samples_list = [10, 100, 300, 600]
reg_list = [1e-1, 1, 10]


if __name__ == "__main__":
    for num_random_samples in num_random_samples_list:
        for reg in reg_list:
            training_func(
                num_random_samples,
                reg,
                batch_size,
                niter_sin,
                image_size,
                nc,
                nz,
                dataset_name,
                device,
                manual_seed,
                lr,
                max_iter,
                data_root,
                R,
                random_,
            )
