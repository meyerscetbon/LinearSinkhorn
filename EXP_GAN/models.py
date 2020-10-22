import torch.nn as nn
import torch


# input: batch_size * nc * image_size * image_size
# output: batch_size * num_random_samples * 1 * 1
class Embedding(nn.Module):
    def __init__(
        self,
        isize,
        nc,
        reg,
        device,
        q,
        C,
        U_init,
        k=100,
        num_random_samples=100,
        R=1,
        ndf=64,
        seed=49,
        random=False,
    ):
        super(Embedding, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # input is nc x isize x isize
        main = nn.Sequential()
        main.add_module(
            "initial_conv_{0}-{1}".format(nc, ndf),
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        )
        main.add_module("initial_relu_{0}".format(ndf), nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module(
                "pyramid_{0}-{1}_conv".format(in_feat, out_feat),
                nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False),
            )
            main.add_module(
                "pyramid_{0}_batchnorm".format(out_feat), nn.BatchNorm2d(out_feat)
            )
            main.add_module(
                "pyramid_{0}_relu".format(out_feat), nn.LeakyReLU(0.2, inplace=True)
            )
            cndf = cndf * 2
            csize = csize / 2

        main.add_module(
            "final_{0}-{1}_conv".format(cndf, 1),
            nn.Conv2d(cndf, k, 4, 1, 0, bias=False),
        )

        self.main = main

        if random == False:
            U = torch.nn.Parameter(U_init)

        else:
            U = U_init

        self.U = U.type(torch.DoubleTensor)
        self.q = q.type(torch.DoubleTensor)
        self.C = C.type(torch.DoubleTensor)
        self.reg = reg
        self.num_random_samples = num_random_samples

    # X and Y are 2D tensors
    def Square_Euclidean_Distance(self, X, Y):
        X_col = X.unsqueeze(1).type(torch.DoubleTensor)
        Y_lin = Y.unsqueeze(0).type(torch.DoubleTensor)
        C = torch.sum((X_col - Y_lin) ** 2, 2)
        return C

    # input: batch_size * k * 1 * 1
    # output: batch_size * num_random_samples * 1 * 1
    def Feature_Map_Gaussian(self, X):
        X = X.squeeze()
        batch_size, dim = X.size()

        SED = self.Square_Euclidean_Distance(X, self.U)
        W = -(2 * SED) / self.reg
        Z = self.U ** 2
        A = torch.sum(Z, 1)
        a = self.reg * self.q
        V = A / a

        res_trans = V + W
        res_trans = self.C * torch.exp(res_trans)

        res = (
            1 / torch.sqrt(torch.DoubleTensor([self.num_random_samples]))
        ) * res_trans
        res = res.view(batch_size, self.num_random_samples, 1, 1)

        return res

    def forward(self, input):
        output = self.main(input)
        output = self.Feature_Map_Gaussian(output)

        return output


# input: batch_size * k * 1 * 1
# output: batch_size * nc * image_size * image_size
class Generator(nn.Module):
    def __init__(self, isize, nc, k=100, ngf=64):
        super(Generator, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module(
            "initial_{0}-{1}_convt".format(k, cngf),
            nn.ConvTranspose2d(k, cngf, 4, 1, 0, bias=False),
        )
        main.add_module("initial_{0}_batchnorm".format(cngf), nn.BatchNorm2d(cngf))
        main.add_module("initial_{0}_relu".format(cngf), nn.ReLU(True))

        csize = 4
        while csize < isize // 2:
            main.add_module(
                "pyramid_{0}-{1}_convt".format(cngf, cngf // 2),
                nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False),
            )
            main.add_module(
                "pyramid_{0}_batchnorm".format(cngf // 2), nn.BatchNorm2d(cngf // 2)
            )
            main.add_module("pyramid_{0}_relu".format(cngf // 2), nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module(
            "final_{0}-{1}_convt".format(cngf, nc),
            nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False),
        )
        main.add_module("final_{0}_tanh".format(nc), nn.Tanh())

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output


# input: batch_size * nz * 1 * 1
# output: batch_size * nc * image_size * image_size
class NetG(nn.Module):
    def __init__(self, decoder):
        super(NetG, self).__init__()
        self.decoder = decoder

    def forward(self, input):
        output = self.decoder(input)
        return output


# input: batch_size * nc * image_size * image_size
# f_emb: batch_size * k * 1 * 1
class NetE(nn.Module):
    def __init__(self, embedding):
        super(NetE, self).__init__()
        self.embedding = embedding

    def forward(self, input):
        f_emb = self.embedding(input)
        f_emb = f_emb.view(input.size(0), -1)

        return f_emb
