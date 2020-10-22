import torch


class Lin_Sinkhorn_AD(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_emb, y_emb, reg, niter_sin, lam=1e-6,tau=1e-9):
        phi_x = x_emb.squeeze().type(torch.DoubleTensor)
        phi_y = y_emb.squeeze().type(torch.DoubleTensor)

        n = phi_x.size()[0]
        m = phi_y.size()[0]

        a = (1. / n) * torch.ones(n)
        a = a.type(torch.DoubleTensor)

        b = (1. / m) * torch.ones(m)
        b = b.type(torch.DoubleTensor)

        actual_nits = 0

        u = 1. * torch.ones(n).type(torch.DoubleTensor)
        v = 1. * torch.ones(m).type(torch.DoubleTensor)
        err = 0.

        u_trans = torch.matmul(phi_x,torch.matmul(phi_y.t(),v)) + lam
        v_trans = torch.matmul(phi_y,torch.matmul(phi_x.t(),u)) + lam

        for k in range(niter_sin):
            u = a / u_trans
            v_trans = torch.matmul(phi_y,torch.matmul(phi_x.t(),u)) + lam

            v = b / v_trans
            u_trans = torch.matmul(phi_x,torch.matmul(phi_y.t(),v)) + lam

            err = torch.sum(torch.abs(u * u_trans - a)) + torch.sum(torch.abs(v *  v_trans - b))

            actual_nits += 1
            if err < tau :
                    break

            if k % 10 == 0:
                ### Stpping Criteria ###s
                with torch.no_grad():
                    err = torch.sum(torch.abs(u * u_trans - a)) + torch.sum(torch.abs(v *  v_trans - b))
                    if err < tau :
                        break

        ctx.u = u
        ctx.v = v
        ctx.reg = reg
        ctx.phi_x = phi_x
        ctx.phi_y = phi_y

        cost =  reg * (torch.sum (a * torch.log(u) ) + torch.sum( b * torch.log(v) ) - 1)
        return cost

    @staticmethod
    def backward(ctx, grad_output):
        u = ctx.u
        v = ctx.v
        reg = ctx.reg
        phi_x = ctx.phi_x
        phi_y = ctx.phi_y

        grad_input = grad_output.clone()
        grad_phi_x = grad_input * torch.matmul(u.view(-1,1),torch.matmul(phi_y.t(),v).view(1,-1)) * ( - reg )
        grad_phi_y = grad_input * torch.matmul(v.view(-1,1),torch.matmul(phi_x.t(),u).view(1,-1)) * ( - reg )

        return grad_phi_x, grad_phi_y, None, None, None, None, None




def Lin_Sinkhorn(phi_x,phi_y,reg,niter_sin,device,lam=1e-6,tau=1e-9,stabilize = False) :
    phi_x = phi_x.squeeze().type(torch.DoubleTensor).to(device)
    phi_y = phi_y.squeeze().type(torch.DoubleTensor).to(device)

    n = phi_x.size()[0]
    m = phi_y.size()[0]

    a = (1. / n) * torch.ones(n)
    a = a.type(torch.DoubleTensor).to(device)

    b = (1. / m) * torch.ones(m)
    b = b.type(torch.DoubleTensor).to(device)

    actual_nits = 0
    if stabilize == True:
        alpha, beta, err = torch.zeros(n).to(device), torch.zeros(m).to(device), 0.
        for i in range(niter_sin) :
            alpha_res = alpha
            beta_res = beta

            lin_M = torch.exp(alpha/reg) * torch.matmul(phi_x,torch.matmul(phi_y.t(),torch.exp(beta/reg)))
            lin_M = lin_M + lam
            alpha =  reg * ( torch.log(a) - torch.log(lin_M)) + alpha

            lin_M_t = torch.exp(beta/reg) * torch.matmul(phi_y,torch.matmul(phi_x.t(),torch.exp(alpha/reg)))
            lin_M_t = lin_M + lam
            beta =  reg * ( torch.log(b) -  torch.log(lin_M_t)) + beta

            err = (alpha - alpha_res).abs().sum() +  (beta - beta_res).abs().sum()

            actual_nits += 1
            if err < tau :
                break
        cost  = torch.sum( a * alpha ) + torch.sum( b * beta )
        print(cost)

    else:
        u = 1. * torch.ones(n).type(torch.DoubleTensor).to(device)
        v = 1. * torch.ones(m).type(torch.DoubleTensor).to(device)
        err = 0.

        u_trans = torch.matmul(phi_x,torch.matmul(phi_y.t(),v)) + lam
        v_trans = torch.matmul(phi_y,torch.matmul(phi_x.t(),u)) + lam


        for k in range(niter_sin):
            u = a / u_trans
            v_trans = torch.matmul(phi_y,torch.matmul(phi_x.t(),u)) + lam

            v = b / v_trans
            u_trans = torch.matmul(phi_x,torch.matmul(phi_y.t(),v)) + lam

            err = torch.sum(torch.abs(u * u_trans - a)) + torch.sum(torch.abs(v *  v_trans - b))

            actual_nits += 1
            if err < tau :
                    break
        cost =  reg * (torch.sum (a * torch.log(u) ) + torch.sum( b * torch.log(v) ) - 1)

    return cost
