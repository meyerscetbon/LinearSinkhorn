import numpy as np
import scipy
import time
from scipy import special

# Here the Regularized version goes from -\infinty to the true OT
def compute_ROT(u, v, a, b, reg):
    res = reg * (np.dot(a, np.log(u)) + np.dot(b, np.log(v)))
    return res


################ Classical Sinkhorn Algorithm ####################


def Sinkhorn(C, reg, a, b, delta=1e-9, lam=1e-6):

    K = np.exp(-C / reg)
    u = np.ones(np.shape(a)[0])
    v = np.ones(np.shape(b)[0])

    u_trans = np.dot(K, v) + lam  # add regularization to avoid divide 0
    v_trans = np.dot(K.T, u) + lam  # add regularization to avoid divide 0

    err_1 = np.sum(np.abs(u * u_trans - a))
    err_2 = np.sum(np.abs(v * v_trans - b))

    while True:
        if err_1 + err_2 > delta:
            u = a / u_trans
            v_trans = np.dot(K.T, u) + lam

            v = b / v_trans
            u_trans = np.dot(K, v) + lam

            err_1 = np.sum(np.abs(u * u_trans - a))
            err_2 = np.sum(np.abs(v * v_trans - b))
        else:
            return u, v


# Classical Sinkhorn algorithm: Square Euclidean Distance
def Sinkhorn_RBF(X, Y, reg, a, b, delta=1e-9, num_iter=50, lam=1e-100):
    start = time.time()

    acc = []
    times = []

    C = Square_Euclidean_Distance(X, Y)
    K = np.exp(-C / reg)
    u = np.ones(np.shape(a)[0])
    v = np.ones(np.shape(b)[0])

    u_trans = np.dot(K, v) + lam
    v_trans = np.dot(K.T, u) + lam

    for k in range(num_iter):

        u = a / u_trans
        v_trans = np.dot(K.T, u) + lam

        v = b / v_trans
        u_trans = np.dot(K, v) + lam

        ROT_trans = compute_ROT(u, v, a, b, reg)
        if np.isnan(ROT_trans) == True:
            return "Error"
        else:
            acc.append(compute_ROT(u, v, a, b, reg))
            end = time.time()
            times.append(end - start)

    return acc[-1], np.array(acc), np.array(times)


################ Positive Random Features ####################

# Positive Random Features Sinkhorn: K = AB
def Lin_Sinkhorn(A, B, a, b, delta=1e-9, max_iter=1e5, lam=1e-100):
    u = np.ones(np.shape(a)[0])
    v = np.ones(np.shape(b)[0])
    u_trans = np.dot(A, np.dot(B, v)) + lam
    v_trans = np.dot(B.T, np.dot(A.T, u)) + lam

    err_1 = np.sum(np.abs(u * u_trans - a))
    err_2 = np.sum(np.abs(v * v_trans - b))
    k = 0
    while True and k < max_iter:
        if err_1 + err_2 > delta:
            u = a / u_trans
            v_trans = np.dot(B.T, np.dot(A.T, u)) + lam

            v = b / v_trans
            u_trans = np.dot(A, np.dot(B, v)) + lam

            err_1 = np.sum(np.abs(u * u_trans - a))
            err_2 = np.sum(np.abs(v * v_trans - b))
            k = k + 1
        else:
            return u, v
    return u, v


# Positive Random Features Sinkhorn: Square Euclidean Distance
def Lin_Sinkhorn_RBF(
    X, Y, reg, a, b, num_samples, seed=49, delta=1e-9, num_iter=50, lam=1e-100
):
    start = time.time()

    acc = []
    times = []

    R = theoritical_R(X, Y)
    A = Feature_Map_Gaussian(X, reg, R=R, num_samples=num_samples, seed=seed)
    B = Feature_Map_Gaussian(Y, reg, R=R, num_samples=num_samples, seed=seed).T

    u = np.ones(np.shape(a)[0])
    v = np.ones(np.shape(b)[0])
    u_trans = np.dot(A, np.dot(B, v)) + lam
    v_trans = np.dot(B.T, np.dot(A.T, u)) + lam

    for k in range(num_iter):
        u = a / u_trans
        v_trans = np.dot(B.T, np.dot(A.T, u)) + lam

        v = b / v_trans
        u_trans = np.dot(A, np.dot(B, v)) + lam

        ROT_trans = compute_ROT(u, v, a, b, reg)
        if np.isnan(ROT_trans) == True:
            return "Error"
        else:
            acc.append(ROT_trans)
            end = time.time()
            times.append(end - start)

    return acc[-1], np.array(acc), np.array(times)


# Random Feature Map: Square Euclidean Distance
def Feature_Map_Gaussian(X, reg, R=1, num_samples=100, seed=49):
    n, d = np.shape(X)

    # q = (1/2) +  (R**2) / reg
    y = R ** 2 / (reg * d)
    q = np.real((1 / 2) * np.exp(special.lambertw(y)))
    C = (2 * q) ** (d / 4)

    var = (q * reg) / 4

    np.random.seed(seed)
    U = np.random.multivariate_normal(np.zeros(d), var * np.eye(d), num_samples)

    SED = Square_Euclidean_Distance(X, U)
    W = -(2 * SED) / reg
    V = np.sum(U ** 2, axis=1) / (reg * q)

    res_trans = V + W
    res_trans = C * np.exp(res_trans)

    res = (1 / np.sqrt(num_samples)) * res_trans

    return res


def theoritical_R(X, Y):
    norm_X = np.linalg.norm(X, axis=1)
    norm_Y = np.linalg.norm(Y, axis=1)
    norm_max = np.maximum(np.max(norm_X), np.max(norm_Y))

    return norm_max


# Random Feature Map: Arccos Kernel
def Feature_Map_Arccos(X, s=1, sig=1.5, num_samples=100, kappa=1e-6, seed=49):
    n, d = np.shape(X)
    C = (sig ** (d / 2)) * np.sqrt(2)

    np.random.seed(seed)
    U = np.random.multivariate_normal(np.zeros(d), (sig ** 2) * np.eye(d), num_samples)

    IP = Inner_Product(X, U)
    res_trans = C * (np.maximum(IP, 0) ** s)

    V = ((sig ** 2) - 1) / (sig ** 2)
    V = -(1 / 4) * V * np.sum(U ** 2, axis=1)
    V = np.exp(V)

    res = np.zeros((n, num_samples + 1))
    res[:, :num_samples] = (1 / np.sqrt(num_samples)) * res_trans * V
    res[:, -1] = kappa

    return res


######################## Nystrom Method #######################

# Nystrom Sinkhorn: K =VA^{-1}V
def Nys_Sinkhorn(A, V, a, b, delta=1e-9, max_iter=1e3, lam=1e-100):
    u = np.ones(np.shape(a)[0])
    v = np.ones(np.shape(b)[0])

    u_trans = np.dot(V, np.linalg.solve(A, np.dot(V.T, v))) + lam
    v_trans = np.dot(V, np.linalg.solve(A, np.dot(V.T, u))) + lam

    err_1 = np.sum(np.abs(u * u_trans - a))
    err_2 = np.sum(np.abs(v * v_trans - b))
    k = 0
    while True and k < max_iter:
        if err_1 + err_2 > delta:
            u = a / u_trans
            v_trans = np.dot(V, np.linalg.solve(A, np.dot(V.T, u))) + lam

            v = b / v_trans
            u_trans = np.dot(V, np.linalg.solve(A, np.dot(V.T, v))) + lam

            err_1 = np.sum(np.abs(u * u_trans - a))
            err_2 = np.sum(np.abs(v * v_trans - b))
            k = k + 1
        else:
            return u, v
    return u, v


# Nystrom Sinkhorn: Square Euclidean Distance
def Nys_Sinkhorn_RBF_inv(
    X, Y, reg, a, b, rank, seed=49, delta=1e-9, num_iter=50, lam=1e-100
):
    start = time.time()

    acc = []
    times = []

    n = np.shape(X)[0]
    m = np.shape(Y)[0]

    a_nys = np.zeros(n + m)
    a_nys[:n] = a

    b_nys = np.zeros(n + m)
    b_nys[n:] = b

    A, V = Nystrom_RBF(X, Y, reg, rank, seed=seed, stable=1e-10)
    A_inv = np.linalg.inv(A)

    u = np.ones(np.shape(a_nys)[0])
    v = np.ones(np.shape(b_nys)[0])

    u_trans = np.dot(V, np.dot(A_inv, np.dot(V.T, v))) + lam
    v_trans = np.dot(V, np.dot(A_inv, np.dot(V.T, u))) + lam

    for k in range(num_iter):

        u = a_nys / u_trans
        v_trans = np.dot(V, np.dot(A_inv, np.dot(V.T, u))) + lam

        v = b_nys / v_trans
        u_trans = np.dot(V, np.dot(A_inv, np.dot(V.T, v))) + lam

        u_rot, v_rot = u[:n], v[n:]

        ROT_trans = compute_ROT(u_rot, v_rot, a, b, reg)
        if np.isnan(ROT_trans) == True:
            return "Error"
        else:
            acc.append(ROT_trans)
            end = time.time()
            times.append(end - start)

    return acc[-1], np.array(acc), np.array(times)


# Uniform Nyström: Square Euclidean Distance
def Nystrom_RBF(X, Y, reg, rank, seed=49, stable=1e-100):
    n, d = np.shape(X)
    m, d = np.shape(Y)
    n_tot = n + m
    Z = np.concatenate((X, Y), axis=0)

    rank_trans = int(np.minimum(rank, n_tot))

    np.random.seed(seed)
    ind = np.random.choice(n_tot, rank_trans, replace=False)
    ind = np.sort(ind)

    Z_1 = Z[ind, :]
    A = np.exp(-Square_Euclidean_Distance(Z_1, Z_1) / reg)
    A = A + stable * np.eye(rank_trans)
    V = np.exp(-Square_Euclidean_Distance(Z, Z_1) / reg)

    return A, V


# Recursive Nyström Sampling: Square Euclidean Distance
def recursive_Nystrom_RBF(X, Y, rank, reg, seed=49, stable=1e-100):
    Z = np.concatenate((X, Y), axis=0)
    n, d = np.shape(Z)

    ## Start of algorithm
    sLevel = rank
    oversamp = np.log(sLevel)
    k = int(sLevel / (4 * oversamp)) + 1
    nLevels = int(np.log(n / sLevel) / np.log(2)) + 1

    np.random(seed)
    perm = np.random.permutation(n)

    # set up sizes for recursive levels
    lSize = np.zeros(nLevels)
    lSize[0] = n
    for i in range(1, nLevels):
        lSize[i] = int(lSize[i - 1] / 2) + 1

    # rInd: indices of points selected at previous level of recursion
    # at the base level it's just a uniform sample of ~sLevel points
    samp = np.arange(lSize[-1]).astype(int)
    rInd = perm[samp]
    weights = np.ones((np.shape(rInd)[0], 1))

    # we need the diagonal of the whole kernel matrix
    kDiag = np.zeros(n)
    for i in range(n):
        kDiag[i] = np.exp(-Square_Euclidean_Distance(Z[i, :], Z[i, :]) / reg)

    # Main recursion, unrolled for efficiency
    for l in range(nLevels - 1, -1, -1):
        np.random(seed + l)
        # indices of current uniform sample
        rIndCurr = perm[: int(lSize[l])]
        # build sampled kernel
        SED = Square_Euclidean_Distance(Z[rIndCurr, :], Z[rInd, :])
        KS = np.exp(-SED / reg)
        SKS = KS[samp, :]
        SKSn = np.shape(SKS)[0]

        # optimal lambda for taking O(klogk) samples
        if k >= SKSn:
            # for the rare chance we take less than k samples in a round
            lam = 10e-6
            # don't set to exactly 0 to avoid stability issues
        else:
            ######
            Q = np.diag(weights.reshape(SKSn))
            Q = np.dot(Q, SKS)
            Oper = Q * weights.reshape(SKSn, 1)
            eigen = np.sort(np.linalg.eig(Oper)[1])[-k:]
            lam = (
                np.sum(np.diag(SKS) * (weights ** 2)) - np.sum(np.abs(np.real(eigen)))
            ) / k

        # compute and sample by lambda ridge leverage scores
        if l != 0:
            # on intermediate levels, we independently sample each column
            # by its leverage score. the sample size is sLevel in expectation
            R = np.linalg.inv(SKS + np.diag(np.dot(lam, weights ** (-2))))
            # max(0,.) helps avoid numerical issues, unnecessary in theory
            z = np.sum(np.dot(KS, R) * KS, 1)
            z = kDiag[rIndCurr] - z
            z = np.maximum(0, z)
            z = oversamp * (1 / lam) * z
            levs = np.minimum(1, z)

            M = np.random.rand(1, int(lSize[l])) - levs
            ind_matrix = M < 0
            ind_matrix = ind_matrix.reshape(int(lSize[l]))
            samp = np.where(ind_matrix == 1)[0]
            # with very low probability, we could accidentally sample no
            # columns. In this case, just take a fixed size uniform sample.
            samp_list = np.ndarray.tolist(samp)
            if len(samp_list) == 0:
                levs[:] = sLevel / lSize[l]
                samp = np.random.choice(int(lSize[l]), sLevel, replace=False)

            weights = np.sqrt(1.0 / (levs[samp]))

        else:
            # on the top level, we sample exactly s landmark points without replacement
            R = np.linalg.inv(SKS + np.diag(np.dot(lam, weights ** (-2))))
            z = np.sum(np.dot(KS, R) * KS, 1)
            z = kDiag[rIndCurr] - z
            z = np.maximum(0, z)
            levs = np.minimum(1, (1 / lam) * z)
            ########
            total_sum = np.sum(levs)
            levs_norm = levs / total_sum
            samp = np.random.choice(
                np.shape(levs)[0], rank, replace=False, p=levs_norm.reshape(-1)
            )

        rInd = perm[samp]

    # build final Nystrom approximation
    # pinv or inversion with slight regularization helps stability
    V = np.exp(-Square_Euclidean_Distance(Z, Z[rInd, :]) / reg)
    A = V[rInd, :]
    A = A + stable * np.eye(rank)
    # A_inv = np.linalg.inv(A)

    return A, V


# Adaptative Rank Nystrom: Square Euclidean Distance
def Adaptive_Nystrom_RBF(X, Y, reg, tau=1e-1, seed=49):
    err = 1e30
    r = 1
    while err > tau:
        r = 2 * r
        A, V = Nystrom_RBF(X, Y, reg, r)

        diag = np.zeros(np.shape(A)[0])
        for i in range(np.shape(A)[0]):
            M = np.dot(V, A)
            diag[i] = np.dot(M[i, :], V.T[:, i])

        err = 1 - np.min(diag)

    return A, V


# Square Euclidean Distance
def Square_Euclidean_Distance(X, Y):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = np.sum((X_col - Y_lin) ** 2, 2)
    return C


# Arccos Cost
def Arccos_Cost(X, Y, s=1, kappa=1e-6):
    if len(np.shape(X)) == 1:
        X = X.reshape(1, -1)

    if len(np.shape(Y)) == 1:
        Y = Y.reshape(1, -1)

    n, d = np.shape(X)
    m, d = np.shape(Y)
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            norm = np.linalg.norm(X[i, :]) * np.linalg.norm(Y[j, :])
            theta = np.arccos(Inner_Product(X[i, :], Y[j, :]) / norm)
            if s == 0:
                M[i, j] = (1 / np.pi) * (np.pi - theta)
            if s == 1:
                J = np.sin(theta) + (np.pi - theta) * np.cos(theta)
                M[i, j] = (1 / np.pi) * norm * J
            if s == 2:
                J = 3 * np.sin(theta) * np.cos(theta) + (np.pi - theta) * (
                    1 + 2 * np.cos(theta) ** 2
                )
                M[i, j] = (1 / np.pi) * (norm ** 2) * J

    M = M + kappa
    M = -np.log(M)
    return M


# Inner Product Cost
def Inner_Product(X, Y):
    if len(np.shape(X)) == 1:
        X = X.reshape(1, -1)

    if len(np.shape(Y)) == 1:
        Y = Y.reshape(1, -1)

    n, d = np.shape(X)
    m, d = np.shape(Y)
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            M[i, j] = np.sum(X[i, :] * Y[j, :])
    return M
