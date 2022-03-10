import torch
import math
import numpy as np

def get_1d_dct_array(L):
    dct1d = torch.zeros(L, L)
    for i in range(L):
        for j in range(L):
            if i == 0:
                dct1d[i, j] = np.sqrt(1/L)
            else:
                dct1d[i, j] = np.sqrt(2/L) * np.cos((2 * j + 1) * i * np.pi/(2 * L))
    return dct1d

def transpose_matrix(A):
    '''
    Args:
        A: it's shape is (N, C, H, W)
    '''
    N, C, H, W = A.shape
    AT = np.zeros((N, C, H, W))
    for i in range(H):
        for j in range(W):
            AT[:, :, j, i] = A[:, :, i, j]
    return AT

def get_2d_dct(X):
    N, C, H, W = X.shape
    dct1d_array = get_1d_dct_array(H)
    dct1d_arrayT = dct1d_array.permute(1, 0)
    dct2d = torch.matmul(dct1d_array, X)
    dct2d = torch.matmul(dct2d, dct1d_arrayT)
    return dct2d

def keep_k_frequency(X, keepf=6):
    '''
    Args:
        X: the feature map, shape is (N, C, H, W)
        keepf: the number of keep frequency
    '''
    dct2d = get_2d_dct(X.cpu())
    N, C, H, W = dct2d.shape
    dct2d = dct2d.reshape(N, C, -1)
    sort_dct, _ = torch.sort(dct2d[:, :], descending=True)
    sort_dct = sort_dct[:, :, 0:keepf].sum(dim=2).reshape(N, C, 1, 1)
    return sort_dct

'''
b = torch.randint(10,(2,3,2,2))
a = b.reshape(2, 3, -1)
a1, idx1 = torch.sort(a[:, :], descending=True)
a2 = a1[:, :, 0:3]
a3 = a2.sum(dim=2)
a4 = a3.reshape(a2.shape[0], a2.shape[1], 1, 1)
print(a3)
print(a4)
'''






