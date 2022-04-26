# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author:被追杀的狼 time:2022/4/26


import os
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
import scipy.linalg as lin


class CP_dec:
    '''
    Different methods are used to realize the gradient descent and least square method of CP decomposition

    parameter:
    __________

    Tensor: The tensor which need be decompositon.
    epoch：The number of optimize,ALS and Gradient descent method reach best ruselt with many times optimize.
    rank: The rank of the tensors is set in advance
    err_tol: Acceptable error value
    alpha: Gradient descent rate
    lambda_1: Gradient down to the penalty rate

    returm:
    ____________

    factors: factors of tensor after is decompositon.
    dec_core_tensor: The core tensor of tensor after decomposed.

    '''
    def __init__(self, Tensor, epoch, rank, err_tol, alpha, lambda_1):
        self.tensor = Tensor
        self.epoch = epoch
        self.rank = rank
        self.err_tol = err_tol
        self.alpha = alpha
        self.lambda_1 = lambda_1

    def pthonly_CP_ALS(self):
        factors = parafac(self.tensor, self.rank)
        dec_core_tensor = tl.kruskal_to_tensor(factors)
        return factors,dec_core_tensor

    def hand_CP_ALS(self):

        np.random.seed(2022)

        def col_normalize(tensor):
            return tensor/np.sqrt(np.sum(tensor**2, axis=0))

        def tensor2matrix(tensor, mode):
            num_dim = len(tensor.shape)
            n = tensor.shape[num_dim - mode]
            tensor = np.moveaxis(tensor, num_dim - mode, -1)
            return np.reshape(tensor,(-1, n)).T

        def matrix2tensor(tensor1, out_shape):
            return np.reshape(tensor1.T, out_shape)

        n3, n2, n1 = self.tensor.shape
        C = np.random.normal(0, 1, (n2, self.rank))
        B = np.random.normal(0, 1, (n3, self.rank))
        tensor1 = tensor2matrix(self.tensor, 1)
        tensor2 = tensor2matrix(self.tensor, 2)
        tensor3 = tensor2matrix(self.tensor, 3)

        X_norm = lin.norm(tensor1, 'fro')
        err = np.inf
        B = col_normalize(B)
        i = 0
        while(err >= self.err_tol) and i < self.epoch:
            C = col_normalize(C)
            tem1 = lin.khatri_rao(C, B)
            A, res, rnk, s = lin.lstsq(tem1, tensor1.T)
            A = A.T

            A = col_normalize(A)
            tem2 = lin.khatri_rao(C, A)
            B, res, rnk, s = lin.lstsq(tem2, tensor3.T)
            B = B.T

            B = col_normalize(B)
            tem3 = lin.khatri_rao(B, A)
            C, res, rnk, s = lin.lstsq(tem3, tensor2.T)
            C = C.T

            X_hat1 = A.dot(lin.khatri_rao(C, B).T)
            err = lin.norm(X_hat1 - tensor1, 'fro')/ X_norm
            i +=1
            print('Relative error at iteration',i,':', err)
        X_hat = matrix2tensor(X_hat1, self.tensor.shape)
        print("Finished")
        return [A, B, C], X_hat



    def hand_CP_GD(self):

        np.random.seed(2022)

        def kr_prod(a, b):
            return np.einsum('ir, jr -> ijr', a, b).reshape(a.shape[0]*b.shape[0], -1)


        def ten2mat(tensor, mode):
            num_dim = len(tensor.shape)
            n = tensor.shape[mode]
            tensor = np.moveaxis(tensor, mode, 0)
            return np.reshape(tensor,(n, -1), order='F')


        def mat2ten(mat1, mat2, mat3):
            return np.einsum('ir, jr, tr -> ijt',mat1, mat2, mat3)


        n1, n2, n3 = self.tensor.shape
        A = 0.1*np.random.rand(n1, self.rank)
        B = 0.1*np.random.rand(n2, self.rank)
        C = 0.1*np.random.rand(n3, self.rank)

        pos = np.where(self.tensor != 0)
        bin_tensor = np.zeros((n1, n2, n3))
        bin_tensor[pos] = 1
        for iters in range(0, self.epoch):
            var1 = kr_prod(C, B)
            xi = self.tensor - mat2ten(A, B, C)
            grad = self.lambda_1*A - np.dot(ten2mat(bin_tensor*xi, 0),var1)
            A = A - self.alpha * grad/np.linalg.norm(grad)

            var1 = kr_prod(C, A)
            xi = self.tensor - mat2ten(A, B, C)
            grad = self.lambda_1*B - np.dot(ten2mat(bin_tensor*xi, 1),var1)
            B = B - self.alpha * grad/np.linalg.norm(grad)

            var1 = kr_prod(B, A)
            xi = self.tensor - mat2ten(A, B, C)
            grad = self.lambda_1*C - np.dot(ten2mat(bin_tensor*xi, 2),var1)
            C = C - self.alpha * grad/np.linalg.norm(grad)

            tensor_hat = mat2ten(A, B, C)
            loss = np.sum(np.square(self.tensor[pos] - tensor_hat[pos]))/ self.tensor[pos].shape[0]
            if (iters + 1) % 200 ==0:
                print('迭代次数', iters, '代价函数', loss)
        return [A, B, C], tensor_hat

#print(X)






if __name__== '__main__':
    Tensor = tl.tensor(np.arange(24.0).reshape(3, 4, 2))
    cp = CP_dec(Tensor,epoch=20000, rank=3, err_tol=1e-4, alpha=2e-3, lambda_1 = 1e-4)
    #factors,dec_core_tensor = cp.pthonly_CP_ALS()
    factors,dec_core_tensor = cp.hand_CP_ALS()
    #factors,dec_core_tensor = cp.hand_CP_GD()
    print("original_tensor: % r" % Tensor)
    print("full_tensor: % r" % dec_core_tensor)
    print(dec_core_tensor.shape)