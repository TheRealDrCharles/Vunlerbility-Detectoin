"""
solving constraint optimization problem using Newton-Raphson method.
"""
import sys
import os
sys.path.append('..')
import numpy as np
import torch
from src.D_constraint import d_constraint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Newton(X, domain_set, C):
    n, d = X.shape
    #a = np.ones(d)
    a = torch.ones(d, dtype=torch.double).to(device)
    fudge = 0.000001
    threshold1 = 0.001
    reduction = 2

    # sum(d'Ad)=sum(trace(d'Ad))=sum(trace(dd'A))=trace(sum(dd'A))=trace(sum(dd')A)
    # sum(d_ij'a) = sum(d_ij')a where d_ij = [(di1-dj1)**2...(din-djn)**2]'
#     s_sum = np.zeros(d)
#     d_sum = np.zeros(d)
    s_sum = torch.zeros(d, dtype=torch.double).to(device)
    d_sum = torch.zeros(d, dtype=torch.double).to(device)
    for i in range(n):
        for j in range(i+1, n):
            d_ij = X[i] - X[j]
            #if S[i, j] == 1:
            if domain_set[i] == domain_set[j] == 1:
                s_sum += d_ij**2
            #elif D[i, j] == 1:
            elif domain_set[i] == domain_set[j] == 0:
                d_sum += d_ij**2
    
    tt = 1
    error = 1
    while error > threshold1:
        #fd0, fd_1st_d, fd_2nd_d = d_constraint(X, D, a)
        fd0, fd_1st_d, fd_2nd_d = d_constraint(X, domain_set, a)
#         obj_initial = s_sum.dot(a) - C*fd0
        obj_initial = s_sum @ a - C*fd0
        fs_1st_d = s_sum  # first derivative of the S constraint
        gradient = fs_1st_d - C*fd_1st_d  # the gradient of objective
#         Hessain = -C*fd_2nd_d + fudge*np.eye(d)
        Hessain = -C*fd_2nd_d + fudge*torch.eye(d, dtype=torch.double).to(device)
#         invHessian = np.linalg.inv(Hessain)
        invHessian = torch.linalg.inv(Hessain)
#         print('invhessain shape: ' + str(invHessian.shape))
#         print('gradient shape: ' + str(gradient.shape))

#         step = np.dot(invHessian, gradient)
        step = invHessian @ gradient

    # Newton-Raphson update
    # Search over optimal lambda
        lambda1 = 1
        t = 1
        a_previous = 0
        atemp = a - lambda1*step  # x[n+1] = x[n]-f(x[n])/df([xn])dx[n]
#         atemp = np.maximum(atemp, 0.000001)  # keep a to be positive
        atemp = torch.maximum(atemp, torch.tensor(0.000001, dtype=torch.double).to(device))  # keep a to be positive

        #fdd0 = d_constraint(X, D, atemp)
        fdd0 = d_constraint(X, domain_set, atemp)
#         obj = s_sum.dot(atemp) - C*fdd0[0]  # the a update to be atemp, compare this to obj_initial
        obj = s_sum @ atemp - C*fdd0[0]  # the a update to be atemp, compare this to obj_initial
        obj_previous = obj * torch.tensor(1.1, dtype=torch.double).to(device)  # just to get the while loop start

        while obj < obj_previous:
            obj_previous = obj
            a_previous = atemp
            lambda1 /= reduction
            atemp = a - lambda1*step
#             atemp = np.maximum(atemp, 0.000001)
            atemp = torch.maximum(atemp, torch.tensor(0.000001, dtype=torch.double).to(device))
            #fdd0 = d_constraint(X, D, atemp)
            fdd0 = d_constraint(X, domain_set, atemp)
#             obj = s_sum.dot(atemp) - C*fdd0[0]
            obj = s_sum @ atemp - C*fdd0[0]
            t += 1
            print(obj, obj_previous)
        a = a_previous
        error = abs((obj_previous - obj_initial)/obj_previous)
        tt += 1
        print('tt: ' + str(tt) + ' error: ' + str(error))
    return a

