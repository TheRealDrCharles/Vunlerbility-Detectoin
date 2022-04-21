import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def d_constraint(X, domain_set, a):
    n, d = X.shape
    sum_dist = 0
    sum_deri1 = torch.zeros(d, dtype=torch.double).to(device)
    sum_deri2 = torch.zeros((d, d), dtype=torch.double).to(device)

    for i in range(n):
        for j in range(i+1, n):
            #if D[i, j] == 1:
            if domain_set[i] == domain_set[j] == 0:
                d_ij = X[i] - X[j]
                dist_ij, deri1_d_ij, deri2_d_ij = distance1(a, d_ij)
                sum_dist += dist_ij
                sum_deri1 += deri1_d_ij
                sum_deri2 += deri2_d_ij
    fD, fD_1st_d, fD_2nd_d = gf(sum_dist, sum_deri1, sum_deri2)
    return [fD, fD_1st_d, fD_2nd_d]
    
    

def gf(sum_dist, sum_deri1, sum_deri2):
    #fD = np.log(sum_dist)
    fD = torch.log(sum_dist)
    fD_1st_d = sum_deri1/sum_dist
    #fD_2nd_d = sum_deri2/sum_dist - np.outer(sum_deri1, sum_deri1)/sum_dist**2
    fD_2nd_d = sum_deri2/sum_dist - torch.outer(sum_deri1, sum_deri1)/sum_dist**2
    return [fD, fD_1st_d, fD_2nd_d]


def distance1(a, d_ij):
    fudge = 0.000001
    #dist_ij = np.sqrt(np.dot(d_ij**2, a))  # distance between X[i] and X[j]
    dist_ij = torch.sqrt(torch.dot(d_ij**2, a))  # distance between X[i] and X[j]
    deri1_d_ij = 0.5*(d_ij**2)/(dist_ij + (dist_ij==0)*fudge)  # in case of dist_ij==0, shift dist_ij by 0.000001.
    #deri2_d_ij = -0.25*np.outer(d_ij**2, d_ij**2)/(dist_ij**3+(dist_ij==0)*fudge)  # the same as last one.
    deri2_d_ij = -0.25*torch.outer(d_ij**2, d_ij**2)/(dist_ij**3+(dist_ij==0)*fudge)  # the same as last one.
    return [dist_ij, deri1_d_ij, deri2_d_ij]
