#from mpi4py import MPI
#import multiprocessing
import pandas as pd
import math
import numpy as np
from sklearn.cluster import KMeans
#import subprocess
import time
import sys
import random
#import scipy
import copy
from sklearn.metrics.cluster import adjusted_rand_score


def read_job_cmd():
    gmin = int(sys.argv[1])
    gmax = int(sys.argv[2])
    qmin = int(sys.argv[3])
    qmax = int(sys.argv[4])
    tmin = int(sys.argv[5])
    tmax = int(sys.argv[6])
    n_in = int(sys.argv[7])
    p1_in = int(sys.argv[8])
    p2_in = int(sys.argv[9])
    trun = int(sys.argv[10])
    srun = int(sys.argv[11])
    ccolmn = int(sys.argv[12])
    seed_in = int(sys.argv[13])
    header_yn = int(sys.argv[14])
    data_in = str(sys.argv[15])
    return gmin, gmax, qmin, qmax, tmin, tmax, n_in, p1_in, p2_in, trun, srun, ccolmn, seed_in, header_yn, data_in


def convergtest_new(l, at, v_max, v, n, it, g, tol):
    l = np.append(l, 0.0)
    at = np.append(at, 0)
    flag = 0
    for i in range(0, n):
        summ = 0
        for j in range(0, g):
            summ += np.exp(v[i, j] - v_max[i])
        l[it] += math.log(summ) + v_max[i]
        if np.isnan(l[it]) or np.isinf(l[it]):
            return -1, l, at
    if it > 0:
        if l[it] < l[it-1]:
            return -1, l, at
    if it > 2:
        at[it - 1] = (l[it] - l[it - 1]) / (l[it - 1] - l[it - 2])
        if at[it - 1] < 1.0:
            l_inf = l[it - 1] + (l[it] - l[it - 1]) / (1 - at[it - 1])
            if abs(l_inf - l[it]) < tol:
                flag = 1
    return flag, l, at


def update_mu(n1, x, z, g, n, p):
    mu = np.zeros((g, p))
    ksum = 0
    for j in range(0, g):
        for k in range(0, p):
            mu[j, k] = sum(z[:, j] * x[:, k])
        mu[j, :] /= n1[j]
        ksum += sum(mu[j, :])
    return mu


def woodbury(x, lamda, psi, mu, p, q, lt):
    xm = np.transpose(x-mu)
    xm = xm[np.newaxis]
    lhs = np.sum(xm ** 2) / psi
    lvec = xm / psi
    tvec = np.dot(lvec, lamda)
    temp = lt / psi
    result = np.dot(temp, lamda) + np.identity(q, dtype=float)
    cp = np.linalg.inv(result)
    result = np.dot(cp, lt)
    cvec = np.dot(tvec, result)
    rhs = np.sum(cvec * xm) / psi
    output = lhs - rhs
    return output


def woodbury2(x, lamda, psi, mu, p, q, lt):
    psi = np.transpose(psi)
    xm = np.transpose(x-mu)
    xm = xm[np.newaxis]
    lhs = np.sum((xm ** 2) / psi)
    lvec = xm / psi
    tvec = np.dot(lvec, lamda)
    temp = lt / psi
    result = np.dot(temp, lamda) + np.identity(q, dtype=float)
    cp = np.linalg.inv(result)
    result = np.dot(cp, lt)
    cvec = np.dot(tvec, result)
    rhs = np.sum((cvec * xm) / psi)
    output = lhs - rhs
    return output


def update_z(x, z, lamda, psi, mu, pyi, log_c, n, g, p, q):
    x0 = np.zeros(p)
    mu0 = np.zeros(p)
    v0 = np.zeros(g)
    v = np.zeros((n, g))
    max_v = np.zeros(n)
    lt = np.transpose(lamda)
    for i in range(0, n):
        for j in range(0, g):
            x0 = x[i, :]
            mu0 = mu[j, :]
            a = woodbury(x0, lamda, psi, mu0, p, q, lt)
            e = a / 2.0 * (-1.0)
            v[i, j] = e + math.log(pyi[j]) - log_c
        v0 = v[i, :]
        max_v[i] = max(v0)
        d_alt = 0
        vmv = v[i, :] - max_v[i]
        d_alt += sum(np.exp(vmv))
        z[i, :] = np.exp(vmv) / d_alt
    return z, v, max_v


def update_z2(x, z, lamda, psi, mu, pyi, log_c, n, g, p, q):
    x0 = np.zeros(p)
    mu0 = np.zeros(p)
    v0 = np.zeros(g)
    v = np.zeros((n, g))
    max_v = np.zeros(n)
    lt = np.transpose(lamda)
    for i in range(0, n):
        for j in range(0, g):
            x0 = x[i, :]
            mu0 = mu[j, :]
            a = woodbury2(x0, lamda, psi, mu0, p, q, lt)
            e = a / 2.0 * (-1.0)
            v[i, j] = e + math.log(pyi[j]) - log_c
        v0 = v[i, :]
        max_v[i] = max(v0)
        d_alt = 0
        vmv = v[i, :] - max_v[i]
        d_alt += sum(np.exp(vmv))
        z[i, :] = np.exp(vmv) / d_alt
    return z, v, max_v


def update_z3(x, z, lamda, psi, mu, pyi, log_c, n, g, p, q):
    x0 = np.zeros(p)
    mu0 = np.zeros(p)
    v0 = np.zeros(g)
    v = np.zeros((n, g))
    max_v = np.zeros(n)
    lt = np.transpose(lamda)
    for i in range(0, n):
        for j in range(0, g):
            x0 = x[i, :]
            mu0 = mu[j, :]
            a = woodbury(x0, lamda, psi[j], mu0, p, q, lt)
            e = a / 2.0 * (-1.0)
            v[i, j] = e + math.log(pyi[j]) - log_c[j]
        v0 = v[i, :]
        max_v[i] = max(v0)
        d_alt = 0
        vmv = v[i, :] - max_v[i]
        d_alt += sum(np.exp(vmv))
        z[i, :] = np.exp(vmv) / d_alt
    return z, v, max_v


def update_z4(x, z, lamda, psi, mu, pyi, log_c, n, g, p, q):
    x0 = np.zeros(p)
    mu0 = np.zeros(p)
    v0 = np.zeros(g)
    v = np.zeros((n, g))
    max_v = np.zeros(n)
    lt = np.transpose(lamda)
    for i in range(0, n):
        for j in range(0, g):
            x0 = x[i, :]
            mu0 = mu[j, :]
            psi0 = psi[j*p:j*p+p]
            a = woodbury2(x0, lamda, psi0, mu0, p, q, lt)
            e = a / 2.0 * (-1.0)
            v[i, j] = e + math.log(pyi[j]) - log_c[j]
        v0 = v[i, :]
        max_v[i] = max(v0)
        d_alt = 0
        vmv = v[i, :] - max_v[i]
        d_alt += sum(np.exp(vmv))
        z[i, :] = np.exp(vmv) / d_alt
    return z, v, max_v


def update_z5(x, z, lamda, psi, mu, pyi, log_c, n, g, p, q):
    x0 = np.zeros(p)
    mu0 = np.zeros(p)
    v0 = np.zeros(g)
    v = np.zeros((n, g))
    max_v = np.zeros(n)
    for i in range(0, n):
        for j in range(0, g):
            x0 = x[i, :]
            mu0 = mu[j, :]
            lambda0 = np.transpose(np.reshape(lamda[:, j], (q, p), order='F'))
            lt = np.transpose(lambda0)
            a = woodbury(x0, lambda0, psi, mu0, p, q, lt)
            e = a / 2.0 * (-1.0)
            v[i, j] = e + math.log(pyi[j]) - log_c[j]
        v0 = v[i, :]
        max_v[i] = max(v0)
        d_alt = 0
        vmv = v[i, :] - max_v[i]
        d_alt += sum(np.exp(vmv))
        z[i, :] = np.exp(vmv) / d_alt
    return z, v, max_v


def update_z6(x, z, lamda, psi, mu, pyi, log_c, n, g, p, q):
    x0 = np.zeros(p)
    mu0 = np.zeros(p)
    v0 = np.zeros(g)
    v = np.zeros((n, g))
    max_v = np.zeros(n)
    for i in range(0, n):
        for j in range(0, g):
            x0 = x[i, :]
            mu0 = mu[j, :]
            lambda0 = np.transpose(np.reshape(lamda[:, j], (q, p), order='F'))
            lt = np.transpose(lambda0)
            a = woodbury2(x0, lambda0, psi, mu0, p, q, lt)
            e = a / 2.0 * (-1.0)
            v[i, j] = e + math.log(pyi[j]) - log_c[j]
        v0 = v[i, :]
        max_v[i] = max(v0)
        d_alt = 0
        vmv = v[i, :] - max_v[i]
        d_alt += sum(np.exp(vmv))
        z[i, :] = np.exp(vmv) / d_alt
    return z, v, max_v


def update_z7(x, z, lamda, psi, mu, pyi, log_c, n, g, p, q):
    x0 = np.zeros(p)
    mu0 = np.zeros(p)
    v0 = np.zeros(g)
    v = np.zeros((n, g))
    max_v = np.zeros(n)
    for i in range(0, n):
        for j in range(0, g):
            x0 = x[i, :]
            mu0 = mu[j, :]
            lambda0 = np.transpose(np.reshape(lamda[:, j], (q, p), order='F'))
            lt = np.transpose(lambda0)
            a = woodbury(x0, lambda0, psi[j], mu0, p, q, lt)
            e = a / 2.0 * (-1.0)
            v[i, j] = e + math.log(pyi[j]) - log_c[j]
        v0 = v[i, :]
        max_v[i] = max(v0)
        d_alt = 0
        vmv = v[i, :] - max_v[i]
        d_alt += sum(np.exp(vmv))
        z[i, :] = np.exp(vmv) / d_alt
    return z, v, max_v


def update_z8(x, z, lamda, psi, mu, pyi, log_c, n, g, p, q):
    x0 = np.zeros(p)
    mu0 = np.zeros(p)
    v0 = np.zeros(g)
    v = np.zeros((n, g))
    max_v = np.zeros(n)
    for i in range(0, n):
        for j in range(0, g):
            x0 = x[i, :]
            mu0 = mu[j, :]
            psi0 = psi[j, :]
            lambda0 = np.transpose(np.reshape(lamda[:, j], (q, p), order='F'))
            lt = np.transpose(lambda0)
            a = woodbury2(x0, lambda0, psi0, mu0, p, q, lt)
            e = a / 2.0 * (-1.0)
            v[i, j] = e + math.log(pyi[j]) - log_c[j]
        v0 = v[i, :]
        max_v[i] = max(v0)
        d_alt = 0
        vmv = v[i, :] - max_v[i]
        d_alt += sum(np.exp(vmv))
        z[i, :] = np.exp(vmv) / d_alt
    return z, v, max_v


def known_z(klass, z, n, g):
    z = np.zeros((n, g))
    for i in range(0, n):
        if klass[i] != 0:
            for j in range(0, g):
                z[i, j] = 0
                if j == klass[i]:
                    z[i, j] = 1
    return z


def update_stilde(x, z, mu, g, n, p, pyi):
    sampcovtilde = np.zeros((p, p))
    for t in range(0, g):
        #dsub = x - np.reshape(mu[t, :],(1,-1))
        #wv = weights(z[:,t])
        sampcovtilde += pyi[t]*np.cov(x, aweights=z[:, t], rowvar=False)*(sum(z[:, t]**2)-1)/sum(z[:, t])
    return sampcovtilde


def update_sg(x, z, mu, g, n, p, n1):
    sampcovtilde = {}
    for t in range(0, g):
        #dsub = x - np.reshape(mu[t, :],(1,-1))
        #wv = weights(z[:, t])
        sampcovtilde[t] = np.cov(x, aweights=z[:, t], rowvar=False)*(sum(z[:, t]**2)-1)/sum(z[:, t])
    return sampcovtilde


def update_beta1(psi, llambda, p, q):
    lhs = np.transpose(llambda) / psi
    cp = np.dot(lhs, llambda)
    result = cp + np.identity(q, dtype=float)
    rhs = np.linalg.inv(result)
    res = np.dot(cp, rhs)
    rhs = np.dot(res, lhs)
    beta = lhs - rhs
    return beta


def update_beta2(psi, llambda, p, q):
    lhs = np.transpose(llambda) / psi
    cp = np.dot(lhs, llambda)
    result = cp + np.identity(q, dtype=float)
    rhs = np.linalg.inv(result)
    res = np.dot(cp, rhs)
    rhs = np.dot(res, lhs)
    beta = lhs - rhs
    return beta


def update_theta(beta, llambda, sampcovtilde, p, q):
    r_1 = np.dot(beta, llambda)
    r_2 = np.dot(beta, sampcovtilde)
    r_3 = np.dot(r_2, np.transpose(beta))
    theta = np.identity(q, dtype=float) - r_1 + r_3
    return theta


def update_lambda(beta, s, theta, p, q):
    res1 = np.dot(s, np.transpose(beta))
    llambda = np.dot(res1, np.linalg.inv(theta))
    return llambda


def update_lambda2(beta, s, theta, n1, psi, p, q, g):
    res2 = np.zeros((p, q))
    for j in range(0, g):
        tran = np.transpose(beta[j])
        res1 = np.dot(s[j], tran)
        if j == 0:
            for i in range(0, p):
                for k in range(0, q):
                    res2[i, k] = res1[i, k] * n1[j] / psi[j]
        else:
            for i in range(0, p):
                for k in range(0, q):
                    res2[i, k] += res1[i, k] * n1[j] / psi[j]
    res3 = np.zeros((q, q))
    for j in range(0, g):
        if j == 0:
            for i in range(0, q):
                for k in range(0, q):
                    res3[i, k] = theta[j][i, k] * n1[j] / psi[j]
        else:
            for i in range(0, q):
                for k in range(0, q):
                    res3[i, k] += theta[j][i, k] * n1[j] / psi[j]
    llambda = np.dot(res2, np.linalg.inv(res3))
    return llambda


def update_lambda_cuu(beta, s, theta, n1, psi, p, q, g):
    llambda = np.zeros((p, q))
    res2 = np.zeros((p, q))
    for j in range(0, g):
        tran = np.transpose(beta[j])
        res1 = np.dot(s[j], tran)
        if j == 0:
            for i in range(0, p):
                for k in range(0, q):
                    res2[i, k] = res1[i, k] * n1[j] / psi[j*p+i]
        else:
            for i in range(0, p):
                for k in range(0, q):
                    res2[i, k] += res1[i, k] * n1[j] / psi[j*p+i]
    res3 = np.zeros((q, q))
    for ii in range(0, p):
        for j in range(0, g):
            if j == 0:
                for i in range(0, q):
                    for k in range(0, q):
                        res3[i, k] = theta[j][i, k] * n1[j] / psi[j*p+ii]
            else:
                for i in range(0, q):
                    for k in range(0, q):
                        res3[i, k] += theta[j][i, k] * n1[j] / psi[j*p+ii]
        llambda0 = np.transpose(res2[ii, :]) * np.linalg.inv(res3)
        llambda0 = llambda0.flatten('F')
        for j in range(0, q):
            llambda[ii, j] = llambda0[j]
    return llambda


def update_psi(llambda, beta, sampcovtilde, p, q):
    res1 = np.dot(llambda, beta)
    res2 = np.dot(res1, sampcovtilde)
    psi = sum(np.diag(sampcovtilde) - np.diag(res2)) / p
    return psi


def update_psi2(llambda, beta, sampcovtilde, p, q):
    res1 = np.dot(llambda, beta)
    res2 = np.dot(res1, sampcovtilde)
    psi = np.diag(sampcovtilde) - np.diag(res2)
    return psi


def update_psi3(llambda, beta, sampcovtilde, theta, p, q):
    res1 = np.dot(llambda, beta)
    res2 = np.diag(np.dot(res1, sampcovtilde))
    temp = np.transpose(llambda)
    res1 = np.dot(llambda, theta)
    res3 = np.diag(np.dot(res1, temp))
    psi = sum(np.diag(sampcovtilde) - 2*res2 + res3) / p
    return psi


def update_psi_cuu(llambda, beta, sampcovtilde, theta, p, q, g):
    res2 = np.zeros((g, p))
    for j in range(0, g):
        res1 = np.dot(llambda, beta[j])
        result = np.diag(np.dot(res1, sampcovtilde[j]))
        res2[j, :] = result
    res3 = np.zeros((g, p))
    temp = np.transpose(llambda)
    for j in range(0, g):
        res1 = np.dot(llambda, theta[j])
        result = np.diag(np.dot(res1, temp))
        res3[j, :] = result
    psi = np.zeros(g*p)
    for j in range(0, g):
        for i in range(0, p):
            psi[j*p+i] = sampcovtilde[j][i, i] - 2*res2[j, i] + res3[j, i]
    return psi


def update_psi_ucc(llambda, beta, sampcovtilde, p, q, pyi, g):
    res2 = np.zeros((g, p))
    for j in range(0, g):
        llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
        res1 = np.dot(llambda0, beta[j])
        result = np.diag(np.dot(res1, sampcovtilde[j]))
        res2[j, :] = result
    psi = 0
    for j in range(0, g):
        for i in range(0, p):
            psi += pyi[j] * (sampcovtilde[j][i, i] - res2[j, i])
    psi = psi / p
    return psi


def update_psi_ucu(llambda, beta, sampcovtilde, p, q, pyi, g):
    res2 = np.zeros((g, p))
    for j in range(0, g):
        llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
        res1 = np.dot(llambda0, beta[j])
        result = np.diag(np.dot(res1, sampcovtilde[j]))
        res2[j, :] = result
    psi = np.zeros(p)
    for i in range(0, p):
        for j in range(0, g):
            psi[i] += pyi[j] * (sampcovtilde[j][i, i] - res2[j, i])
    return psi


def update_det_sigma_new(llambda, psi, log_detpsi, p, q):
    tmp2 = update_beta1(psi, llambda, p, q)
    tmp = np.dot(tmp2, llambda)
    tmp2 = -1 * tmp + np.identity(q, dtype=float)
    det_sigma_new = log_detpsi - np.log(np.linalg.det(tmp2))
    return det_sigma_new


def update_det_sigma_new2(llambda, psi, log_detpsi, p, q):
    tmp2 = update_beta2(psi, llambda, p, q)
    tmp = np.dot(tmp2, llambda)
    tmp2 = -1 * tmp + np.identity(q, dtype=float)
    det_sigma_new = log_detpsi - np.log(np.linalg.det(tmp2))
    return det_sigma_new


def update_omega(llambda, delta, beta, sampcovtilde, theta, p, q):
    result_1 = np.dot(llambda, beta)
    result_2 = np.diag(np.dot(result_1, sampcovtilde))
    temp = np.transpose(llambda)
    result_1 = np.dot(llambda, theta)
    result_3 = np.diag(np.dot(result_1, temp))
    omega = sum((np.diag(sampcovtilde) - 2*result_2 + result_3) / delta) / p
    return omega


def update_omega2(llambda, delta, beta, sampcovtilde, p, q):
    result_1 = np.dot(llambda, beta)
    result_2 = np.diag(np.dot(result_1, sampcovtilde))
    omega = sum((np.diag(sampcovtilde) - result_2) / delta) / p
    return omega


def update_delta(llambda, omega, beta, sampcovtilde, theta, n1, p, q, n, g):
    result_2 = np.zeros((g, p))
    for j in range(0, g):
        result_1 = np.dot(llambda, beta[j])
        result = np.diag(np.dot(result_1, sampcovtilde[j]))
        result_2[j, :] = result
    result_3 = np.zeros((g, p))
    temp = np.transpose(llambda)
    for j in range(0, g):
        result_1 = np.dot(llambda, theta[j])
        result = np.diag(np.dot(result_1, temp))
        result_3[j, :] = result
    temp1 = np.zeros(p)
    for i in range(0, p):
        for j in range(0, g):
            temp1[i] += (sampcovtilde[j][i, i] - 2*result_2[j, i] +
            result_3[j, i]) * n1[j] / omega[j]
    lagrange = sum(np.log(temp1)) / p
    lagrange = (np.exp(lagrange) - n)/2
    delta = temp1 / (n + 2*lagrange)
    return delta


def update_delta2(llambda, omega, beta, sampcovtilde, theta, n1, p, q, n, g):
    result_2 = np.zeros((g, p))
    for j in range(0, g):
        llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
        result_1 = np.dot(llambda0, beta[j])
        result = np.diag(np.dot(result_1, sampcovtilde[j]))
        result_2[j, :] = result
    result_3 = np.zeros((g, p))
    for j in range(0, g):
        llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
        temp = np.transpose(llambda0)
        result_1 = np.dot(llambda0, theta[j])
        result = np.diag(np.dot(result_1, temp))
        result_3[j, :] = result
    temp1 = np.zeros(p)
    lagrange = 0
    for i in range(0, p):
        for j in range(0, g):
            temp1[i] += (sampcovtilde[j][i, i] - 2*result_2[j, i] + result_3[j, i]) * n1[j] / omega[j]
            lagrange += np.log(temp1[i])
    lagrange = lagrange / p
    lagrange = (np.exp(lagrange) - n)/2
    delta = temp1 / (n + 2*lagrange)
    return delta


def update_delta3(llambda, omega, beta, sampcovtilde, theta, n1, p, q):
    result_1 = np.dot(llambda, beta)
    result_2 = np.diag(np.dot(result_1, sampcovtilde))
    temp = np.transpose(llambda)
    result_1 = np.dot(llambda, theta)
    result_3 = np.diag(np.dot(result_1, temp))
    temp1 = np.diag(sampcovtilde) - 2*result_2 + result_3
    lagrange = sum(np.log(temp1)) / p
    lagrange = np.exp(lagrange) / omega
    lagrange = (lagrange-1) * (n1/2)
    delta = temp1/((1+(2*lagrange/n1))*omega)
    return delta


def update_z9(x, z, lamda, omega, delta, mu, pyi, log_c, n, g, p, q):
    x0 = np.zeros(p)
    mu0 = np.zeros(p)
    v0 = np.zeros(g)
    v = np.zeros((n, g))
    max_v = np.zeros(n)
    lt = np.transpose(lamda)
    for i in range(0, n):
        for j in range(0, g):
            psi = np.dot(omega[j], delta)
            x0 = x[i, :]
            mu0 = mu[j, :]
            a = woodbury2(x0, lamda, psi, mu0, p, q, lt)
            e = a / 2.0 * (-1.0)
            v[i, j] = e + np.log(pyi[j]) - log_c[j]
        v0 = v[i, :]
        max_v[i] = max(v0)
        d_alt = 0
        vmv = v[i, :] - max_v[i]
        d_alt += sum(np.exp(vmv))
        z[i, :] = np.exp(vmv) / d_alt
    return z, v, max_v


def update_z10(x, z, lamda, omega, delta, mu, pyi, log_c, n, g, p, q):
    x0 = np.zeros(p)
    mu0 = np.zeros(p)
    v0 = np.zeros(g)
    v = np.zeros((n, g))
    max_v = np.zeros(n)
    for i in range(0, n):
        for j in range(0, g):
            psi = np.dot(omega[j],  delta)
            x0 = x[i, :]
            mu0 = mu[j, :]
            lambda0 = np.transpose(np.reshape(lamda[:, j], (q, p), order='F'))
            lt = np.transpose(lambda0)
            a = woodbury2(x0, lambda0, psi, mu0, p, q, lt)
            e = a / 2.0 * (-1.0)
            v[i, j] = e + np.log(pyi[j]) - log_c[j]
        v0 = v[i, :]
        max_v[i] = max(v0)
        d_alt = 0
        vmv = v[i, :] - max_v[i]
        d_alt += sum(np.exp(vmv))
        z[i, :] = np.exp(vmv) / d_alt
    return z, v, max_v


def update_z11(x, z, lamda, omega, delta, mu, pyi, log_c, n, g, p, q):
    x0 = np.zeros(p)
    mu0 = np.zeros(p)
    v0 = np.zeros(g)
    v = np.zeros((n, g))
    max_v = np.zeros(n)
    lt = np.transpose(lamda)
    for i in range(0, n):
        for j in range(0, g):
            psi = np.dot(omega, delta[j, :])
            x0 = x[i, :]
            mu0 = mu[j, :]
            a = woodbury2(x0, lamda, psi, mu0, p, q, lt)
            e = a / 2.0 * (-1.0)
            v[i, j] = e + np.log(pyi[j]) - log_c[j]
        v0 = v[i, :]
        max_v[i] = max(v0)
        d_alt = 0
        vmv = v[i, :] - max_v[i]
        d_alt += sum(np.exp(vmv))
        z[i, :] = np.exp(vmv) / d_alt
    return z, v, max_v


def update_z12(x, z, lamda, omega, delta, mu, pyi, log_c, n, g, p, q):
    x0 = np.zeros(p)
    mu0 = np.zeros(p)
    v0 = np.zeros(g)
    v = np.zeros((n, g))
    max_v = np.zeros(n)
    for i in range(0, n):
        for j in range(0, g):
            psi = np.dot(omega, delta[j, :])
            x0 = x[i, :]
            mu0 = mu[j, :]
            lambda0 = np.transpose(np.reshape(lamda[:, j], (q, p), order='F'))
            lt = np.transpose(lambda0)
            a = woodbury2(x0, lambda0, psi, mu0, p, q, lt)
            e = a / 2.0 * (-1.0)
            v[i, j] = e + np.log(pyi[j]) - log_c[j]
        v0 = v[i, :]
        max_v[i] = max(v0)
        d_alt = 0
        vmv = v[i, :] - max_v[i]
        d_alt += sum(np.exp(vmv))
        z[i, :] = np.exp(vmv) / d_alt
    return z, v, max_v


def aecm(z, x, cls, q, p, g, n, llambda, psi, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.transpose(np.reshape(llambda[0:q*p], (q, p), order='F'))
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
            z = known_z(cls, z, n, g)
        sampcovtilde = update_stilde(x, z, mu, g, n, p, pyi)
        beta = update_beta1(psi, llambda, p, q)
        theta = update_theta(beta, llambda, sampcovtilde, p, q)
        llambda = update_lambda(beta, sampcovtilde, theta, p, q)
        psi = update_psi(llambda, beta, sampcovtilde, p, q)
        log_detpsi = p * np.log(psi)
        log_detsig = update_det_sigma_new(llambda, psi, log_detpsi, p, q)
        log_c = (p / 2) * np.log(2 * np.pi) + 0.5 * log_detsig
        tmpzv = update_z(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        z = known_z(cls, z, n, g)
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + p*q - q*(q-1)/2 + 1
    bic = 2*l[it-1] - paras * np.log(n)
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def aecm2(z, x, cls, q, p, g, n, llambda, psi, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.transpose(np.reshape(llambda[0:q*p], (q, p), order='F'))
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z2(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
            z = known_z(cls, z, n, g)
        sampcovtilde = update_stilde(x, z, mu, g, n, p, pyi)
        beta = update_beta2(psi, llambda, p, q)
        theta = update_theta(beta, llambda, sampcovtilde, p, q)
        llambda = update_lambda(beta, sampcovtilde, theta, p, q)
        psi = update_psi2(llambda, beta, sampcovtilde, p, q)
        log_detpsi = 0
        for i in range(0, p):
            log_detpsi += np.log(psi[i])
        log_detsig = update_det_sigma_new2(llambda, psi, log_detpsi, p, q)
        log_c = (p / 2) * np.log(2 * np.pi) + 0.5 * log_detsig
        tmpzv = update_z2(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        z = known_z(cls, z, n, g)
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + p*q - q*(q-1)/2 + p
    bic = 2*l[it-1] - paras * np.log(n)
    #println(bic)
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def aecm3(z, x, cls, q, p, g, n, llambda, psi, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.transpose(np.reshape(llambda[0:q*p], (q, p), order='F'))
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z3(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
            z = known_z(cls, z, n, g)
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            beta[j] = update_beta1(psi[j], llambda, p, q)
        theta = {}
        for j in range(0, g):
            theta[j] = update_theta(beta[j], llambda, sampcovtilde[j], p, q)
        llambda = update_lambda2(beta, sampcovtilde, theta, n1, psi, p, q, g)
        psi = np.zeros(g)
        for j in range(0, g):
            psi[j] = update_psi3(llambda, beta[j], sampcovtilde[j], theta[j], p, q)
        log_detpsi = np.zeros(g)
        log_detsig = np.zeros(g)
        log_c = np.zeros(g)
        for j in range(0, g):
            log_detpsi[j] = p * np.log(psi[j])
            log_detsig[j] = update_det_sigma_new(llambda, psi[j], log_detpsi[j], p, q)
            log_c[j] = (p/2) * np.log(2 * np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z3(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        z = known_z(cls, z, n, g)
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + p*q - q*(q-1)/2 + g
    bic = 2*l[it-1] - paras * np.log(n)
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def aecm4(z, x, cls, q, p, g, n, llambda, psi, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.transpose(np.reshape(llambda[0:q*p], (q, p), order='F'))
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z4(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
            z = known_z(cls, z, n, g)
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            psi0 = psi[j*p:j*p+p]
            beta[j] = update_beta2(psi0, llambda, p, q)
        theta = {}
        for j in range(0, g):
            theta[j] = update_theta(beta[j], llambda, sampcovtilde[j], p, q)
        llambda = update_lambda_cuu(beta, sampcovtilde, theta, n1, psi, p, q, g)
        psi = update_psi_cuu(llambda, beta, sampcovtilde, theta, p, q, g)
        log_detpsi = np.zeros(g)
        for j in range(0, g):
            log_detpsi[j] += sum(np.log(psi[j*p:j*p+p]))
        log_detsig = np.zeros(g)
        for j in range(0, g):
            psi0 = psi[j*p:j*p+p]
            log_detsig[j] = update_det_sigma_new2(llambda, psi0, log_detpsi[j], p, q)
        log_c = np.zeros(g)
        for j in range(0, g):
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z4(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        z = known_z(cls, z, n, g)
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + p*q - q*(q-1)/2 + g*p
    bic = 2*l[it-1] - paras * np.log(n)
    #println(bic)
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def aecm5(z, x, cls, q, p, g, n, llambda, psi, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.reshape(llambda, (q*p, g), order='F')
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z5(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
            z = known_z(cls, z, n, g)
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            beta[j] = update_beta1(psi, llambda0, p, q)
        theta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
        for j in range(0, g):
            t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
            llambda[:, j] = (np.transpose(t_llambda)).flatten('F')
        psi = update_psi_ucc(llambda, beta, sampcovtilde, p, q, pyi, g)
        log_detpsi = 0
        for j in range(0, p):
            log_detpsi += np.sum(np.log(psi))
        log_detsig = np.zeros(g)
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            log_detsig[j] = update_det_sigma_new(llambda0, psi, log_detpsi, p, q)
        log_c = np.zeros(g)
        for j in range(0, g):
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z5(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        z = known_z(cls, z, n, g)
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + 1
    bic = 2*l[it-1] - paras * np.log(n)
    #println(bic)
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def aecm6(z, x, cls, q, p, g, n, llambda, psi, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.reshape(llambda, (q*p, g), order='F')
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z6(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
            z = known_z(cls, z, n, g)
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            beta[j] = update_beta2(psi, llambda0, p, q)
        theta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
        for j in range(0, g):
            t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
            llambda[:, j] = (np.transpose(t_llambda)).flatten('F')
        psi = update_psi_ucu(llambda, beta, sampcovtilde, p, q, pyi, g)
        log_detpsi = 0
        for j in range(0, p):
            log_detpsi += np.sum(np.log(psi[j]))
        log_detsig = np.zeros(g)
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            log_detsig[j] = update_det_sigma_new2(llambda0, psi, log_detpsi, p, q)
        log_c = np.zeros(g)
        for j in range(0, g):
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z6(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        z = known_z(cls, z, n, g)
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + p
    bic = 2*l[it-1] - paras * np.log(n)
    #println(bic)
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def aecm7(z, x, cls, q, p, g, n, llambda, psi, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.reshape(llambda, (q*p, g), order='F')
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z7(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
            z = known_z(cls, z, n, g)
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            beta[j] = update_beta1(psi[j], llambda0, p, q)
        theta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
        for j in range(0, g):
            t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
            llambda[:, j] = (np.transpose(t_llambda)).flatten('F')
        psi = np.zeros(g)
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            psi[j] = update_psi(llambda0, beta[j], sampcovtilde[j], p, q)
        log_detpsi = p * np.log(psi)
        log_detsig = np.zeros(g)
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            log_detsig[j] = update_det_sigma_new(llambda0, psi[j], log_detpsi[j], p, q)
        log_c = np.zeros(g)
        for j in range(0, g):
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z7(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        z = known_z(cls, z, n, g)
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + g
    bic = 2*l[it-1] - paras * np.log(n)
    #println(bic)
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def aecm8(z, x, cls, q, p, g, n, llambda, psi, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.reshape(llambda, (q*p, g), order='F')
    psi = np.transpose(np.reshape(psi, (p, g), order='F'))
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z8(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
            z = known_z(cls, z, n, g)
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            psi0 = psi[j, :]
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            beta[j] = update_beta2(psi0, llambda0, p, q)
        theta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
        for j in range(0, g):
            t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
            llambda[:, j] = (np.transpose(t_llambda)).flatten('F')
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            psi_temp = update_psi2(llambda0, beta[j], sampcovtilde[j], p, q)
            psi[j, :] = psi_temp
        log_detpsi = np.zeros(g)
        for j in range(0, g):
            psi0 = psi[j, :]
            log_detpsi[j] = sum(np.log(psi0))
        log_detsig = np.zeros(g)
        for j in range(0, g):
            psi0 = psi[j, :]
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            log_detsig[j] = update_det_sigma_new2(llambda0, psi0, log_detpsi[j], p, q)
        log_c = np.zeros(g)
        for j in range(0, g):
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z8(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        z = known_z(cls, z, n, g)
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + g*p
    bic = 2*l[it-1] - paras * np.log(n)
    #println(bic)
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def aecm9(z, x, cls, q, p, g, n, llambda, omega, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.transpose(np.reshape(llambda[0:q*p], (q, p), order='F'))
    delta = np.ones(p)
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z9(x, z, llambda, omega, delta, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
            z = known_z(cls, z, n, g)
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            psi = np.dot(omega[j],  delta)
            beta[j] = update_beta2(psi, llambda, p, q)
        theta = {}
        for j in range(0, g):
            theta[j] = update_theta(beta[j], llambda, sampcovtilde[j], p, q)
        llambda = update_lambda2(beta, sampcovtilde, theta, n1, omega, p, q, g)
        for j in range(0, g):
            omega[j] = update_omega(llambda, delta, beta[j], sampcovtilde[j], theta[j], p, q)
        delta = update_delta(llambda, omega, beta, sampcovtilde, theta, n1, p, q, n, g)
        log_detpsi = np.zeros(g)
        log_detsig = np.zeros(g)
        log_c = np.zeros(g)
        for j in range(0, g):
            psi = np.dot(omega[j], delta)
            log_detpsi[j] = p * np.log(omega[j])
            log_detsig[j] = update_det_sigma_new2(llambda, psi, log_detpsi[j], p, q)
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z9(x, z, llambda, omega, delta, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        z = known_z(cls, z, n, g)
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + p*q - q*(q-1)/2 + g + (p-1)
    bic = 2*l[it-1] - paras * np.log(n)
    #println(l)
    for i in range(0, p):
        omega[g+i] = delta[i]
    if flag == -1:
        return z, -np.inf, llambda, omega
    return z, bic, llambda, omega


def aecm10(z, x, cls, q, p, g, n, llambda, omega, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.reshape(llambda, (q*p, g), order='F')
    delta = np.ones(p)
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z10(x, z, llambda, omega, delta, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
            z = known_z(cls, z, n, g)
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            psi = np.dot(omega[j], delta)
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            beta[j] = update_beta2(psi, llambda0, p, q)
        theta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
        for j in range(0, g):
            t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
            llambda[:, j] = (np.transpose(t_llambda)).flatten('F')
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            omega[j] = update_omega2(llambda0, delta, beta[j], sampcovtilde[j], p, q)
        delta = update_delta2(llambda, omega, beta, sampcovtilde, theta, n1, p, q, n, g)
        log_detpsi = np.zeros(g)
        log_detsig = np.zeros(g)
        log_c = np.zeros(g)
        for j in range(0, g):
            psi = np.dot(omega[j], delta)
            log_detpsi[j] = p * np.log(omega[j])
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            log_detsig[j] = update_det_sigma_new2(llambda0, psi, log_detpsi[j], p, q)
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z10(x, z, llambda, omega, delta, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        z = known_z(cls, z, n, g)
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + g + (p-1)
    bic = 2*l[it-1] - paras * np.log(n)
    #println(bic)
    for i in range(0, p):
        omega[g+i] = delta[i]
    if flag == -1:
        return z, -np.inf, llambda, omega
    return z, bic, llambda, omega


def aecm11(z, x, cls, q, p, g, n, llambda, omega, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.transpose(np.reshape(llambda[0:q*p], (q, p), order='F'))
    delta = np.ones((g, p))
    omega = omega[0]
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z11(x, z, llambda, omega, delta, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
            z = known_z(cls, z, n, g)
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            psi = np.dot(omega, delta[j, :])
            beta[j] = update_beta2(psi, llambda, p, q)
        theta = {}
        for j in range(0, g):
            theta[j] = update_theta(beta[j], llambda, sampcovtilde[j], p, q)
        del0 = (np.transpose(delta)).flatten('F')
        llambda = update_lambda_cuu(beta, sampcovtilde, theta, n1, del0, p, q, g)
        omega = 0
        for j in range(0, g):
            delta0 = delta[j, :]
            omega += pyi[j] * update_omega(llambda, delta0, beta[j], sampcovtilde[j], theta[j], p, q)
        for j in range(0, g):
            delta0 = delta[j, :]
            delta_temp = update_delta3(llambda, omega, beta[j], sampcovtilde[j], theta[j], n1[j], p, q)
            delta[j, :] = delta_temp
        log_detpsi = p * np.log(omega)
        log_detsig = np.zeros(g)
        log_c = np.zeros(g)
        for j in range(0, g):
            psi = np.dot(omega, delta[j, :])
            log_detsig[j] = update_det_sigma_new2(llambda, psi, log_detpsi, p, q)
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z11(x, z, llambda, omega, delta, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        z = known_z(cls, z, n, g)
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + p*q - q*(q-1)/2 + 1 + g*(p-1)
    #print(l)
    bic = 2*l[it-1] - paras * np.log(n)
    psi = np.zeros(g * p + 1)
    psi[0] = omega
    k = 0
    for j in range(0, g):
        for i in range(0, p):
            psi[k] = delta[j, i]
            k += 1
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def aecm12(z, x, cls, q, p, g, n, llambda, omega, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.reshape(llambda, (q*p, g), order='F')
    delta = np.ones((g, p))
    omega = omega[0]
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z12(x, z, llambda, omega, delta, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
            z = known_z(cls, z, n, g)
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            psi = np.dot(omega, delta[j, :])
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            beta[j] = update_beta2(psi, llambda0, p, q)
        theta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
        for j in range(0, g):
            t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
            llambda[:, j] = (np.transpose(t_llambda)).flatten('F')
        omega = 0
        for j in range(0, g):
            delta0 = delta[j, :]
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            omega += pyi[j] * update_omega2(llambda0, delta0, beta[j], sampcovtilde[j], p, q)
        for j in range(0, g):
            delta0 = delta[j, :]
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            delta_temp = update_delta3(llambda0, omega, beta[j], sampcovtilde[j], theta[j], n1[j], p, q)
            delta[j, :] = delta_temp
        log_detpsi = p * np.log(omega)
        log_detsig = np.zeros(g)
        log_c = np.zeros(g)
        for j in range(0, g):
            psi = np.dot(omega, delta[j, :])
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            log_detsig[j] = update_det_sigma_new2(llambda0, psi, log_detpsi, p, q)
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z12(x, z, llambda, omega, delta, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        z = known_z(cls, z, n, g)
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + 1 + g*(p-1)
    #print(l)
    bic = 2*l[it-1] - paras * np.log(n)
    psi = np.zeros(g * p + 1)
    psi[0] = omega
    k = 1 #global
    for j in range(0, g):
        for i in range(0, p):
            psi[k] = delta[j, i]
            k += 1
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def claecm(z, x, q, p, g, n, llambda, psi, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.transpose(np.reshape(llambda[0:q*p], (q, p), order='F'))
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
        sampcovtilde = update_stilde(x, z, mu, g, n, p, pyi)
        beta = update_beta1(psi, llambda, p, q)
        theta = update_theta(beta, llambda, sampcovtilde, p, q)
        llambda = update_lambda(beta, sampcovtilde, theta, p, q)
        psi = update_psi(llambda, beta, sampcovtilde, p, q)
        log_detpsi = p * np.log(psi)
        log_detsig = update_det_sigma_new(llambda, psi, log_detpsi, p, q)
        log_c = (p / 2) * np.log(2 * np.pi) + 0.5 * log_detsig
        tmpzv = update_z(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + p*q - q*(q-1)/2 + 1
    bic = 2*l[it-1] - paras * np.log(n)
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def claecm2(z, x, q, p, g, n, llambda, psi, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.transpose(np.reshape(llambda[0:q*p], (q, p), order='F'))
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z2(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
        sampcovtilde = update_stilde(x, z, mu, g, n, p, pyi)
        beta = update_beta2(psi, llambda, p, q)
        theta = update_theta(beta, llambda, sampcovtilde, p, q)
        llambda = update_lambda(beta, sampcovtilde, theta, p, q)
        psi = update_psi2(llambda, beta, sampcovtilde, p, q)
        log_detpsi = 0
        for i in range(0, p):
            log_detpsi += np.log(psi[i])
        log_detsig = update_det_sigma_new2(llambda, psi, log_detpsi, p, q)
        log_c = (p / 2) * np.log(2 * np.pi) + 0.5 * log_detsig
        tmpzv = update_z2(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + p*q - q*(q-1)/2 + p
    bic = 2*l[it-1] - paras * np.log(n)
    #println(bic)
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def claecm3(z, x, q, p, g, n, llambda, psi, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.transpose(np.reshape(llambda[0:q*p], (q, p), order='F'))
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z3(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            beta[j] = update_beta1(psi[j], llambda, p, q)
        theta = {}
        for j in range(0, g):
            theta[j] = update_theta(beta[j], llambda, sampcovtilde[j], p, q)
        llambda = update_lambda2(beta, sampcovtilde, theta, n1, psi, p, q, g)
        psi = np.zeros(g)
        for j in range(0, g):
            psi[j] = update_psi3(llambda, beta[j], sampcovtilde[j], theta[j], p, q)
        log_detpsi = np.zeros(g)
        log_detsig = np.zeros(g)
        log_c = np.zeros(g)
        for j in range(0, g):
            log_detpsi[j] = p * np.log(psi[j])
            log_detsig[j] = update_det_sigma_new(llambda, psi[j], log_detpsi[j], p, q)
            log_c[j] = (p/2) * np.log(2 * np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z3(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + p*q - q*(q-1)/2 + g
    bic = 2*l[it-1] - paras * np.log(n)
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def claecm4(z, x, q, p, g, n, llambda, psi, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.transpose(np.reshape(llambda[0:q*p], (q, p), order='F'))
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z4(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            psi0 = psi[j*p:j*p+p]
            beta[j] = update_beta2(psi0, llambda, p, q)
        theta = {}
        for j in range(0, g):
            theta[j] = update_theta(beta[j], llambda, sampcovtilde[j], p, q)
        llambda = update_lambda_cuu(beta, sampcovtilde, theta, n1, psi, p, q, g)
        psi = update_psi_cuu(llambda, beta, sampcovtilde, theta, p, q, g)
        log_detpsi = np.zeros(g)
        for j in range(0, g):
            log_detpsi[j] += sum(np.log(psi[j*p:j*p+p]))
        log_detsig = np.zeros(g)
        for j in range(0, g):
            psi0 = psi[j*p:j*p+p]
            log_detsig[j] = update_det_sigma_new2(llambda, psi0, log_detpsi[j], p, q)
        log_c = np.zeros(g)
        for j in range(0, g):
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z4(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + p*q - q*(q-1)/2 + g*p
    bic = 2*l[it-1] - paras * np.log(n)
    #println(bic)
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def claecm5(z, x, q, p, g, n, llambda, psi, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.reshape(llambda, (q*p, g), order='F')
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z5(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            beta[j] = update_beta1(psi, llambda0, p, q)
        theta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
        for j in range(0, g):
            t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
            llambda[:, j] = (np.transpose(t_llambda)).flatten('F')
        psi = update_psi_ucc(llambda, beta, sampcovtilde, p, q, pyi, g)
        log_detpsi = 0
        for j in range(0, p):
            log_detpsi += np.sum(np.log(psi))
        log_detsig = np.zeros(g)
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            log_detsig[j] = update_det_sigma_new(llambda0, psi, log_detpsi, p, q)
        log_c = np.zeros(g)
        for j in range(0, g):
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z5(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + 1
    bic = 2*l[it-1] - paras * np.log(n)
    #println(bic)
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def claecm6(z, x, q, p, g, n, llambda, psi, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.reshape(llambda, (q*p, g), order='F')
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z6(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            beta[j] = update_beta2(psi, llambda0, p, q)
        theta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
        for j in range(0, g):
            t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
            llambda[:, j] = (np.transpose(t_llambda)).flatten('F')
        psi = update_psi_ucu(llambda, beta, sampcovtilde, p, q, pyi, g)
        log_detpsi = 0
        for j in range(0, p):
            log_detpsi += np.sum(np.log(psi[j]))
        log_detsig = np.zeros(g)
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            log_detsig[j] = update_det_sigma_new2(llambda0, psi, log_detpsi, p, q)
        log_c = np.zeros(g)
        for j in range(0, g):
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z6(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + p
    bic = 2*l[it-1] - paras * np.log(n)
    #println(bic)
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def claecm7(z, x, q, p, g, n, llambda, psi, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.reshape(llambda, (q*p, g), order='F')
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z7(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            beta[j] = update_beta1(psi[j], llambda0, p, q)
        theta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
        for j in range(0, g):
            t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
            llambda[:, j] = (np.transpose(t_llambda)).flatten('F')
        psi = np.zeros(g)
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            psi[j] = update_psi(llambda0, beta[j], sampcovtilde[j], p, q)
        log_detpsi = p * np.log(psi)
        log_detsig = np.zeros(g)
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            log_detsig[j] = update_det_sigma_new(llambda0, psi[j], log_detpsi[j], p, q)
        log_c = np.zeros(g)
        for j in range(0, g):
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z7(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + g
    bic = 2*l[it-1] - paras * np.log(n)
    #println(bic)
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def claecm8(z, x, q, p, g, n, llambda, psi, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.reshape(llambda, (q*p, g), order='F')
    psi = np.transpose(np.reshape(psi, (p, g), order='F'))
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z8(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            psi0 = psi[j, :]
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            beta[j] = update_beta2(psi0, llambda0, p, q)
        theta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
        for j in range(0, g):
            t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
            llambda[:, j] = (np.transpose(t_llambda)).flatten('F')
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            psi_temp = update_psi2(llambda0, beta[j], sampcovtilde[j], p, q)
            psi[j, :] = psi_temp
        log_detpsi = np.zeros(g)
        for j in range(0, g):
            psi0 = psi[j, :]
            log_detpsi[j] = sum(np.log(psi0))
        log_detsig = np.zeros(g)
        for j in range(0, g):
            psi0 = psi[j, :]
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            log_detsig[j] = update_det_sigma_new2(llambda0, psi0, log_detpsi[j], p, q)
        log_c = np.zeros(g)
        for j in range(0, g):
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z8(x, z, llambda, psi, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + g*p
    bic = 2*l[it-1] - paras * np.log(n)
    #println(bic)
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def claecm9(z, x, q, p, g, n, llambda, omega, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.transpose(np.reshape(llambda[0:q*p], (q, p), order='F'))
    delta = np.ones(p)
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z9(x, z, llambda, omega, delta, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            psi = np.dot(omega[j],  delta)
            beta[j] = update_beta2(psi, llambda, p, q)
        theta = {}
        for j in range(0, g):
            theta[j] = update_theta(beta[j], llambda, sampcovtilde[j], p, q)
        llambda = update_lambda2(beta, sampcovtilde, theta, n1, omega, p, q, g)
        for j in range(0, g):
            omega[j] = update_omega(llambda, delta, beta[j], sampcovtilde[j], theta[j], p, q)
        delta = update_delta(llambda, omega, beta, sampcovtilde, theta, n1, p, q, n, g)
        log_detpsi = np.zeros(g)
        log_detsig = np.zeros(g)
        log_c = np.zeros(g)
        for j in range(0, g):
            psi = np.dot(omega[j], delta)
            log_detpsi[j] = p * np.log(omega[j])
            log_detsig[j] = update_det_sigma_new2(llambda, psi, log_detpsi[j], p, q)
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z9(x, z, llambda, omega, delta, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + p*q - q*(q-1)/2 + g + (p-1)
    bic = 2*l[it-1] - paras * np.log(n)
    #println(l)
    for i in range(0, p):
        omega[g+i] = delta[i]
    if flag == -1:
        return z, -np.inf, llambda, omega
    return z, bic, llambda, omega


def claecm10(z, x, q, p, g, n, llambda, omega, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.reshape(llambda, (q*p, g), order='F')
    delta = np.ones(p)
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z10(x, z, llambda, omega, delta, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            psi = np.dot(omega[j], delta)
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            beta[j] = update_beta2(psi, llambda0, p, q)
        theta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
        for j in range(0, g):
            t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
            llambda[:, j] = (np.transpose(t_llambda)).flatten('F')
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            omega[j] = update_omega2(llambda0, delta, beta[j], sampcovtilde[j], p, q)
        delta = update_delta2(llambda, omega, beta, sampcovtilde, theta, n1, p, q, n, g)
        log_detpsi = np.zeros(g)
        log_detsig = np.zeros(g)
        log_c = np.zeros(g)
        for j in range(0, g):
            psi = np.dot(omega[j], delta)
            log_detpsi[j] = p * np.log(omega[j])
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            log_detsig[j] = update_det_sigma_new2(llambda0, psi, log_detpsi[j], p, q)
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z10(x, z, llambda, omega, delta, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + g + (p-1)
    bic = 2*l[it-1] - paras * np.log(n)
    #println(bic)
    for i in range(0, p):
        omega[g+i] = delta[i]
    if flag == -1:
        return z, -np.inf, llambda, omega
    return z, bic, llambda, omega


def claecm11(z, x, q, p, g, n, llambda, omega, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.transpose(np.reshape(llambda[0:q*p], (q, p), order='F'))
    delta = np.ones((g, p))
    omega = omega[0]
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z11(x, z, llambda, omega, delta, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            psi = np.dot(omega, delta[j, :])
            beta[j] = update_beta2(psi, llambda, p, q)
        theta = {}
        for j in range(0, g):
            theta[j] = update_theta(beta[j], llambda, sampcovtilde[j], p, q)
        del0 = (np.transpose(delta)).flatten('F')
        llambda = update_lambda_cuu(beta, sampcovtilde, theta, n1, del0, p, q, g)
        omega = 0
        for j in range(0, g):
            delta0 = delta[j, :]
            omega += pyi[j] * update_omega(llambda, delta0, beta[j], sampcovtilde[j], theta[j], p, q)
        for j in range(0, g):
            delta0 = delta[j, :]
            delta_temp = update_delta3(llambda, omega, beta[j], sampcovtilde[j], theta[j], n1[j], p, q)
            delta[j, :] = delta_temp
        log_detpsi = p * np.log(omega)
        log_detsig = np.zeros(g)
        log_c = np.zeros(g)
        for j in range(0, g):
            psi = np.dot(omega, delta[j, :])
            log_detsig[j] = update_det_sigma_new2(llambda, psi, log_detpsi, p, q)
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z11(x, z, llambda, omega, delta, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + p*q - q*(q-1)/2 + 1 + g*(p-1)
    #print(l)
    bic = 2*l[it-1] - paras * np.log(n)
    psi = np.zeros(g * p + 1)
    psi[0] = omega
    k = 0
    for j in range(0, g):
        for i in range(0, p):
            psi[k] = delta[j, i]
            k += 1
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def claecm12(z, x, q, p, g, n, llambda, omega, tol):
    l = []
    at = []
    it = 0
    flag = 0
    log_c = 0
    llambda = np.reshape(llambda, (q*p, g), order='F')
    delta = np.ones((g, p))
    omega = omega[0]
    while flag == 0:
        n1 = np.sum(z, axis=0)
        pyi = n1/n
        mu = update_mu(n1, x, z, g, n, p)
        if it > 0:
            tmpzv = update_z12(x, z, llambda, omega, delta, mu, pyi, log_c, n, g, p, q)
            z = tmpzv[0]
            v = tmpzv[1]
            max_v = tmpzv[2]
        sampcovtilde = update_sg(x, z, mu, g, n, p, n1)
        beta = {}
        for j in range(0, g):
            psi = np.dot(omega, delta[j, :])
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            beta[j] = update_beta2(psi, llambda0, p, q)
        theta = {}
        for j in range(0, g):
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            theta[j] = update_theta(beta[j], llambda0, sampcovtilde[j], p, q)
        for j in range(0, g):
            t_llambda = update_lambda(beta[j], sampcovtilde[j], theta[j], p, q)
            llambda[:, j] = (np.transpose(t_llambda)).flatten('F')
        omega = 0
        for j in range(0, g):
            delta0 = delta[j, :]
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            omega += pyi[j] * update_omega2(llambda0, delta0, beta[j], sampcovtilde[j], p, q)
        for j in range(0, g):
            delta0 = delta[j, :]
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            delta_temp = update_delta3(llambda0, omega, beta[j], sampcovtilde[j], theta[j], n1[j], p, q)
            delta[j, :] = delta_temp
        log_detpsi = p * np.log(omega)
        log_detsig = np.zeros(g)
        log_c = np.zeros(g)
        for j in range(0, g):
            psi = np.dot(omega, delta[j, :])
            llambda0 = np.transpose(np.reshape(llambda[:, j], (q, p), order='F'))
            log_detsig[j] = update_det_sigma_new2(llambda0, psi, log_detpsi, p, q)
            log_c[j] = (p/2) * np.log(2*np.pi) + 0.5 * log_detsig[j]
        tmpzv = update_z12(x, z, llambda, omega, delta, mu, pyi, log_c, n, g, p, q)
        z = tmpzv[0]
        v = tmpzv[1]
        max_v = tmpzv[2]
        stop = convergtest_new(l, at, max_v, v, n, it, g, tol)
        flag = stop[0]
        l = stop[1]
        at = stop[2]
        it += 1
    paras = g-1 + g*p + g*(p*q - q*(q-1)/2) + 1 + g*(p-1)
    #print(l)
    bic = 2*l[it-1] - paras * np.log(n)
    psi = np.zeros(g * p + 1)
    psi[0] = omega
    k = 1 #global
    for j in range(0, g):
        for i in range(0, p):
            psi[k] = delta[j, i]
            k += 1
    if flag == -1:
        return z, -np.inf, llambda, psi
    return z, bic, llambda, psi


def run_pgmm(x, z, bic, cls, q, p, g, n, model, clust, lamda, psi, tol):
    functype = [aecm, aecm2, aecm3, aecm4, aecm5, aecm6, aecm7, aecm8, aecm9, aecm10, aecm11, aecm12]
    functype2 = [claecm, claecm2, claecm3, claecm4, claecm5, claecm6, claecm7, claecm8, claecm9, claecm10,
                 claecm11, claecm12]
    if clust != 0:
        func = functype[model]
        out = func(z, x, cls, q, p, g, n, lamda, psi, tol)
    else:
        func2 = functype2[model]
        out = func2(z, x, q, p, g, n, lamda, psi, tol)
    z = out[0]
    bic = out[1]
    lamda = out[2]
    psi = out[3]
    return z, bic, lamda, psi


def init_load(x4, z4, g4, p4, q4):
    sampcov = {}
    for g in range(0, g4):
        sampcov[g] = np.zeros((p4, p4))
    mu = np.zeros((g4, p4))
    n = np.sum(z4, axis=0)
    pi = np.sum(z4, axis=0)/x4.shape[0]
    for g in range(0, g4):
        temp_sum = x4.T*z4[:, g]
        mu[g][:] = np.sum(temp_sum, axis=1)/n[g]
    for g in range(0, g4):
        sampcov[g] = np.cov(x4, aweights=z4[:, g], rowvar=False)*(sum(z4[:, g]**2)-1)/sum(z4[:, g])
    lmbda = {}
    lmbda1_temp = np.zeros(p4*q4*g4)
    s = 0
    for g in range(0, g4):
        eival = np.linalg.eig(sampcov[g])[0]
        evec = np.linalg.eig(sampcov[g])[1]
        for i in range(0, p4):
            for j in range(0, q4):
                lmbda1_temp[s] = math.sqrt(eival[j])*evec[i, 0]
                s = s+1
    lmbda["sep"] = lmbda1_temp
    psi = {}
    lam_mat = {}
    k4 = 0
    for g in range(1, g4+1):
        lam_mat[g-1] = lmbda1_temp[k4:g*p4*q4].reshape((p4, q4))
        k4 = p4*q4*g
    temp_p6 = np.zeros(p4)
    for g in range(0, g4):
        t1 = np.diagonal(sampcov[g] - np.dot(lam_mat[g], np.transpose(lam_mat[g])))
        temp_p6 = temp_p6 + pi[g] * abs(t1)
    psi[6] = temp_p6
    psi[5] = sum(psi[6])/p4
    psi_temp = np.zeros((g4, p4))
    for g in range(0, g4):
        psi_temp[g][:] = abs(np.diagonal(sampcov[g] - np.dot(lam_mat[g], np.transpose(lam_mat[g]))))
    psi[7] = np.mean(psi_temp, axis=1)
    psi[8] = psi_temp.flatten()
    stilde = np.zeros((p4, p4))
    for g in range(0, g4):
        stilde = stilde + pi[g]*sampcov[g]
    eival = np.linalg.eig(stilde)[0]
    evec = np.linalg.eig(stilde)[1]
    lmbda_tilde = copy.deepcopy(lmbda1_temp)  # np.zeros((p4, q4))  # not used in R either
    s = 0
    for i in range(0, p4):
        for j in range(0, q4):
            lmbda_tilde[s] = math.sqrt(eival[j])*evec[i, 0]
            s = s+1
    lmbda["tilde"] = lmbda_tilde
    lam_mat[1] = np.array(lmbda_tilde[0:p4*q4])
    lam_mat[1] = lam_mat[1].reshape((p4, q4))
    psi[2] = abs(np.diagonal(stilde - np.dot(lam_mat[1], np.transpose(lam_mat[1]))))
    psi[1] = sum(psi[2])/p4
    psi_temp = np.zeros((g4, p4))
    for g in range(0, g4):
        psi_temp[g][:] = abs(np.diagonal(sampcov[g] - np.dot(lam_mat[1], np.transpose(lam_mat[1]))))
    psi[3] = np.mean(psi_temp, axis=1)
    psi[4] = psi_temp.flatten()
    psi[9] = np.concatenate((psi[3], np.zeros(p4)))
    psi[10] = np.concatenate((psi[7], np.zeros(p4)))
    t1 = np.zeros(g4*p4+1)
    t1[0] = psi[1]
    psi[11] = t1
    t1 = np.zeros(g4*p4+1)
    t1[0] = psi[5]
    psi[12] = t1
    lmbda["psi"] = psi
    return lmbda


def end_print(icl, zstart, loop, m_best, q_best, g_best, bic_best, class_ind):
    start_names = ('Blank Index', 'NA', 'K-Means', 'Custom')
    if not class_ind:
        if not icl:
            if zstart == 1:
                if loop == 1:
                    print("Based on 1 random start, the best model (BIC) for the range of factors and components used "
                          "is a " + str(m_best) + " model with q = " + str(q_best) + " and G = " + str(g_best)
                          + ".The BIC for this model is " + str(bic_best) + ".")
                else:
                    print("Based on " + str(loop) + " random starts, the best model (BIC) for the range of factors and "
                                                    "components used is a " + str(m_best) + " model with q = " + str(
                        q_best) + " and G = "
                          + str(g_best) + ".The BIC for this model is " + str(bic_best) + ".")
            else:
                print("Based on " + str(
                    start_names[zstart]) + " starting values, the best model (BIC) for the range of factors and "
                                           "components used is a " + str(m_best) + " model with q = " + str(
                    q_best) + " and G = "
                      + str(g_best) + ".The BIC for this model is " + str(bic_best) + ".")
        else:
            if zstart == 1:
                if loop == 1:
                    print("Based on 1 random start, the best model (ICL) for the range of factors and components used "
                          "is a " + str(m_best) + " model with q = " + str(q_best) + " and G = " + str(g_best)
                          + ".The ICL for this model is " + str(bic_best) + ".")
                else:
                    print("Based on " + str(loop) + " random starts, the best model (ICL) for the range of factors and "
                                                    "components used is a " + str(m_best) + " model with q = " + str(
                        q_best) + " and G = "
                          + str(g_best) + ".The ICL for this model is " + str(bic_best) + ".")
            else:
                print("Based on " + str(
                    start_names[zstart]) + " starting values, the best model (ICL) for the range of factors and "
                                           "components used is a " + str(m_best) + " model with q = " + str(
                    q_best) + " and G = "
                      + str(g_best) + ".The ICL for this model is " + str(bic_best) + ".")
    else:
        if not icl:
            print(
                "Based on the labelled and unlabelled data provided, the best model (BIC) for the range of factors and "
                "components used is a " + str(m_best) + " model with q = " + str(q_best) + " and G = " + str(g_best)
                + ".The BIC for this model is " + str(bic_best) + ".")
        else:
            print(
                "Based on the labelled and unlabelled data provided, the best model (ICL) for the range of factors and "
                "components used is a " + str(m_best) + " model with q = " + str(q_best) + " and G = " + str(g_best)
                + ".The ICL for this model is " + str(bic_best) + ".")


def pgmm_em(rg=range(2, 4), rq=range(1, 4), klass=None, icl=False, zstart=2, cccstart=True,
            loop=3, zlist=None, modelss=None, seeder=123456, tol=0.1, relax=False):
    time_in = time.time()
    get_params = read_job_cmd()
    rg = range(get_params[0], get_params[1]+1)
    rq = range(get_params[2], get_params[3]+1)
    models_num = range(get_params[4], get_params[5]+1)
    n = get_params[6]
    p1 = get_params[7]-1
    p2 = get_params[8]
    rl = get_params[9]  # 1 #Number of Total Loops
    sl = get_params[10]  # 6 #Number of Start Loops
    ctrue = get_params[11]-1  # 1 for wine #2 for coffee #Column of Known True Clusters
    seeder = get_params[12]
    headers = get_params[13]
    data_name = get_params[14]
    send_count = 0
    p = len(range(p1, p2))
    print("PGMM paramters G =", rg, "Q =", rq, "and Model = ", models_num,". Number of Random Starts per G is Q *", sl, "for ", rl,
          "loops. The seed is set to ", seeder, ". Applying to Data = ", data_name, ".")
    # df_raw = Array{Float64}(undef, 0)
    if headers == 1:
        df_raw = pd.read_csv(data_name, sep=" ")
    elif headers == 0:
        df_raw = pd.read_csv(data_name, sep=" ", header=None)
    df_raw_col = df_raw.iloc[:, p1:p2]
    # x = convert(Matrix{Float64}, df_raw_col)
    x1 = df_raw_col.values
    #random.seed(seeder)
    models_all = ('CCC', 'CCU', 'CUC', 'CUU', 'UCC', 'UCU', 'UUC', 'UUU', 'CCUU', 'UCUU', 'CUCU', 'UUCU')
    #models_num = range(0, 12)
    #if modelss is None:
    #    modelsubset = models_all
    #    models_num = range(0, 12)
    #else:
    #    modelsubset = []
    #    models_num = [modelss]
    #    for im in range(0, len(models_num)):
    #        modelsubset.append(models_all[models_num[im]-1])
    bic_out = {}
    gmin = rg[0]
    gmax = max(rg)+1
    qmin = rq[0]
    qmax = max(rq)+1
    g_offset = gmin-1
    q_offset = qmin-1
    bic_max = -np.inf
    bic_best = -np.inf
    test_paras = [qmax, qmin, gmax, gmin]
    if klass is None:
        klass = np.zeros(n)
        klass_ind = 0
    else:
        klass_ind = 1
    for mod in range(0, len(models_num)):
        bic_temp = np.zeros((gmax-gmin+1, qmax-qmin+1))
        bic_temp = pd.DataFrame(bic_temp, index=range(min(rg), max(rg)+2), columns=range(min(rq), max(rq)+2))
        bic_out[mod] = bic_temp
    if klass_ind:
        for g1 in rg:
            for m in range(0, len(models_num)):
                bic_out[m] = np.zeros((max(rg)-g_offset, max(rq)-q_offset))
            zt = np.zeros((n, g1))
            kls_ind = klass == 0
            for i in range(0, n):
                if (kls_ind[i]):
                    zt[i][:] = 1/g1
                else:
                    zt[i][klass[i]] = 1
            z1 = zt.flatten()
            lmda = {}
            for q1 in rq:
                lmda[q1 - q_offset] = init_load(x1, zt, g1, p, q1)
            for m in range(0, len(models_num)):
                for q1 in rq:
                    if models_all[m][0] == "C":
                        lam_temp = lmda[q1 - q_offset]["tilde"]
                    else:
                        lam_temp = lmda[q1 - q_offset]["sep"]
                    psi_temp = lmda[q1 - q_offset]["psi"][models_num[m]]
                    temp = run_pgmm(x1, z1, 0, klass, q1, p, g1, n, models_num[m]-1, klass_ind, lam_temp, psi_temp, tol)
                    bic_out[m][(g1 - g_offset)-1, (q1 - q_offset)-1] = temp[1]
                    if not math.isnan(temp[1]):
                        if icl and g1 > 1:
                            z_mat_tmp = temp[0]
                            z_mat_tmp = z_mat_tmp.reshape((n, g1))
                            mapz = np.zeros(n)
                            for i9 in range(0, n):
                                mapz[i9] = np.argmax(z_mat_tmp[i9][0:g1])
                            icl2 = 0
                            for i9 in range(0, n):
                                icl2 = icl2 + math.log(z_mat_tmp[i9][mapz[i9]])
                            bic_out[m][(g1 - g_offset)-1, (q1 - q_offset)-1] = \
                                bic_out[m][(g1 - g_offset)-1, (q1 - q_offset)-1]+2*icl2
                        if temp[1] > bic_max:
                            z_best = temp[0]
                            bic_best = temp[2]
                            bic_max = bic_best
                            g_best = g1
                            q_best = q1
                            m_best = models_num[m]
                            lmda_best = temp[2]
                            psi_best = temp[3]
    else:
        bic_start = np.zeros((gmax - gmin + 1, qmax - qmin + 1))
        if zstart == 1:
            for l in range(1, loop + 1):
                for g1 in rg:
                    z = np.zeros((n, g1))
                    for i in range(0, n):
                        summ = 0
                        for j in range(0, g1):
                            z[i][j] = np.random.uniform(0, 1, 1)
                            summ = summ + z[i][j]
                        for j in range(0, g1):
                            z[i][j] = z[i][j] / summ
                    z1 = z.flatten()
                    lmda = {}
                    for q1 in rq:
                        lmda[q1 - q_offset] = init_load(x1, z, g1, p, q1)
                    if cccstart:
                        bic_ccc_max = -np.inf
                        for q1 in rq:
                            temp = run_pgmm(x1, z1, 0, klass, q1, p, g1, n, 1, klass_ind, lmda[q1 - q_offset]["tilde"],
                                            lmda[q1 - q_offset]["psi"][1], tol)
                            bic_start[g1 - g_offset][q1 - q_offset] = temp[1]
                            if not math.isnan(temp[1]):
                                if icl and g1 > 1:
                                    z_mat_tmp = temp[0]
                                    z_mat_tmp = z_mat_tmp.reshape((n, g1))
                                    mapz = np.zeros(n)
                                    for i9 in range(0, n):
                                        mapz[i9] = np.argmax(z_mat_tmp[i9][0:g1])
                                    icl1 = 0
                                    if (icl):
                                        for i9 in range(0, n):
                                            icl1 = icl1 + math.log(z_mat_tmp[i9][mapz[i9]])
                                    bic_start[g1 - g_offset][q1 - q_offset] = bic_start[g1 - g_offset][
                                                                                  q1 - q_offset] + 2 * icl1
                                if bic_start[g1 - g_offset][q1 - q_offset] > bic_ccc_max:
                                    z_init_best = temp[0]
                                    bic_ccc_max = bic_start[g1 - g_offset][q1 - q_offset]
                        z_init_mat = np.array(z_init_best)
                        z_init_mat = z_init_mat.reshape((n, g1))
                        for q1 in rq:
                            lmda[q1 - q_offset] = init_load(x1, z_init_mat, g1, p, q1)
                        z1 = z_init_mat.flatten()
                    for m in range(0, len(models_num)):
                        for q1 in rq:
                            if models_all[m][0] == "C":
                                lam_temp = lmda[q1 - q_offset]["tilde"]
                            else:
                                lam_temp = lmda[q1 - q_offset]["sep"]
                            psi_temp = lmda[q1 - q_offset]["psi"][models_num[m]]
                            temp = run_pgmm(x1, z1, 0, klass, q1, p, g1, n, models_num[m]-1, klass_ind, lam_temp,
                                            psi_temp, tol)
                            if not math.isnan(temp[1]):
                                if icl and g1 > 1:
                                    z_mat_tmp = temp[0]
                                    z_mat_tmp = z_mat_tmp.reshape((n, g1))
                                    mapz = np.zeros(n)
                                    for i9 in range(0, n):
                                        mapz[i9] = np.argmax(z_mat_tmp[i9][0:g1])
                                    icl2 = 0
                                    for i9 in range(0, n):
                                        icl2 = icl2 + math.log(z_mat_tmp[i9][mapz[i9]])
                                    temp[1] = temp[1] + 2 * icl2
                                if l == 1:
                                    bic_out[m][g1 - g_offset][q1 - q_offset] = temp[1]
                                elif math.isnan(bic_out[m][g1 - g_offset][q1 - q_offset]):
                                    bic_out[m][g1 - g_offset][q1 - q_offset] = temp[1]
                                elif bic_out[m][g1 - g_offset][q1 - q_offset] < temp[1]:
                                    bic_out[m][g1 - g_offset][q1 - q_offset] = temp[1]
                                if temp[1] > bic_max:
                                    z_best = temp[0]
                                    bic_best = temp[1]
                                    bic_max = bic_best
                                    g_best = g1
                                    q_best = q1
                                    m_best = models_num[m]
                                    lmda_best = temp[2]
                                    psi_best = temp[3]
        elif zstart == 2 or zstart == 3:
            for m in range(0, len(models_num)):
                bic_out[m] = np.zeros((max(rg)-g_offset, max(rq)-q_offset))
            for g1 in rg:
                z = np.zeros((n, g1))
                if zstart == 3:
                    if g1 == 1:
                        z_ind = np.ones(n)
                    else:
                        z_ind = zlist[g1]
                if zstart == 2:
                    if g1 == 1:
                        z_ind = np.ones(n)
                    else:
                        random.seed(seeder)
                        z_ind = KMeans(n_clusters=g1, n_init=5, random_state=seeder).fit(x1)
                        z_ind = z_ind.labels_
                for i in range(0, n):
                    z[i][z_ind[i]] = 1
                lmda = {}
                for q1 in rq:
                    lmda[q1 - q_offset] = init_load(x1, z, g1, p, q1)
                #z1 = z #.flatten()
                for m in range(0, len(models_num)):
                    for q1 in rq:
                        z1 = copy.deepcopy(z)
                        if models_all[m][0] == "C":
                            lam_temp = copy.deepcopy(lmda[q1 - q_offset]["tilde"])
                        else:
                            lam_temp = copy.deepcopy(lmda[q1 - q_offset]["sep"])
                        psi_temp = copy.deepcopy(lmda[q1 - q_offset]["psi"][models_num[m]])
                        #print("Running for G = ", g1, "Q = ", q1, "and M = ", models_num[m])
                        temp = run_pgmm(x1, z1, 0, klass, q1, p, g1, n, (models_num[m]-1), klass_ind, lam_temp,
                                        psi_temp, tol)
                        bic_out[m][(g1 - g_offset)-1, (q1 - q_offset)-1] = temp[1]
                        #print("G = ", g1, " Q = ", q1, " Model = ", (models_num[m]), " BIC = ", temp[1])
                        if not math.isnan(temp[1]):
                            if icl and g1 > 1:
                                z_mat_tmp = temp[0]
                                z_mat_tmp = z_mat_tmp.reshape((n, g1))
                                mapz = np.zeros(n)
                                for i9 in range(0, n):
                                    mapz[i9] = np.argmax(z_mat_tmp[i9, :])
                                icl2 = 0
                                for i9 in range(0, n):
                                    icl2 = icl2 + math.log(z_mat_tmp[i9, int(mapz[i9])])
                                bic_out[m][(g1 - g_offset)-1, (q1 - q_offset)-1] = \
                                    bic_out[m][(g1 - g_offset)-1, (q1 - q_offset)-1] + 2 * icl2
                            if temp[1] > bic_max:
                                z_best = temp[0]
                                bic_best = bic_out[m][(g1 - g_offset)-1, (q1 - q_offset)-1]
                                bic_max = bic_best
                                g_best = g1
                                q_best = q1
                                z_mat = z_best.reshape((n, g_best))
                                m_best = models_all[m]
                                lmda_best = temp[2]
                                psi_best = temp[3]
                        else:
                            g_best = -1
                            z_mat = math.nan
        else:
            print("Invalid entry for ztsart: 1 random; 2 k-means; 3 user-specified list.")
    time_out = time.time()
    #print(bic_out)
    print("Time taken is", time_out - time_in)
    if not math.isnan(temp[1]):
        if m_best[0] == "C":
            lmda_mat = np.array(lmda_best)
            lmda_mat = lmda_mat[0:(q_best*p)].reshape((p, q_best))
        else:
            lmda_mat = {}
            for g1 in range(1, g_best+1):
                upper = (q_best*p)*g1
                lmda_mat[g1] = np.array(lmda_best[(upper-(p*q_best)):(upper)])
                lmda_mat[g1] = lmda_mat[g1].reshape((p, q_best))
        if m_best == "CUU" or m_best == "UUU":
            psi_mat = {}
            for g1 in range(1, g_best+1):
                upper = p*g1
                psi_mat_temp = np.zeros((p, p))
                psi_mat[g1] = np.fill_diagonal(psi_mat_temp, psi_best[(upper-p):upper])
        elif m_best == "CCC" or m_best == "UCC":
            psi_mat = psi_best #[0]
        elif m_best == "CCU" or m_best == "UCU":
            psi_mat = psi_best[0:p]
        elif m_best == "CUC" or m_best == "UUC":
            psi_mat = {}
            for g1 in range(1, g_best+1):
                psi_mat[g1] = psi_best[g1-1]
        elif m_best == "CCUU" or m_best == "UCUU":
            psi_mat = {}
            psi_mat["omega"] = psi_best[0:g_best]
            psi_mat_temp = np.zeros((p, p))
            psi_mat["delta"] = np.fill_diagonal(psi_mat_temp, psi_best[(g_best):(g_best+p)])
        elif m_best == "CUCU" or m_best == "UUCU":
            psi_mat = {}
            psi_mat["omega"] = psi_best[0]
            for g1 in range(1, g_best+1):
                temp_string = "delta " + str(g1)
                lower = 2+(g1-1)*p
                psi_mat_temp = np.zeros((p, p))
                psi_mat["delta"] = np.fill_diagonal(psi_mat_temp, psi_best[lower:(lower+p-1)])
        z_mat = np.array(z_best)
        z_mat = z_mat.reshape((n, g_best))
    else:
        z_mat = math.nan
    #print(bic_out)
    if g_best > 0:
        klass_best = np.zeros(n)
        for i in range(0, n):
            klass_best[i] = np.argmax(z_mat[i][0:g_best])
        tab_res = pd.crosstab(df_raw.iloc[:, ctrue], klass_best)
        ari = adjusted_rand_score(df_raw.iloc[:, ctrue], klass_best)
        print(tab_res)
        print("The ARI of the best result is ARI = ", ari)
        end_print(icl, zstart, loop, m_best, q_best, g_best, bic_best, klass_ind)
        if not icl:
            return klass_best, m_best, g_best, q_best, bic_out, z_mat, lmda_mat, psi_mat, bic_best
        else:
            return klass_best, m_best, g_best, q_best, bic_out, z_mat, lmda_mat, psi_mat, bic_best
    else:
        print("G_Best is not greater than 0.")


pgmm_em()

