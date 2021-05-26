from mpi4py import MPI
import multiprocessing
import pandas as pd
import math
import numpy as np
from sklearn.cluster import KMeans
import subprocess
import time
import sys
import os
import random
#import scipy
import itertools
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
    ntasks_node = int(sys.argv[15])
    data_in = str(sys.argv[16])
    return gmin, gmax, qmin, qmax, tmin, tmax, n_in, p1_in, p2_in, trun, srun, ccolmn, seed_in, header_yn, ntasks_node, data_in


def run_pgmm(x, z, bic, kls, q8, p, g8, n, model, cluster, llambda, psi, tol):
    #path = os.getcwd()
    #path2 = "/tmp" + str(g8) + str(q8) + str(model)
    np.savetxt("x" + str(g8) + str(q8) + str(model) + ".txt", x)
    np.savetxt("z" + str(g8) + str(q8) + str(model) + ".txt", z)
    np.savetxt("kls" + str(g8) + str(q8) + str(model) + ".txt", kls)
    np.savetxt("lam_temp" + str(g8) + str(q8) + str(model) + ".txt", llambda)
    if isinstance(psi, float) == True:
        np.savetxt("psi_temp" + str(g8) + str(q8) + str(model) + ".txt", psi[None])
    else:
        np.savetxt("psi_temp" + str(g8) + str(q8) + str(model) + ".txt", psi)
    p4 = subprocess.Popen(["Rscript --vanilla run_pgmm_prc2.R %s %s %s %s %s %s %s %s %s %s %s %s %s"
                           % ("x" + str(g8) + str(q8) + str(model), "z" + str(g8) + str(q8) + str(model), str(bic),
                              "kls" + str(g8) + str(q8) + str(model), str(q8), str(p), str(g8), str(n), str(model),
                              str(cluster), "lam_temp" + str(g8) + str(q8) + str(model),
                              "psi_temp" + str(g8) + str(q8) + str(model), str(tol))],
                          stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True).communicate()[0]
    zbest = pd.read_csv("zbest" + str(g8) + str(q8) + str(model) + ".txt", sep=" ", header=None)
    zbest = zbest.values
    bic_best = pd.read_csv("bicbest" + str(g8) + str(q8) + str(model) + ".txt", sep=" ", header=None)
    bic_best = bic_best.values[0][0]
    lmbda_best = pd.read_csv("lambdabest" + str(g8) + str(q8) + str(model) + ".txt", sep=" ", header=None)
    lmbda_best = lmbda_best.values
    psi_best = pd.read_csv("psibest" + str(g8) + str(q8) + str(model) + ".txt", sep=" ", header=None)
    psi_best = psi_best.values
    os.remove("x" + str(g8) + str(q8) + str(model) + ".txt")
    os.remove("z" + str(g8) + str(q8) + str(model) + ".txt")
    os.remove("kls" + str(g8) + str(q8) + str(model) + ".txt")
    os.remove("lam_temp" + str(g8) + str(q8) + str(model) + ".txt")
    os.remove("psi_temp" + str(g8) + str(q8) + str(model) + ".txt")
    os.remove("zbest" + str(g8) + str(q8) + str(model) + ".txt")
    os.remove("bicbest" + str(g8) + str(q8) + str(model) + ".txt")
    os.remove("lambdabest" + str(g8) + str(q8) + str(model) + ".txt")
    os.remove("psibest" + str(g8) + str(q8) + str(model) + ".txt")
    return zbest, bic_best, lmbda_best, psi_best


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
        eival = np.linalg.eig(sampcov[g])[0].real
        evec = np.linalg.eig(sampcov[g])[1].real
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
    eival = np.linalg.eig(stilde)[0].real
    evec = np.linalg.eig(stilde)[1].real
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


def work(x, gg, qq, mm, n, p, seed, index, rank):
    print("Starting work for G = {} Q = {}, and Model = {} on Rank {}".format(gg, qq, mm, rank))
    mm = mm-1
    m_all = ["CCC", "CCU", "CUC", "CUU", "UCC", "UCU", "UUC", "UUU", "CCUU", "UCUU", "CUCU", "UUCU"]
    tol = 0.1
    icl = False
    klass = np.zeros(n)
    klass_ind = 0
    z_init = np.zeros((n, gg))
    if gg == 1:
        z_ind = np.ones(n)
    else:
        random.seed(seed)
        z_ind = KMeans(n_clusters=gg, n_init=5, random_state=seed).fit(x)
        z_ind = z_ind.labels_
    for i in range(0, n):
        z_init[i][z_ind[i]] = 1
    z1 = copy.deepcopy(z_init)
    lmda = init_load(x, z1, gg, p, qq)
    if m_all[mm][0] == "C":
        lam_temp = copy.deepcopy(lmda["tilde"])
    else:
        lam_temp = copy.deepcopy(lmda["sep"])
    psi_temp = copy.deepcopy(lmda["psi"][mm+1])
    temp = run_pgmm(x, z1, 0, klass, qq, p, gg, n, mm+1, klass_ind, lam_temp, psi_temp, tol)
    #print(temp[1])
    if not math.isnan(temp[1]):
        if icl and gg > 1:
            z_mat_tmp = temp[0]
            z_mat_tmp = z_mat_tmp.reshape((n, gg))
            mapz = np.zeros(n)
            for i9 in range(0, n):
                mapz[i9] = np.argmax(z_mat_tmp[i9, :])
            icl2 = 0
            for i9 in range(0, n):
                icl2 = icl2 + math.log(z_mat_tmp[i9, int(mapz[i9])])
                bic_out = temp[1] + 2 * icl2
        z_best = temp[0]
        bic_out = temp[1]
        lambda_best = temp[2]
        psi_best = temp[3]
    else:
        z_best = np.zeros(n * gg)
        bic_out = -np.Inf
    return bic_out, z_best, gg, qq, mm+1


def master(comm, size, rank, status):
    time_in = time.time()
    bic_save = -np.Inf
    g_b = -np.Inf
    q_b = -np.Inf
    m_b = -np.Inf
    get_params = read_job_cmd()
    rg = range(get_params[0], get_params[1] + 1)
    rq = range(get_params[2], get_params[3] + 1)
    rt = range(get_params[4], get_params[5] + 1)
    n = get_params[6]
    p1 = get_params[7] - 1
    p2 = get_params[8]
    rl = get_params[9]
    sl = get_params[10]
    ctrue = get_params[11] - 1
    seeder = get_params[12]
    headers = get_params[13]
    ntasks_node = get_params[14]
    data_name = get_params[15]
    p = len(range(p1, p2))
    #df_raw = Array{Float64}(undef, 0)
    print("There is ", size, "total processors")
    print("PGMM paramters G =", rg, "Q =", rq, "and Model = ", rt, ". Number of Random Starts per G is Q *", sl,
          "for ", rl, "loops. The seed is set to ", seeder, ". Applying to Data = ", data_name, ".")
    # df_raw = Array{Float64}(undef, 0)
    if headers == 1:
        df_raw = pd.read_csv(data_name, sep=" ")
    elif headers == 0:
        df_raw = pd.read_csv(data_name, sep=" ", header=None)
    df_raw_col = df_raw.iloc[:, p1:p2]
    # x = convert(Matrix{Float64}, df_raw_col)
    x1 = df_raw_col.values
    random.seed(seeder)
    models_all = ["CCC","CCU","CUC","CUU","UCC","UCU","UUC","UUU","CCUU","UCUU","CUCU","UUCU"]
    m_min = min(rt)
    m_max = max(rt)+1
    g_min = min(rg)
    g_max = max(rg)+1
    q_index = 1
    q_index_max = len(rq)+1
    q_max = len(rq)+1
    run_loop_index = 1
    run_loop = rl + 1
    num_workers = size - 1
    closed_workers = 0
    send_count = 0
    #print(g_max)
    #print(m_max)
    #print(q_max)
    while closed_workers < num_workers:
        message_recv = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == 0:
            if q_index < q_max:
                send_count += 1
                message_send = [g_min, g_max, m_min, m_max, rq[q_index-1], n, p, seeder, ntasks_node]
                comm.send(message_send, dest=source, tag=3)
                comm.send(x1, dest=source, tag=3)
            if q_index < q_max:
                q_index += 1
            else:
                message_send = np.zeros(6)
                comm.send(message_send, dest=source, tag=2)
                closed_workers += 1
        elif tag == 1:
            bic_temp = message_recv[0]
            g_temp = math.trunc(message_recv[1])
            q_temp = math.trunc(message_recv[2])
            m_temp = math.trunc(message_recv[3])
            message_final_z = comm.recv(source=source, tag=tag, status=status)
            if message_recv[0] > bic_save and message_recv[1] != 0:
                bic_save = bic_temp
                g_b = g_temp
                q_b = q_temp
                m_b = m_temp
                z_final = message_final_z
        elif tag == 25:
            bic_temp = message_recv[0]
            g_temp = math.trunc(message_recv[1])
            q_temp = math.trunc(message_recv[2])
            m_temp = math.trunc(message_recv[3])
    time_out = time.time()
    z_mat_final = z_final.reshape((n, g_b))
    map_out = np.zeros(n)
    for i in range(0, n):
        map_out[i] = np.argmax(z_mat_final[i, :])
    print("This is estimate")
    print(map_out)
    print("This is true")
    print(df_raw.iloc[:, ctrue].values)
    print("This is Classification Table")
    pd.set_option('display.max_columns', None)
    tab_res = pd.crosstab(df_raw.iloc[:, ctrue], map_out, dropna=False)
    print(tab_res)
    print("The best BIC = ", bic_save, "for G = ", g_b, "Q = ", q_b, "and M = ", m_b)
    ari = adjusted_rand_score(df_raw.iloc[:, ctrue].values, map_out)
    print("The ARI of this result is ARI = ", ari)
    print("Time taken is ", (time_out - time_in))


def slave(comm, size, rank, status):
    message_recv = np.zeros(4)
    message_send = np.zeros(6)
    go = True
    while go == True:
        comm.send(message_recv, dest=0, tag=0)
        message_send = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == 3:
            gmin = math.trunc(message_send[0])
            gmax = math.trunc(message_send[1])
            mmin = math.trunc(message_send[2])
            mmax = math.trunc(message_send[3])
            qq = math.trunc(message_send[4])
            n = math.trunc(message_send[5])
            p = math.trunc(message_send[6])
            seed = math.trunc(message_send[7])
            ntask_workers = math.trunc(message_send[8])
            data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            x_df = data.reshape((n, p))
            rgs = range(gmin, gmax)
            rts = range(mmin, mmax)
            combo = [rgs, rts]
            set_tuple = list(itertools.product(*combo))
            set_tuple = np.asarray(set_tuple)
            index = range(0, len(set_tuple))
            set_tuple = np.c_[set_tuple, index]
            bic_save2 = -np.inf
            print('Calling Python multiprocessing process from Python MPI rank {} with Q = {} and {} workers'.format(rank, qq, ntask_workers))
            with multiprocessing.Pool(processes=ntask_workers) as pool:
                jobs = [pool.apply_async
                    (work, args=(x_df, gqm[0], qq, gqm[1], n, p, seed, gqm[2], rank))
                    for gqm in set_tuple
                    ]
                pool.close()
                pool.join()
                z_save = [p.get() for p in jobs]
            #print(z_save)
            ind_save = 0
            for ind in index:
                if z_save[ind][0] > bic_save2:
                    ind_save = ind
                    bic_save2 = z_save[ind][0]
            message_recv[0] = bic_save2
            message_recv[1] = int(z_save[ind_save][2])
            message_recv[2] = int(z_save[ind_save][3])
            message_recv[3] = int(z_save[ind_save][4])
            comm.send(message_recv, dest=0, tag=1)
            message_final_z = (z_save[ind_save][1]).flatten()
            comm.send(message_final_z, dest=0, tag=1)
        elif tag == 2:
            message_recv = np.zeros(4)
            comm.send(message_recv, dest=0, tag=25)
            go = False


def main():
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    status = MPI.Status()
    if rank == 0:
        master(comm, size, rank, status)
    else:
        slave(comm, size, rank, status)


main()
