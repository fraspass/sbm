#!/usr/bin/env python
import sys, os
import argparse
import numpy as np
from numpy.linalg import svd,eigh
import pandas as pd
import matplotlib.pyplot as plt
import mcmc_sampler_sbm
from estimate_cluster import estimate_clustering
from sklearn.cluster import KMeans
from scipy.stats import mode
from collections import Counter

## Boolean type for parser
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

###### MODEL ARGUMENTS 

## PARSER to give parameter values 
parser = argparse.ArgumentParser()
# Number of nodes to generate
parser.add_argument("-n","--nodes", type=int, dest="n", default=2500, const=True, nargs="?",\
    help="Integer: number of nodes, default 2500")
parser.add_argument("-n2","--nodes2", type=int, dest="n2", default=2500, const=True, nargs="?",\
    help="Integer: number of nodes for 2nd node set in bipartite graphs, default 2500")
# Latent dimension
parser.add_argument("-d","--dim", type=int, dest="d", default=2, const=True, nargs="?",\
    help="Integer: latent dimension, default 2")
# Redundant dimension of the embedding
parser.add_argument("-m", type=int, dest="m", default=50, const=True, nargs="?",\
    help="Integer: redundant dimension, default 50")
# Number of clusters
parser.add_argument("-K","--clust", type=int, dest="K", default=5, const=True, nargs="?",\
    help="Integer: number of clusters, default 5")
parser.add_argument("-K2","--clust2", type=int, dest="K2", default=3, const=True, nargs="?",\
    help="Integer: number of clusters for 2nd node set in coclustering, default 3")
# Boolean variable to generate directed graphs
parser.add_argument("-g","--dir", type=str2bool, dest="gen_directed", default=False, const=False, nargs="?",\
    help="Boolean variable for generating a directed graph, default FALSE")
# Boolean variable to generate bipartite graphs
parser.add_argument("-b","--bip", type=str2bool, dest="gen_bipartite", default=False, const=False, nargs="?",\
    help="Boolean variable for generating a bipartite graph, default FALSE")
# Boolean variable to use LSE (default ASE) - *** only if undirected graph is generated, must be used with -d 0 -b 0 (default) **
parser.add_argument("-l","--lap", type=str2bool, dest="use_laplacian", default=False, const=False, nargs="?",\
    help="Boolean variable for LSE, default FALSE (i.e. use ASE)")
# Boolean variable for coclustering
parser.add_argument("-c","--coclust", type=str2bool, dest="coclust", default=False, const=False, nargs="?",\
    help="Boolean variable for coclustering, default FALSE")
# Constrained/unconstrained model
parser.add_argument("-q","--constraint", type=str2bool, dest="constraint", default=False, const=False, nargs="?",\
    help="Boolean variable for the constrained/unconstrained model, default UNCONSTRAINED (0)")
# Run MCMC yes/no or simulate data only
parser.add_argument("-r","--run", type=str2bool, dest="run_mcmc", default=False, const=False, nargs="?",\
    help="Boolean variable for running MCMC on the simulated dataset. If TRUE, runs MCMC. If FALSE, produces summary plots on the simulated dataset. Default FALSE")
# Boolean variable to use second level clustering (default True)
parser.add_argument("-s","--sord", type=str2bool, dest="second_order_clustering", default=True, const=False, nargs="?",\
    help="Boolean variable for second level clustering, default TRUE")
# Burnin
parser.add_argument("-B","--nburn", type=int, dest="nburn", default=25000, const=True, nargs="?",\
    help="Integer: length of burnin, default 25000")
# Number of samples
parser.add_argument("-M","--nsamp", type=int, dest="nsamp", default=500000, const=True, nargs="?",\
    help="Integer: length of MCMC chain after burnin, default 500000")
## Set destination folder for output
parser.add_argument("-f","--folder", type=str, dest="dest_folder", default="Results", const=True, nargs="?",\
    help="String: name of the destination folder for the output files")

## Parse arguments
args = parser.parse_args()
n = args.n
m = args.m
n2 = args.n2
d = args.d
K = args.K
K2 = args.K2
gen_directed = args.gen_directed
gen_bipartite = args.gen_bipartite
use_laplacian = args.use_laplacian
coclust = args.coclust
constraint = args.constraint
run_mcmc = args.run_mcmc
second_order_clustering = args.second_order_clustering
nburn = args.nburn
nsamp = args.nsamp
dest_folder = args.dest_folder

# Create output directory if doesn't exist
if dest_folder != '' and not os.path.exists(dest_folder):
    os.mkdir(dest_folder)

## Check exceptions
if use_laplacian and (gen_directed or gen_bipartite):
    ValueError("LSE can only be used for undirected graphs.")
if gen_bipartite:
    gen_directed = gen_bipartite
if gen_bipartite:
    coclust = True

## Set seed
np.random.seed(14441)

#######################################
## Simulate a stochastic block model ##
#######################################

c = np.random.choice(K,size=n)
c2 = np.random.choice(K2,size=n2)

## Generate B from beta draws
B = -.5*np.ones((K,K2 if coclust else K))
i = 0
## Check that B is a proper matrix
while (B < 0).any() or (B > 1).any() and i <= 100:
    B = np.random.beta(1.2,1.2,size=(K,K2 if coclust else K))
    ## If the graph is undirected, generate a symmetric matrix
    if not gen_directed: 
        B = np.tril(B) + np.tril(B,k=-1).T
    ## Compute low rank approximation
    if not gen_directed: 
        w,v = eigh(B)
        w_mag = (-np.abs(w)).argsort()
        B = np.dot(np.dot(v[:,w_mag[:d]],np.diag(w[w_mag[:d]])),v[:,w_mag[:d]].T)
    else:
        u,s,v = svd(B)
        B = np.dot(np.dot(u[:,:d],np.diag(s[:d])),v[:d])
    i += 1

if i == 100:
    raise ValueError('After 100 simulations, no valid low-rank approximation of B has been obtained --> change parameters of Beta distribution.')

## Construct the adjacency matrix
if not gen_directed:
    A = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            A[i,j] = np.random.binomial(1,B[c[i],c[j]])
            A[j,i] = A[i,j]
else:
    A = np.zeros((n,n2 if coclust else n))
    for i in range(n):
        for j in range(n2 if coclust else n):
            A[i,j] = np.random.binomial(1,B[c[i],c2[j] if coclust else c[j]])

## Construct a Gibbs sampling object
g = mcmc_sampler_sbm.mcmc_sbm(np.dot(np.dot(np.diag(A.sum(axis=0)**(-.5)),A),np.diag(A.sum(axis=0)**(-.5))) if (not gen_directed) and use_laplacian else A, m=m)

## Initialise clustering
if run_mcmc:
    if coclust:
        g.init_cocluster(zs=KMeans(n_clusters=5).fit(g.X['s'][:,:5]).labels_,zr=KMeans(n_clusters=5).fit(g.X['r'][:,:5]).labels_)
    else:
        g.init_cluster(z=KMeans(n_clusters=5).fit(g.X['s'][:,:5]).labels_ if g.directed else KMeans(n_clusters=5).fit(g.X[:,:5]).labels_)
else:
    if coclust:
        g.init_cocluster(zs=c,zr=c2)
    else:
        g.init_cluster(z=c)

## Initialise d 
g.init_dim(d=5,delta=0.1,d_constrained=constraint)

## Initialise parameters
if not gen_directed:
    g.prior_gauss_left_undirected(mean0=np.zeros(g.m),Delta0=0.1*np.diag(np.var(g.X,axis=0)),kappa0=1.0,nu0=1.0)
    g.prior_gauss_right_undirected(sigma0=np.var(g.X,axis=0),lambda0=1.0)
else:
    g.prior_gauss_left_directed(mean0s=np.zeros(g.m),mean0r=np.zeros(g.m),
        Delta0s=0.1*np.diag(np.var(g.X['s'],axis=0)),Delta0r=0.1*np.diag(np.var(g.X['r'],axis=0)))
    g.prior_gauss_right_directed(sigma0s=np.var(g.X['s'],axis=0),sigma0r=np.var(g.X['r'],axis=0))

## Initialise variances
if second_order_clustering:
    if coclust:
        g.init_group_variance_coclust(vs=range(g.K['s']),vr=range(g.K['r']))
    else:
        g.init_group_variance_clust(v=range(g.K)) 

##### Plots #####

if not run_mcmc:

    cc = {}; cc['s'] = c; cc['r'] = c2
    
    ## Marginal likelihoods
    g.marginal_likelihoods_dimension()
    plt.figure()
    plt.plot(range(1,g.m+1),g.mlik[1:(g.m+1)],'x-',c='black')
    plt.legend('Marginal likelihoods',loc='lower left')
    plt.xlabel('$d$')
    if dest_folder != '':
        plt.savefig(dest_folder+'/marginal_likelihoods.pdf')
    else:
        plt.savefig('marginal_likelihoods.pdf')

    ## Plot first two dimensions of Xs or Xr - useful when d=2
    if gen_directed:
        for key in ['s','r']:
            plt.figure()
            plt.scatter(g.X[key][:,0],g.X[key][:,1],c=cc[key] if coclust else c)
            if dest_folder != '':
                plt.savefig(dest_folder+'/scatter12_'+key+'.pdf')
            else:
                plt.savefig('scatter12_'+key+'.pdf')
    else:
        plt.figure()
        plt.scatter(g.X[:,0],g.X[:,1],c=c)
        if dest_folder != '':
            plt.savefig(dest_folder+'/scatter12.pdf')
        else:
            plt.savefig('scatter12.pdf')

    ## Plot dimensions 3 and 4 of Xs or Xr - useful when d=2
    if gen_directed:
        for key in ['s','r']:
            plt.figure()
            plt.scatter(g.X[key][:,2],g.X[key][:,3],c=cc[key] if coclust else c)
            if dest_folder != '':
                plt.savefig(dest_folder+'/scatter34_'+key+'.pdf')
            else:
                plt.savefig('scatter34_'+key+'.pdf')
    else:
        plt.figure()
        plt.scatter(g.X[:,2],g.X[:,3],c=c)
        if dest_folder != '':
            plt.savefig(dest_folder+'/scatter34.pdf')
        else:
            plt.savefig('scatter34.pdf')

    ## Means
    if gen_directed:
        sa = {}
        for key in ['s','r']:
            sa = np.zeros((g.K[key] if coclust else g.K,15))
            for k in range(g.K[key] if coclust else g.K):
                sa[k] = g.X[key][(cc[key] if coclust else c) == k].mean(axis=0)[:15]
            sm = np.append(np.append(np.max(sa,axis=0),np.flip(np.min(sa,axis=0))),np.max(sa[:,0]))
            vv = np.append(np.append(np.arange(1,16),np.flip(np.arange(1,16))),1)
            plt.figure()
            plt.plot(range(1,16),g.X[key].mean(axis=0)[:15],c='black')
            plt.xlabel('$d$')
            plt.plot(vv,sm,'--',c='red')
            for k in range(g.K[key] if coclust else g.K):
                plt.plot(range(1,16),g.X[key][(cc[key] if coclust else c) == k].mean(axis=0)[:15],'x',c='gray')
            plt.legend(['Overall mean','Max./min. within-cluster mean','Within-cluster mean'])
            if dest_folder != '':
                plt.savefig(dest_folder+'/mean_sim_'+key+'.pdf')
            else:
                plt.savefig('mean_sim_'+key+'.pdf')
    else:
        sa = np.zeros((g.K,15))
        for k in range(g.K):
            sa[k] = g.X[c == k].mean(axis=0)[:15]
        sm = np.append(np.append(np.max(sa,axis=0),np.flip(np.min(sa,axis=0))),np.max(sa[:,0]))
        vv = np.append(np.append(np.arange(1,16),np.flip(np.arange(1,16))),1)
        plt.figure()
        plt.plot(range(1,16),g.X.mean(axis=0)[:15],c='black')
        plt.xlabel('$d$')
        plt.plot(vv,sm,'--',c='red')
        for k in range(g.K):
            plt.plot(range(1,16),g.X[c == k].mean(axis=0)[:15],'x',c='gray')
        plt.legend(['Overall mean','Max./min. within-cluster mean','Within-cluster mean'])
        if dest_folder != '':
            plt.savefig(dest_folder+'/mean_sim.pdf')
        else:
            plt.savefig('mean_sim.pdf')

    ## Variances
    if gen_directed:
        for key in ['s','r']:
            plt.figure()
            plt.plot(range(1,26),np.var(g.X[key],axis=0)[:25],'--',c='black')
            plt.xlabel('$d$')
            for k in range(g.K[key] if coclust else g.K):
                plt.plot(range(1,26),np.var(g.X[key][(cc[key] if coclust else c) == k],axis=0)[:25],c='gray')
            plt.legend(['Overall variance','Within-cluster variance'])
            if dest_folder != '':
                plt.savefig(dest_folder+'/var_sim_'+key+'.pdf')
            else:
                plt.savefig('var_sim_'+key+'.pdf')
    else:
        plt.figure()
        plt.plot(range(1,26),np.var(g.X,axis=0)[:25],'--',c='black')
        plt.xlabel('$d$')
        for k in range(g.K):
            plt.plot(range(1,26),np.var(g.X[c == k],axis=0)[:25],c='gray')
        plt.legend(['Overall variance','Within-cluster variance'])
        if dest_folder != '':
            plt.savefig(dest_folder+'/var_sim.pdf')
        else:
            plt.savefig('var_sim.pdf')

    ## Correlations
    from scipy.stats import gaussian_kde as gkde

    if gen_directed:
        for key in ['s','r']:
            plt.figure()
            wc_corcoefs = np.zeros((g.K[key] if coclust else g.K,g.m,30))
            corcoefs = (np.corrcoef(g.X[key].T)-np.diag(np.ones(g.m)))[:,:30]
            for k in range(g.K[key] if coclust else g.K):
                wc_corcoefs[k] = (np.corrcoef(g.X[key][(cc[key] if coclust else c) == k].T)-np.diag(np.ones(g.m)))[:,:30]
            f, ax = plt.subplots(1,1)
            ax.set(xlabel='Correlation coefficient $\\rho_{ij}^{(k)}$')
            ax.hist(wc_corcoefs[:,2:,2:].flatten(),color='gray',bins=30,label='Histogram of $\\rho_{ij}^{(k)}$ for $X_{d:}^'+key+'$',density=True)
            ax.scatter(wc_corcoefs[:,0,1].flatten(),np.zeros(g.K[key] if coclust else g.K),color='red',label='$\\rho_{ij}^{(k)}$ for $X_{:d}^'+key+'$')
            xx = np.linspace(-.2,.2,500)
            kde = gkde(wc_corcoefs[:,2:,2:].flatten())
            ax.plot(xx, kde(xx),color='black')
            ax.legend()
            if dest_folder != '':
                plt.savefig(dest_folder+'/corr_sim_'+key+'.pdf')
            else:
                plt.savefig('corr_sim_'+key+'.pdf')
    else:
        plt.figure()
        wc_corcoefs = np.zeros((g.K,g.m,30))
        corcoefs = (np.corrcoef(g.X.T)-np.diag(np.ones(g.m)))[:,:30]
        for k in range(g.K):
            wc_corcoefs[k] = (np.corrcoef(g.X[c == k].T)-np.diag(np.ones(g.m)))[:,:30]
        f, ax = plt.subplots(1,1)
        ax.set(xlabel='Correlation coefficient $\\rho_{ij}^{(k)}$')
        ax.hist(wc_corcoefs[:,2:,2:].flatten(),color='gray',bins=30,label='Histogram of $\\rho_{ij}^{(k)}$ for $X_{d:}$',density=True)
        ax.scatter(wc_corcoefs[:,0,1].flatten(),np.zeros(g.K),color='red',label='$\\rho_{ij}^{(k)}$ for $X_{:d}$')
        xx = np.linspace(-.2,.2,500)
        kde = gkde(wc_corcoefs[:,2:,2:].flatten())
        ax.plot(xx, kde(xx),color='black')
        ax.legend()
        if dest_folder != '':
            plt.savefig(dest_folder+'/corr_sim.pdf')
        else:
            plt.savefig('corr_sim.pdf')

## MCMC sampler
if run_mcmc:
    d = []
    if g.coclust:
        K = {}
        Ko = {}
        if second_order_clustering:
            H = {}
            Ho = {}
        for key in ['s','r']:
            K[key] = []
            Ko[key] = []
            if second_order_clustering:
                H[key] = []
                Ho[key] = []
    else:
        K = []
        Ko = []
        if second_order_clustering:
            H = []
            Ho = [] 

    ## Posterior similarity matrix
    if g.coclust:
        psm = {}
        for key in ['s','r']:
            psm[key] = np.zeros((g.n[key],g.n[key])) 
    else:
        psm = np.zeros((g.n,g.n))

    ## Sampler
    for s in range(nburn+nsamp):
        ## Print status of MCMC
        if s < nburn:
            sys.stdout.write("\r+++ Burnin +++ %d / %d " % (s+1,nburn))
            sys.stdout.flush()
        elif s == nburn:
            sys.stdout.write("\n")
        elif s < nburn + nsamp - 1:
            sys.stdout.write("\r+++ Sweeps +++ %d / %d " % (s+1-nburn,nsamp))
            sys.stdout.flush()
        else:
            sys.stdout.write("\r+++ Sweeps +++ %d / %d\n " % (s+1-nburn,nsamp))
        ## Choice of the move
        if g.equal_var:
            ## - with second order clustering
            move = np.random.choice(['gibbs_comm','split_merge','change_dim','prop_empty','gibbs_comm_so','split_merge_so','prop_empty_so'])
        else:
            ## - without second order clustering
            move = np.random.choice(['gibbs_comm','split_merge','change_dim','prop_empty'])
        if move == 'gibbs_comm':
            g.gibbs_communities(l=g.n)
        elif move == 'split_merge':
            g.split_merge()
        elif move == 'change_dim':
            g.dimension_change(prop_step=3)
        elif move == 'gibbs_comm_so':
            g.gibbs_second_order()
        elif move == 'split_merge_so':
            g.split_merge_second_order()
        elif move == 'prop_empty_so':
            g.propose_empty_second_order()
        else:
            g.propose_empty()
        ## Update the parameters
        if s >= nburn:
            d += [g.d]
            if g.coclust:
                for key in ['s','r']:
                    K[key] += [g.K[key]]
                    Ko[key] += [g.Ko[key]]
                    if second_order_clustering:
                        H[key] += [g.H[key]]
                        Ho[key] += [np.sum(g.vk[key] > 0)]                
                    psm[key] += np.equal.outer(g.z[key],g.z[key])
            else:            
                K += [g.K]
                Ko += [g.Ko]
                if second_order_clustering:
                    H += [g.H]
                    Ho += [np.sum(g.vk > 0)]
                psm += np.equal.outer(g.z,g.z)

    ## Convert to arrays
    d = np.array(d)
    if coclust:
        for key in ['s','r']:
            K[key] = np.array(K[key])
            Ko[key] = np.array(Ko[key])
            if second_order_clustering:
                H[key] = np.array(H[key])
                Ho[key] = np.array(Ho[key])
    else:
        K = np.array(K)
        Ko = np.array(Ko)
        if second_order_clustering:
            H = np.array(H)
            Ho = np.array(Ho)

    ## Save files
    if dest_folder == '':
        np.savetxt('d.txt',d,fmt='%d')
        if coclust:
            for key in ['s','r']:
                np.savetxt('K_'+key+'.txt',K[key],fmt='%d')
                np.savetxt('Ko_'+key+'.txt',Ko[key],fmt='%d')
                if second_order_clustering:
                    np.savetxt('H_'+key+'.txt',H[key],fmt='%d')
                    np.savetxt('Ho_'+key+'.txt',Ho[key],fmt='%d')
                np.savetxt('psm_'+key+'.txt',psm[key]/float(np.max(psm[key])),fmt='%f')
        else:
            np.savetxt('K.txt',K,fmt='%d')
            np.savetxt('Ko.txt',Ko,fmt='%d')
            if second_order_clustering:
                np.savetxt('H.txt',H,fmt='%d')
                np.savetxt('Ho.txt',Ho,fmt='%d')
            np.savetxt('psm.txt',psm/float(np.max(psm)),fmt='%f')
    else:
        np.savetxt(dest_folder+'/d.txt',d,fmt='%d')
        if coclust:
            for key in ['s','r']:
                np.savetxt(dest_folder+'/K_'+key+'.txt',K[key],fmt='%d')
                np.savetxt(dest_folder+'/Ko_'+key+'.txt',Ko[key],fmt='%d')
                if second_order_clustering:
                    np.savetxt(dest_folder+'/H_'+key+'.txt',H[key],fmt='%d')
                    np.savetxt(dest_folder+'/Ho_'+key+'.txt',Ho[key],fmt='%d')
                np.savetxt(dest_folder+'/psm_'+key+'.txt',psm[key]/float(np.max(psm[key])),fmt='%f')  
        else:     
            np.savetxt(dest_folder+'/d.txt',d,fmt='%d')
            np.savetxt(dest_folder+'/K.txt',K,fmt='%d')
            np.savetxt(dest_folder+'/Ko.txt',Ko,fmt='%d')
            if second_order_clustering:
                np.savetxt(dest_folder+'/H.txt',H,fmt='%d')
                np.savetxt(dest_folder+'/Ho.txt',Ho,fmt='%d')
            np.savetxt(dest_folder+'/psm.txt',psm/float(np.max(psm)),fmt='%f')

    ## MAP for clustering
    if coclust:
        cc_true = {}
        cc_true['s'] = c
        cc_true['r'] = c2
        cc_pear = {}
        cc_map = {}
        for key in ['s','r']:
            cc_pear[key] = estimate_clustering(psm[key])
            cc_map[key] = estimate_clustering(psm[key],k=mode(Ko[key])[0][0])
            if dest_folder == '':
                np.savetxt('true_clusters_'+key+'.txt',cc_true[key],fmt='%d')
                np.savetxt('pear_clusters_'+key+'.txt',cc_pear[key],fmt='%d')
                np.savetxt('map_clusters_'+key+'.txt',cc_map[key],fmt='%d')
            else:
                np.savetxt(dest_folder+'/true_clusters_'+key+'.txt',cc_true[key],fmt='%d')
                np.savetxt(dest_folder+'/pear_clusters_'+key+'.txt',cc_pear[key],fmt='%d')
                np.savetxt(dest_folder+'/map_clusters_'+key+'.txt',cc_map[key],fmt='%d')
    else:
        cc_pear = estimate_clustering(psm)
        cc_map = estimate_clustering(psm,k=mode(Ko)[0][0])
        if dest_folder == '':
            np.savetxt('true_clusters.txt',c,fmt='%d')
            np.savetxt('pear_clusters.txt',cc_pear,fmt='%d')
            np.savetxt('map_clusters.txt',cc_map,fmt='%d')
        else:
            np.savetxt(dest_folder+'/true_clusters.txt',c,fmt='%d')
            np.savetxt(dest_folder+'/pear_clusters.txt',cc_pear,fmt='%d')
            np.savetxt(dest_folder+'/map_clusters.txt',cc_map,fmt='%d')
