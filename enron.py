#!/usr/bin/env python
import sys
import argparse
import numpy as np
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

#########################################
## Analysis of the Enron Email Network ##
#########################################

## Import the dataset
enron = np.loadtxt('Datasets/enron_edges.txt',dtype=int,delimiter='\t')-1

###### MODEL ARGUMENTS 

## PARSER to give parameter values 
parser = argparse.ArgumentParser()
# Boolean variable to use LSE (default ASE)
parser.add_argument("-c","--coclust", type=str2bool, dest="coclust", default=False, const=False, nargs="?",\
    help="Boolean variable for coclustering, default FALSE")
# Boolean variable to use second level clustering (default True)
parser.add_argument("-s","--sord", type=str2bool, dest="second_order_clustering", default=True, const=False, nargs="?",\
    help="Boolean variable for second level clustering, default TRUE")
# Burnin
parser.add_argument("-B","--nburn", type=int, dest="nburn", default=25000, const=True, nargs="?",\
    help="Integer: length of burnin, default 25000")
# Consider Aij as a truncated latent Poisson count Nij 
parser.add_argument("-M","--nsamp", type=int, dest="nsamp", default=500000, const=True, nargs="?",\
    help="Integer: length of MCMC chain after burnin, default 500000")
## Set destination folder for output
parser.add_argument("-f","--folder", type=str, dest="dest_folder", default="", const=True, nargs="?",\
    help="String: name of the destination folder for the output files (*** the folder must exist ***)")
## Parse arguments
args = parser.parse_args()
coclust = args.coclust
second_order_clustering = args.second_order_clustering
nburn = args.nburn
nsamp = args.nsamp
dest_folder = args.dest_folder

## Create the adjacency matrix
n = np.max(enron)+1
A = np.zeros((n,n))
for link in enron:
    A[link[0],link[1]] = 1.0

## Construct the Gibbs sampling object
g = mcmc_sampler_sbm.mcmc_sbm(A,m=25)

## Initialise the clusters using k-means
if not coclust:
    g.init_cluster(z=KMeans(n_clusters=5).fit(g.X['s'][:,:5]).labels_ if g.directed else KMeans(n_clusters=5).fit(g.X[:,:5]).labels_)
else:
    g.init_cocluster(zs=KMeans(n_clusters=5).fit(g.X['s'][:,:5]).labels_,zr=KMeans(n_clusters=5).fit(g.X['r'][:,:5]).labels_)

## Average within-cluster variance
v = {} 
v['s'] = np.zeros(g.m)
v['r'] = np.zeros(g.m)
for key in ['s','r']:
    for k in range(g.K[key] if g.coclust else g.K):
        v[key] += np.var(g.X[key][(g.z[key] if g.coclust else g.z) == k],axis=0) / (g.K[key] if g.coclust else g.K)

## Initialise d 
g.init_dim(d=g.K[key] if g.coclust else g.K,delta=0.1,d_constrained=False)

## Initialise the parameters of the Gaussian distribution
g.prior_gauss_left_directed(mean0s=np.zeros(g.m),mean0r=np.zeros(g.m),Delta0s=np.diag(v['s']),Delta0r=np.diag(v['r']))
g.prior_gauss_right_directed(sigma0s=np.var(g.X['s'],axis=0),sigma0r=np.var(g.X['r'],axis=0))

## Initialise the second level clustering
if second_order_clustering:
    if coclust:
        g.init_group_variance_coclust(vs=range(g.K['s']),vr=range(g.K['r']),betas=1.0,betar=1.0) 
    else:
        g.init_group_variance_clust(v=range(g.K),beta=1.0)

## MCMC sampler
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
        psm[key] = np.zeros((n,n)) 
else:
    psm = np.zeros((n,n))

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
    if second_order_clustering:
        ## - with second order clustering
        move = np.random.choice(['gibbs_comm','split_merge','change_dim','prop_empty','gibbs_comm_so','split_merge_so','prop_empty_so'])
    else:
        ## - without second order clustering
        move = np.random.choice(['gibbs_comm','split_merge','change_dim','prop_empty'])
    if move == 'gibbs_comm':
        g.gibbs_communities()
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

##### Plots #####

## Scree plot
U,S,V = np.linalg.svd(A)
plt.plot(np.arange(len(S))+1,S,c='black')
plt.plot(np.arange(len(S))+1,S,'.',markersize=.3,c='black')
plt.plot(mode(d)[0][0]+1,S[mode(d)[0][0]],"o",c='red')
if dest_folder == '':
    plt.savefig('scree_plot.pdf')
else:
    plt.savefig(dest_folder+'/scree_plot.pdf')

## Posterior barplot (unrestricted)
if coclust:
    for key in ['s','r']:
        fig, ax = plt.subplots()
        ax.bar(np.array(Counter(Ko[key]).keys())-.35,Counter(Ko[key]).values(),width=0.35,color='black',align='edge',alpha=.8,label='$K_\\varnothing$')
        if second_order_clustering:
            ax.bar(np.array(Counter(Ho[key]).keys()),Counter(Ho[key]).values(),width=0.35,color='gray',align='edge',alpha=.8,label='$H_\\varnothing$')
        leg = ax.legend()
        plt.axvline(x=mode(d)[0][0],linestyle='--',c='red')
        if dest_folder == '':
            plt.savefig('posterior_barplot_unrestricted_'+key+'.pdf')
        else:
            plt.savefig(dest_folder+'/posterior_barplot_unrestricted_'+key+'.pdf')
else:
    fig, ax = plt.subplots()
    ax.bar(np.array(Counter(Ko).keys())-.35,Counter(Ko).values(),width=0.35,color='black',align='edge',alpha=.8,label='$K_\\varnothing$')
    if second_order_clustering:
        ax.bar(np.array(Counter(Ho).keys()),Counter(Ho).values(),width=0.35,color='gray',align='edge',alpha=.8,label='$H_\\varnothing$')
    leg = ax.legend()
    plt.axvline(x=mode(d)[0][0],linestyle='--',c='red')
    if dest_folder == '':
        plt.savefig('posterior_barplot_unrestricted.pdf')
    else:
        plt.savefig(dest_folder+'/posterior_barplot_unrestricted.pdf')

## Posterior barplot (restricted)
if coclust:
    for key in ['s','r']:
        fig, ax = plt.subplots()
        ax.bar(np.array(Counter((Ko[key])[Ko[key] >= d]).keys())-.35,Counter((Ko[key])[Ko[key] >= d]).values(),width=0.35,
            color='black',align='edge',alpha=.8,label='$K_\\varnothing$')
        if second_order_clustering:
            ax.bar(np.array(Counter((Ho[key])[Ko[key] >= d]).keys()),Counter((Ho[key])[Ko[key] >= d]).values(),width=0.35,
                color='gray',align='edge',alpha=.8,label='$H_\\varnothing$')
        leg = ax.legend()
        plt.axvline(x=mode(d)[0][0],linestyle='--',c='red')
        if dest_folder == '':
            plt.savefig('posterior_barplot_restricted_'+key+'.pdf')
        else:
            plt.savefig(dest_folder+'/posterior_barplot_restricted_'+key+'.pdf')
else:
    fig, ax = plt.subplots()
    ax.bar(np.array(Counter(Ko[Ko >= d]).keys())-.35,Counter(Ko[Ko >= d]).values(),width=0.35,color='black',align='edge',alpha=.8,label='$K_\\varnothing$')
    if second_order_clustering:
        ax.bar(np.array(Counter(Ho[Ko >= d]).keys()),Counter(Ho[Ko >= d]).values(),width=0.35,color='gray',align='edge',alpha=.8,label='$H_\\varnothing$')
    leg = ax.legend()
    plt.axvline(x=mode(d)[0][0],linestyle='--',c='red')
    if dest_folder == '':
        plt.savefig('posterior_barplot_restricted.pdf')
    else:
        plt.savefig(dest_folder+'/posterior_barplot_restricted.pdf')

## MAP for clustering
if coclust:
    cc_pear = {}
    cc_map = {}
    for key in ['s','r']:
        cc_pear[key] = estimate_clustering(psm[key])
        cc_map[key] = estimate_clustering(psm[key],k=mode(Ko[key])[0][0])
        if dest_folder == '':
            np.savetxt('pear_clusters_'+key+'.txt',cc_pear[key],fmt='%d')
            np.savetxt('map_clusters_'+key+'.txt',cc_map[key],fmt='%d')
        else:
            np.savetxt(dest_folder+'/pear_clusters_'+key+'.txt',cc_pear[key],fmt='%d')
            np.savetxt(dest_folder+'/map_clusters_'+key+'.txt',cc_map[key],fmt='%d')
else:
    cc_pear = estimate_clustering(psm)
    cc_map = estimate_clustering(psm,k=mode(Ko)[0][0])
    if dest_folder == '':
        np.savetxt('pear_clusters.txt',cc_pear,fmt='%d')
        np.savetxt('map_clusters.txt',cc_map,fmt='%d')
    else:
        np.savetxt(dest_folder+'/pear_clusters.txt',cc_pear,fmt='%d')
        np.savetxt(dest_folder+'/map_clusters.txt',cc_map,fmt='%d')

