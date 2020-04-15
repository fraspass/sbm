#!/usr/bin/env python
import sys, os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mcmc_sampler_sbm
from estimate_cluster import estimate_clustering
from sklearn.cluster import KMeans
from scipy.stats import mode
from collections import Counter
from matplotlib2tikz import save as tikz_save

## Boolean type for parser
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

##################################
## Analysis of the ICL netflows ##
##################################

## Import the dataset
huxley = np.loadtxt('Datasets/huxley_edges_filter_map.csv',dtype=int,delimiter=',')

###### MODEL ARGUMENTS 

## PARSER to give parameter values 
parser = argparse.ArgumentParser()
# Boolean variable to use second level clustering (default True)
parser.add_argument("-s","--sord", type=str2bool, dest="second_order_clustering", default=True, const=False, nargs="?",\
    help="Boolean variable for second level clustering, default TRUE")
# Boolean variables for independent analysis (default False)
parser.add_argument("-i","--indep", type=str2bool, dest="indep_analysis", default=False, const=False, nargs="?",\
    help="Boolean variable for independent analysis on source or destination embeddings for directed graphs, default FALSE")
parser.add_argument("-k","--key", type=str, dest="indep_key", default='s', const=False, nargs="?",\
    help="String variable for independent analysis on source or destination embeddings for directed graphs, default 's' (sources).")
# Add options for .tex figures
parser.add_argument("-t","--tex", type=str2bool, dest="tex_figures", default=False, const=False, nargs="?",\
    help="Boolean variable for .tex figures, default FALSE")
# Burnin
parser.add_argument("-B","--nburn", type=int, dest="nburn", default=2500, const=True, nargs="?",\
    help="Integer: length of burnin, default 25000")
# Number of samples
parser.add_argument("-M","--nsamp", type=int, dest="nsamp", default=20000, const=True, nargs="?",\
    help="Integer: length of MCMC chain after burnin, default 500000")
## Set destination folder for output
parser.add_argument("-f","--folder", type=str, dest="dest_folder", default="Results", const=True, nargs="?",\
    help="String: name of the destination folder for the output files (*** the folder must exist ***)")

## Parse arguments
args = parser.parse_args()
coclust = True
second_order_clustering = args.second_order_clustering
indep_analysis = args.indep_analysis
tex_figures = args.tex_figures
indep_key = args.indep_key
nburn = args.nburn
nsamp = args.nsamp
dest_folder = args.dest_folder

# Create output directory if doesn't exist
if dest_folder != '' and not os.path.exists(dest_folder):
    os.mkdir(dest_folder)

## Create the adjacency matrix
n1 = np.max(huxley[:,0])+1
n2 = np.max(huxley[:,1])+1
A = np.zeros((n1,n2))
for link in huxley:
    A[link[0],link[1]] = 1.0

## Construct the Gibbs sampling object
g = mcmc_sampler_sbm.mcmc_sbm(A,m=100)

## Initialise the clusters using k-means
g.init_cocluster(zs=KMeans(n_clusters=6).fit(g.X['s'][:,:6]).labels_,zr=KMeans(n_clusters=10).fit(g.X['r'][:,:10]).labels_)

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
g.init_group_variance_coclust(vs=range(g.K['s']),vr=range(g.K['r']))

## Independent analysis using only the sources/destinations
if indep_analysis:
    g.independent_analysis(key=indep_key)

## MCMC sampler
d = []
K = {}
Ko = {}
if second_order_clustering:
    H = {}
    Ho = {}

for key in [indep_key] if indep_analysis else ['s','r']:
    K[key] = []
    Ko[key] = []
    if second_order_clustering:
        H[key] = []
        Ho[key] = []

## Posterior similarity matrix
psm = {}
if indep_analysis:
    psm[indep_key] = np.zeros((g.n,g.n))
else:
    for key in ['s','r']:
        psm[key] = np.zeros((g.n[key],g.n[key]))

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
        g.dimension_change(prop_step=5)
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
        if indep_analysis:
            K[indep_key] += [g.K]
            Ko[indep_key] += [g.Ko]
            if second_order_clustering:
                H[indep_key] += [g.H]
                Ho[indep_key] += [np.sum(g.vk > 0)]                
            psm[indep_key] += np.equal.outer(g.z,g.z)            
        else:
            for key in ['s','r']:
                K[key] += [g.K[key]]
                Ko[key] += [g.Ko[key]]
                if second_order_clustering:
                    H[key] += [g.H[key]]
                    Ho[key] += [np.sum(g.vk[key] > 0)]                
                psm[key] += np.equal.outer(g.z[key],g.z[key])

## Convert to arrays
d = np.array(d)
for key in [indep_key] if indep_analysis else ['s','r']:
    K[key] = np.array(K[key])
    Ko[key] = np.array(Ko[key])
    if second_order_clustering:
        H[key] = np.array(H[key])
        Ho[key] = np.array(Ho[key])

## Save files
if dest_folder == '':
    np.savetxt('d.txt',d,fmt='%d')
    for key in [indep_key] if indep_analysis else ['s','r']:
        np.savetxt('K_'+key+'.txt',K[key],fmt='%d')
        np.savetxt('Ko_'+key+'.txt',Ko[key],fmt='%d')
        if second_order_clustering:
            np.savetxt('H_'+key+'.txt',H[key],fmt='%d')
            np.savetxt('Ho_'+key+'.txt',Ho[key],fmt='%d')
        np.savetxt('psm_'+key+'.txt',psm[key]/float(np.max(psm[key])),fmt='%f')
else:
    np.savetxt(dest_folder+'/d.txt',d,fmt='%d')
    for key in [indep_key] if indep_analysis else ['s','r']:
        np.savetxt(dest_folder+'/K_'+key+'.txt',K[key],fmt='%d')
        np.savetxt(dest_folder+'/Ko_'+key+'.txt',Ko[key],fmt='%d')
        if second_order_clustering:
            np.savetxt(dest_folder+'/H_'+key+'.txt',H[key],fmt='%d')
            np.savetxt(dest_folder+'/Ho_'+key+'.txt',Ho[key],fmt='%d')
        np.savetxt(dest_folder+'/psm_'+key+'.txt',psm[key]/float(np.max(psm[key])),fmt='%f')  

##### Plots #####

## Scree plot
U,S,V = np.linalg.svd(A)
plt.figure()
plt.plot(np.arange(len(S))+1,S,c='black')
plt.plot(np.arange(len(S))+1,S,'.',markersize=.3,c='black')
plt.plot(mode(d)[0][0]+1,S[mode(d)[0][0]],"o",c='red')
if dest_folder == '':
    if not tex_figures:
        plt.savefig('scree_plot.pdf')
    else:
        tikz_save('scree_plot.tex')
else:
    if not tex_figures:
        plt.savefig(dest_folder+'/scree_plot.pdf')
    else:
        tikz_save(dest_folder+'/scree_plot.tex')

## Posterior barplot (unrestricted)
for key in [indep_key] if indep_analysis else ['s','r']:
    plt.figure()
    fig, ax = plt.subplots()
    ax.bar(np.array(Counter(Ko[key]).keys())-.35,Counter(Ko[key]).values(),width=0.35,color='black',align='edge',alpha=.8,label='$K_\\varnothing$')
    if second_order_clustering:
        ax.bar(np.array(Counter(Ho[key]).keys()),Counter(Ho[key]).values(),width=0.35,color='gray',align='edge',alpha=.8,label='$H_\\varnothing$')
    leg = ax.legend()
    ax.axvline(x=mode(d)[0][0],linestyle='--',c='red')
    if dest_folder == '':
        if not tex_figures:
            plt.savefig('posterior_barplot_unrestricted_'+key+'.pdf')
        else:
            tikz_save('posterior_barplot_unrestricted_'+key+'.tex')
    else:
        if not tex_figures:
            plt.savefig(dest_folder+'/posterior_barplot_unrestricted_'+key+'.pdf')
        else:
            tikz_save(dest_folder+'/posterior_barplot_unrestricted_'+key+'.tex')

## Posterior barplot (restricted)
for key in [indep_key] if indep_analysis else ['s','r']:
    plt.figure()
    fig, ax = plt.subplots()
    ax.bar(np.array(Counter((Ko[key])[Ko[key] >= d]).keys())-.35,Counter((Ko[key])[Ko[key] >= d]).values(),width=0.35,
        color='black',align='edge',alpha=.8,label='$K_\\varnothing$')
    if second_order_clustering:
        ax.bar(np.array(Counter((Ho[key])[Ko[key] >= d]).keys()),Counter((Ho[key])[Ko[key] >= d]).values(),width=0.35,
            color='gray',align='edge',alpha=.8,label='$H_\\varnothing$')
    leg = ax.legend()
    ax.axvline(x=mode(d)[0][0],linestyle='--',c='red')
    if dest_folder == '':
        if not tex_figures:
            plt.savefig('posterior_barplot_restricted_'+key+'.pdf')
        else:
            tikz_save('posterior_barplot_restricted_'+key+'.tex')
    else:
        if not tex_figures:
            plt.savefig(dest_folder+'/posterior_barplot_restricted_'+key+'.pdf')
        else:
            tikz_save(dest_folder+'/posterior_barplot_restricted_'+key+'.tex')

## MAP for clustering
cc_pear = {}
cc_map = {}
for key in [indep_key] if indep_analysis else ['s','r']:
    cc_pear[key] = estimate_clustering(psm[key])
    cc_map[key] = estimate_clustering(psm[key],k=mode(Ko[key])[0][0])
    if dest_folder == '':
        np.savetxt('pear_clusters_'+key+'.txt',cc_pear[key],fmt='%d')
        np.savetxt('map_clusters_'+key+'.txt',cc_map[key],fmt='%d')
    else:
        np.savetxt(dest_folder+'/pear_clusters_'+key+'.txt',cc_pear[key],fmt='%d')
        np.savetxt(dest_folder+'/map_clusters_'+key+'.txt',cc_map[key],fmt='%d')
