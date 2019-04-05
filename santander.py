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

#############################################
## Analysis of the Santander bikes network ##
#############################################

## Import the dataset
cycle = pd.read_csv('Datasets/126JourneyDataExtract05Sep2018-11Sep2018.csv',quotechar='"')
cycle_edges = cycle[['StartStation Id','EndStation Id']]

###### MODEL ARGUMENTS 

## PARSER to give parameter values 
parser = argparse.ArgumentParser()
# Boolean variable to use LSE (default ASE)
parser.add_argument("-l","--lap", type=str2bool, dest="use_laplacian", default=False, const=False, nargs="?",\
	help="Boolean variable for LSE, default FALSE (i.e. use ASE)")
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
    help="String: name of the destination folder for the output files (*** the folder must exist ***)")
## Parse arguments
args = parser.parse_args()
use_laplacian = args.use_laplacian
second_order_clustering = args.second_order_clustering
nburn = args.nburn
nsamp = args.nsamp
dest_folder = args.dest_folder

## Create a dictionary and the adjacency matrix
cycle_dict = {}
cycle_dict_inv = {}
i = 0 
for name in np.unique(cycle_edges):
    cycle_dict[name] = i
    cycle_dict_inv[i] = name
    i += 1

## Create the adjacency matrix
n = len(cycle_dict.keys())
A = np.zeros((n,n))
for link in np.array(cycle_edges):
    if link[0] != link[1]:
        A[cycle_dict[link[0]],cycle_dict[link[1]]] = 1.0
        A[cycle_dict[link[1]],cycle_dict[link[0]]] = 1.0

## Construct the Gibbs sampling object
if not use_laplacian:
	## - Adjacency Spectral Embedding
	g = mcmc_sampler_sbm.mcmc_sbm(A,m=25)
else:
	## - Laplacian Spectral Embedding
	g = mcmc_sampler_sbm.mcmc_sbm(np.dot(np.dot(np.diag(A.sum(axis=0)**(-.5)),A),np.diag(A.sum(axis=0)**(-.5))),m=25)

## Initialise the clusters using k-means
g.init_cluster(z=KMeans(n_clusters=5).fit(g.X[:,:5]).labels_)
## Average within-cluster variance
v = np.zeros(g.m)
for k in range(g.K):
	v += np.var(g.X[g.z == k],axis=0) / g.K

## Initialise d 
g.init_dim(d=g.K,delta=0.1,d_constrained=False)

## Initialise the parameters of the Gaussian distribution
g.prior_gauss_left_undirected(mean0=np.zeros(g.m),Delta0=np.diag(v))
g.prior_gauss_right_undirected(sigma0=np.var(g.X,axis=0))

## Initialise the second order clustering
g.init_group_variance_clust(v=range(g.K),beta=1.0) 

## Track the parameters
d = []
K = []
Ko = []
if second_order_clustering:
	H = []
	Ho = [] 

## Posterior similarity matrix
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
        K += [g.K]
        Ko += [g.Ko]
        if second_order_clustering:
            H += [g.H]
            Ho += [np.sum(g.vk > 0)]
        psm += np.equal.outer(g.z,g.z)

## Save files
if dest_folder == '':
	np.savetxt('d.txt',d,fmt='%d')
	np.savetxt('K.txt',K,fmt='%d')
	np.savetxt('Ko.txt',Ko,fmt='%d')
	if second_order_clustering:
		np.savetxt('H.txt',H,fmt='%d')
		np.savetxt('Ho.txt',Ho,fmt='%d')
	np.savetxt('psm.txt',psm/float(np.max(psm)),fmt='%f')
else:
	np.savetxt(dest_folder+'/d.txt',d,fmt='%d')
	np.savetxt(dest_folder+'/K.txt',K,fmt='%d')
	np.savetxt(dest_folder+'/Ko.txt',Ko,fmt='%d')
	if second_order_clustering:
		np.savetxt(dest_folder+'/H.txt',H,fmt='%d')
		np.savetxt(dest_folder+'/Ho.txt',Ho,fmt='%d')
	np.savetxt(dest_folder+'/psm.txt',psm/float(np.max(psm)),fmt='%f')

## Scree-plot
w,v = np.linalg.eigh(A)
S = w[::-1]
plt.figure()
plt.plot(np.arange(len(S))+1,S,c='black')
plt.plot(np.arange(len(S))+1,S,'.',markersize=.3,c='black')
plt.plot(mode(d)[0][0]+1,S[mode(d)[0][0]],"o",c='red')
if dest_folder == '':
	plt.savefig('scree_plot.pdf')
else:
	plt.savefig(dest_folder+'/scree_plot.pdf')

## Posterior barplot of K and H
plt.figure()
fig, ax = plt.subplots()
ax.bar(np.array(Counter(Ko).keys())-.35,Counter(Ko).values(),width=0.35,color='black',align='edge',alpha=.8,label='$K_\\varnothing$')
if second_order_clustering:
	ax.bar(np.array(Counter(Ho).keys()),Counter(Ho).values(),width=0.35,color='gray',align='edge',alpha=.8,label='$H_\\varnothing$')
ax.axvline(mode(d)[0][0],c='red',linestyle='--')
leg = ax.legend()
if dest_folder == '':
	plt.savefig('posterior_barplot.pdf')
else:
	plt.savefig(dest_folder+'/posterior_barplot.pdf')

## MAP for clustering
cc_pear = estimate_clustering(psm)
cc_map = estimate_clustering(psm,k=mode(Ko)[0][0])
if dest_folder == '':
	np.savetxt('pear_clusters.txt',cc_pear,fmt='%d')
	np.savetxt('map_clusters.txt',cc_map,fmt='%d')
else:
	np.savetxt(dest_folder+'/pear_clusters.txt',cc_pear,fmt='%d')
	np.savetxt(dest_folder+'/map_clusters.txt',cc_map,fmt='%d')

