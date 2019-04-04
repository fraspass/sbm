#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mcmc_sampler_sbm
from estimate_cluster import estimate_clustering
from sklearn.cluster import KMeans

#############################################
## Analysis of the Santander bikes network ##
#############################################

## Import the dataset
cycle = pd.read_csv('Datasets/126JourneyDataExtract05Sep2018-11Sep2018.csv',quotechar='"')
cycle_edges = cycle[['StartStation Id','EndStation Id']]

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
## - Adjacency Spectral Embedding
g = mcmc_sampler_sbm.mcmc_sbm(A,m=25)
## - Laplacian Spectral Embedding
## g = mcmc_sampler_sbm.mcmc_sbm(np.dot(np.dot(np.diag(A.sum(axis=0)**(-.5)),A),np.diag(A.sum(axis=0)**(-.5))),m=25)

## Initialise the clusters using k-means
g.init_cluster(z=KMeans(n_clusters=5).fit(g.X[:,:5]).labels_)
## Average within-cluster variance
v = np.zeros(m)
for k in range(g.K):
	v += np.diag(np.var(g.X[g.z == k],axis=0)) / g.K

## Initialise d 
g.init_dim(d=g.K,delta=0.1,d_constrained=False)

## Initialise the parameters of the Gaussian distribution
g.prior_gauss_left_undirected(mean0=np.zeros(m),Delta0=np.diag(v))
g.prior_gauss_right_undirected(sigma0=np.var(g.X,axis=0))

## Initialise the second order clustering
g.init_group_variance_clust(v=range(g.K),beta=1.0) 

## MCMC sampler
nburn = 25000
nsamp = 500000

## Track the parameters
d = []
K = []
H = []
Ko = []
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
    ## - with second order clustering
    move = np.random.choice(['gibbs_comm','split_merge','change_dim','prop_empty','gibbs_comm_so','split_merge_so','prop_empty_so'])
    ## - without second order clustering
    # move = np.random.choice(['gibbs_comm','split_merge','change_dim','prop_empty'])
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
        H += [g.H]
        Ko += [g.Ko]
        Ho += [np.sum(g.vk > 0)]
        psm += np.equal.outer(g.z,g.z)

## Estimate clustering
cc = estimate_clustering(psm)

## Posterior barplot of K and H
from scipy.stats import mode
from collections import Counter
fig, ax = plt.subplots()
ax.bar(np.array(Counter(Ko).keys())-.35,Counter(Ko).values(),width=0.35,color='black',align='edge',alpha=.8,label='$K_\\varnothing$')
ax.bar(np.array(Counter(Ho).keys()),Counter(Ho).values(),width=0.35,color='gray',align='edge',alpha=.8,label='$H_\\varnothing$')
ax.axvline(mode(d)[0][0],c='red',linestyle='--')
leg = ax.legend()
plt.show()

## Scree-plot
w,v = np.linalg.eigh(A)
S = w[::-1]
plt.plot(np.arange(len(S))+1,S,c='black')
plt.plot(np.arange(len(S))+1,S,'.',markersize=.3,c='black')
plt.plot(stats.mode(d)[0][0]+1,S[stats.mode(d)[0][0]],"o",c='red')
plt.show()
