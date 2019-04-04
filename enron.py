#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import mcmc_sampler_sbm
from sklearn.cluster import KMeans

#########################################
## Analysis of the Enron Email Network ##
#########################################

## Import the dataset
enron = np.loadtxt('Datasets/enron_edges.txt',dtype=int,delimiter='\t')-1

## Create the adjacency matrix
n = enron_attr.shape[0]
A = np.zeros((n,n))
for link in enron:
    A[link[0],link[1]] = 1.0

## Construct the Gibbs sampling object
g = mcmc_sampler_sbm.mcmc_sbm(A,m=25)

## Initialise the clusters using k-means
g.init_cluster(z=KMeans(n_clusters=5).fit(g.X['s'][:,:5]).labels_ if g.directed else KMeans(n_clusters=5).fit(g.X[:,:5]).labels_)
#g.init_cocluster(zs=KMeans(n_clusters=5).fit(g.X['s'][:,:g.d]).labels_,zr=KMeans(n_clusters=5).fit(g.X['r'][:,:g.d]).labels_)

## Average within-cluster variance
v = {} 
v['s'] = np.zeros(m)
v['r'] = np.zeros(m)
for key in ['s','r']:
    for k in range(g.K[key] if g.coclust else g.K):
        v[key] += np.diag(np.var(g.X[key][(g.z[key] if g.coclust else g.z) == k],axis=0)) / (g.K[key] if g.coclust else g.K)

## Initialise d 
g.init_dim(d=g.K[key] if g.coclust else g.K,delta=0.1,d_constrained=False)

## Initialise the parameters of the Gaussian distribution
g.prior_gauss_left_directed(mean0s=np.zeros(m),mean0r=np.zeros(m),Delta0s=np.diag(v['s']),Delta0r=np.diag(v['r']))
g.prior_gauss_right_directed(sigma0s=np.var(g.X['s'],axis=0),sigma0r=np.var(g.X['r'],axis=0))

## Initialise the second order clustering
if g.coclust:
    g.init_group_variance_coclust(vs=range(g.K['s']),vr=range(g.K['r']),beta=1.0) 
else:
    g.init_group_variance_clust(v=range(g.K),beta=1.0)

## MCMC sampler
nburn = 25000
nsamp = 500000
d = []
if g.coclust:
    for key in ['s','r']:
        K[key] = []
        H[key] = []
        Ko[key] = []
        Ho[key] = []
else:
    K = []
    H = [] 
    Ko = []
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
        if g.coclust:
            for key in ['s','r']:
                K[key] = [g.K[key]]
                H[key] = [g.H[key]]
                Ko[key] = [g.Ko[key]]
                Ho[key] = [np.sum(g.vk[key] > 0)]                
                psm[key] += np.equal.outer(g.z[key],g.z[key])
        else:            
            K += [g.K]
            H += [g.H]
            Ko += [g.Ko]
            Ho += [np.sum(g.vk > 0)]
            psm += np.equal.outer(g.z,g.z)


d = np.array(d)
K = np.array(K)
Ko = np.array(Ko)
H = np.array(H)
Ho = np.array(Ho)

### Plots
from scipy.stats import mode
from collections import Counter
fig, ax = plt.subplots()
ax.bar(np.array(Counter(Ko).keys())-.35,Counter(Ko).values(),width=0.35,color='black',align='edge',alpha=.8,label='$K_\\varnothing$')
ax.bar(np.array(Counter(Ho).keys()),Counter(Ho).values(),width=0.35,color='gray',align='edge',alpha=.8,label='$H_\\varnothing$')
leg = ax.legend()
plt.axvline(x=mode(d)[0][0],linestyle='--',c='red')
plt.show()

fig, ax = plt.subplots()
ax.bar(np.array(Counter(Ko[Ko >= d]).keys())-.35,Counter(Ko[Ko >= d]).values(),width=0.35,color='black',align='edge',alpha=.8,label='$K_\\varnothing$')
ax.bar(np.array(Counter(Ho[Ko >= d]).keys()),Counter(Ho[Ko >= d]).values(),width=0.35,color='gray',align='edge',alpha=.8,label='$H_\\varnothing$')
leg = ax.legend()
plt.axvline(x=mode(d)[0][0],linestyle='--',c='red')
plt.show()

fig, ax = plt.subplots()
ax.bar(np.array(Counter(Ko).keys())-.175,Counter(Ko).values(),width=0.35,color='black',align='edge',alpha=.8,label='$K_\\varnothing$')
leg = ax.legend()
plt.axvline(x=mode(d)[0][0],linestyle='--',c='red')
plt.show()

fig, ax = plt.subplots()
ax.bar(np.array(Counter(Ko[Ko >= d]).keys())-.175,Counter(Ko[Ko >= d]).values(),width=0.35,color='black',align='edge',alpha=.8,label='$K_\\varnothing$')
leg = ax.legend()
plt.axvline(x=mode(d)[0][0],linestyle='--',c='red')
plt.show()

U,S,V = np.linalg.svd(A)
plt.plot(np.arange(len(S))+1,S,c='black')
plt.plot(np.arange(len(S))+1,S,'.',markersize=.3,c='black')
plt.plot(mode(d)[0][0]+1,S[mode(d)[0][0]],"o",c='red')
plt.show()
