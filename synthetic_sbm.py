#!/usr/bin/env python
import sys
import numpy as np
from numpy.linalg import svd,eigh
import matplotlib.pyplot as plt
import mcmc_sampler_sbm
from sklearn.cluster import KMeans

## Set seed
np.random.seed(14441)

#######################################
## Simulate a stochastic block model ##
#######################################

## Set the number of nodes
n = 250
n2 = 300

## Randomly allocate to clusters
K = 5
K2 = 3
c = np.random.choice(K,size=n)
c2 = np.random.choice(K2,size=n2)

## Generate B from beta draws
sym = False
coclust = True
B = np.random.beta(1.2,1.2,size=(K,K2 if coclust else K))

## If the graph is undirected, generate a symmetric matrix
if sym: 
    B = np.tril(B) + np.tril(B,k=-1).T

## Compute low rank approximation
d = 2
if sym: 
    w,v = eigh(B)
    w_mag = (-np.abs(w)).argsort()
    B = np.dot(np.dot(v[:,w_mag[:d]],np.diag(w[w_mag[:d]])),v[:,w_mag[:d]].T)
else:
    u,s,v = svd(B)
    B = np.dot(np.dot(u[:,:d],np.diag(s[:d])),v[:d])

## Check that B is a proper matrix
if (B < 0).any() or (B > 1).any():
    raise ValueError('The low-rank approximation of B is NOT a matrix of probabilities.')

## Construct the adjacency matrix
if sym:
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
g = mcmc_sampler_sbm.mcmc_sbm(A,m=50)

## Initialise clustering
if coclust:
    g.init_cocluster(zs=KMeans(n_clusters=5).fit(g.X['s'][:,:5]).labels_,zr=KMeans(n_clusters=5).fit(g.X['r'][:,:5]).labels_)
else:
    g.init_cluster(z=KMeans(n_clusters=5).fit(g.X['s'][:,:5]).labels_ if g.directed else KMeans(n_clusters=5).fit(g.X[:,:5]).labels_)

## Initialise d 
g.init_dim(d=10,delta=0.1)

## Initialise parameters
if sym:
    g.prior_gauss_left_undirected(mean0=np.mean(g.X,axis=0),Delta0=0.1*np.diag(np.var(g.X,axis=0)),kappa0=1.0,nu0=1.0)
    g.prior_gauss_right_undirected(sigma0=np.var(g.X,axis=0),lambda0=1.0)
else:
    g.prior_gauss_left_directed(mean0s=np.mean(g.X['s'],axis=0),mean0r=np.mean(g.X['r'],axis=0),
        Delta0s=0.1*np.diag(np.var(g.X['s'],axis=0)),Delta0r=0.1*np.diag(np.var(g.X['r'],axis=0)))
    g.prior_gauss_right_directed(sigma0s=np.var(g.X['s'],axis=0),sigma0r=np.var(g.X['r'],axis=0))

## Initialise variances
group_var = True
if group_var:
    if coclust:
        g.init_group_variance_coclust(vs=np.random.choice([0,1],size=g.K['s']),vr=np.random.choice([0,1],size=g.K['r']))
    else:
        g.init_group_variance_clust(v=np.random.choice([0,1],size=g.K)) 

## MCMC sampler
nburn = 25000
nsamp = 500000
d = []
if g.coclust:
    K = {}
    H = {}
    for key in ['s','r']:
        K[key] = []
        H[key] = []
else:
    K = []
    H = [] 

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
        g.gibbs_communities(l=500)
    elif move == 'split_merge':
        g.split_merge(verbose=True)
    elif move == 'change_dim':
        g.dimension_change(verbose=True,prop_step=3)
    elif move == 'gibbs_comm_so':
        g.gibbs_second_order()
    elif move == 'split_merge_so':
        g.split_merge_second_order(verbose=True)
    elif move == 'prop_empty_so':
        g.propose_empty_second_order(verbose=True)
    else:
        g.propose_empty(verbose=True)
    ## Update the parameters
    if s >= nburn:
        d += [g.d]
        if g.coclust:
            for key in ['s','r']:
                K[key] += [np.sum(g.nk[key] != 0)]
                psm[key] += np.equal.outer(g.z[key],g.z[key])
                if g.equal_var:
                    H[key] += [np.sum(g.vk[key] != 0)]          
        else:            
            K += [g.K]
            psm += np.equal.outer(g.z,g.z)
            if g.equal_var:
                H += [g.H]
