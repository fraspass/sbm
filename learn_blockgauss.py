#!/usr/bin/env python
import sys
import argparse
import numpy as np
from numpy.linalg import eig,slogdet,inv
from numpy import pi,log,exp,sqrt
from collections import Counter, OrderedDict
from operator import itemgetter
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as rmvn
from scipy.stats import invwishart as iw
from scipy.special import gammaln

###################################
## Simulate from the prior model ##
###################################

## Number of nodes
n = 700
## Community allocation
c = np.random.choice(5,size=n)
## Size of the dataset
m = n
## Latent dimension
d = 2
K = 5
## Simulate inverse Wishart for 'garbage' part
nu0 = 2.0
kappa0 = 0.1
np.random.seed(117)
Wr = iw.rvs(df=nu0+m-d-1,scale=np.diag(0.1*np.ones(m-d)))
vr = rmvn(mean=np.zeros(m-d),cov=Wr/kappa0)
W = iw.rvs(df=nu0+d-1,scale=np.diag(0.1*np.ones(d)),size=K)
v = np.zeros((K,d))
for k in range(K):
    v[k] = rmvn(mean=np.zeros(2),cov=W[k]/kappa0)

X = np.zeros((n,m))
for i in range(n):
    X[i,:d] = rmvn(mean=v[c[i]],cov=W[c[i]])
    X[i,d:] = rmvn(mean=vr,cov=Wr)

## Plot result
plt.scatter(X[:,0],X[:,1],c=c)
plt.show()

###### Calculate marginal likelihoods for all the possible values of d
mlik = np.zeros(m)
## Posterior values for the marginalised 'garbage' Gaussian
kappan = kappa0 + n 
nun = nu0 + n
## Prior means
mean0 = np.zeros(m)
prior_sum = kappa0 * mean0
## Posterior values for the marginalised 'garbage' Gaussian
post_mean = (prior_sum + np.sum(X,axis=0)) / kappan
## Prior outer product, scaled by kappa0
prior_outer = kappa0 * np.outer(mean0,mean0)
## Prior covariance
Delta0 = np.diag(0.1*np.ones(m))

## Initialise sum and mean
sums = np.zeros((K,m))
nk = np.zeros(K)
means = np.zeros((K,m))
Deltank = {}
Deltank_det = np.zeros((K,m+1))
for k in range(K):
    x = X[c==k]
    sums[k] = x.sum(axis=0)
    nk[k] = x.shape[0]
    means[k] = (sums[k] + prior_sum) / (nk[k] + kappa0)
    Deltank[k] = Delta0 + np.dot(x.T,x) + prior_outer - (kappa0 + nk[k]) * np.outer(means[k],means[k])
    for d in range(m+1):
        Deltank_det[k,d] = slogdet(Deltank[k][:d,:d])[1]

post_Delta_tot = Delta0 + np.dot(X.T,X) + prior_outer - kappan * np.outer(post_mean,post_mean)
Deltar_det = np.zeros(m+1)
for d in range(m+1):
    Deltar_det[d] = slogdet(post_Delta_tot[d:,d:])[1] 

## Calculate the determinants sequentialy for the prior
Delta0_det = np.zeros(m+1)
Delta0_det_r = np.zeros(m+1)
for i in range(m+1):
    Delta0_det[i] = slogdet(Delta0[:i,:i])[1]
    Delta0_det_r[i] = slogdet(Delta0[i:,i:])[1]

for d in range(m):
    ## Calculate the marginal likelihood for the garbage Gaussian
    if d != m:
        mlik[d] = -.5*n*m*log(np.pi) + .5*(m-d)*(log(kappa0) - log(kappan)) + .5*(nu0+m-d-1)*(Delta0_det_r[d] - Deltar_det[d])
        mlik[d] += np.sum([gammaln(.5*(nun+m-d-i)) - gammaln(.5*(nu0+m-d-i)) for i in range(1,m-d+1)])
    else:
        mlik = 0
    ## Calculate the marginal likelihood for the community allocations
    if d != 0: 
        for k in range(K):
            mlik[d] += .5*d*(log(kappa0) - log(nk[k] + kappa0)) + .5*(nu0+d-1)*(Delta0_det[d] - Deltank_det[k,d])
            mlik[d] += np.sum([gammaln(.5*(nu0+nk[k]+d-i)) - gammaln(.5*(nu0+d-i)) for i in range(1,d+1)]) 

pos = np.array([[7,4],[3,1],[4,8],[1,.2],[9,2]])
covs = .01*np.array([[2,.5],[.5,1]])
## Create the array
X = np.zeros((n,m))
for i in range(n):
    X[i,:d] = rmvn(mean=pos[c[i]],cov=covs) 
    X[i,d:] = rmvn(mean=np.zeros(m-d),cov=.01*np.diag(np.ones(m-d)))

## Plot result
plt.scatter(X[:,0],X[:,1],c=c)
plt.show()

## Plot result
plt.scatter(X[:,2],X[:,3],c=c)
plt.show()

## Set number of clusters, latent dimension and initial cluster allocation z
K = 5
d = 2
z = np.copy(c)

## Set the hyperparameters
alpha = 1.0
## Posterior values for the marginalised 'garbage' Gaussian
kappan = kappa0 + n 
nun = nu0 + n
## Initialise means
mean0 = np.mean(X,axis=0)
prior_sum = kappa0 * mean0
## Posterior values for the marginalised 'garbage' Gaussian
post_mean_tot = (prior_sum + np.sum(X,axis=0)) / kappan
mean_r = post_mean_tot[d:]
## Prior outer product, scaled by kappa0
prior_outer = kappa0 * np.outer(mean0,mean0)
## Parameters of the geometric distributions
omega = .1
delta = .1
## Prior covariance
Delta0 = np.diag(np.diag(np.cov(X)))
## Initialise sum and mean
sum_x = np.zeros((K,d))
nk = np.zeros(K)
mean_k = np.zeros((K,d))
for k in range(K):
    sum_x[k] = X[z==k,:d].sum(axis=0)
    nk[k] = X[z == k,:d].shape[0]
    mean_k[k] = (sum_x[k] + prior_sum[:d]) / (nk[k] + kappa0)

## Calculate marginal posterior covariance for the 'garbage' Gaussian and its determinant (sequentially)
post_Delta_tot = Delta0 + np.dot(X.T,X) + prior_outer - kappan * np.outer(post_mean_tot,post_mean_tot)
Delta_r = post_Delta_tot[d:,d:]
post_Delta_tot_det = np.zeros(m+1)
for i in range(m+1):
    post_Delta_tot_det[i] = slogdet(post_Delta_tot[i:,i:])[1]
Delta_r_det = post_Delta_tot_det[d]

## Posterior values for nu and kappa
nunk = nu0 + nk
kappank = kappa0 + nk

## Calculate the sum of squares
squared_sum_x = {}
for k in range(K):
    x = X[z == k,:d]
    squared_sum_x[k] = np.dot(x.T,x)

## Initialise the Deltas 
Delta_k = {}
Delta_k_inv = {}
Delta_k_det = np.zeros(K)
for k in range(K):
    Delta_k[k] = Delta0[:d,:d] + squared_sum_x[k] + prior_outer[:d,:d] - kappank[k] * np.outer(mean_k[k],mean_k[k])
    Delta_k_inv[k] = kappank[k] / (kappank[k] + 1.0) * inv(Delta_k[k])
    Delta_k_det[k] = slogdet(Delta_k[k])[1]

## Calculate an array of outer products for the entire node set
full_outer_x = np.zeros((n,m,m))
for x in range(n):
    full_outer_x[x] = np.outer(X[x],X[x])

## Calculate the determinants sequentialy for the prior
Delta0_det = np.zeros(m+1)
Delta0_det_r = np.zeros(m+1)
for i in range(m+1):
    Delta0_det[i] = slogdet(Delta0[:i,:i])[1]
    Delta0_det_r[i] = slogdet(Delta0[i:,i:])[1]

## Run Gibbs sampling
for _ in range(100): print _; gibbs_communities(l=n)

## Plot result
plt.scatter(X[:,0],X[:,1],c=z)
plt.show()

## Change dimension
for _ in range(100): print _; dimension_change()