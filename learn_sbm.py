#!/usr/bin/env python
import sys
import argparse
import numpy as np
from numpy.linalg import eig,slogdet,inv
from numpy import pi,log,exp,sqrt
from collections import Counter, OrderedDict
from operator import itemgetter
import matplotlib.pyplot as plt

## Set the number of nodes
n = 500
## Randomly allocate to clusters
c = np.random.choice(5,size=n)
## Vector of latent positions
m = np.array([[.7,.4],[.3,.1],[.4,.8],[.1,.2],[.9,.2]])
## Construct the adajcency matrix
A = np.zeros((n,n))
for i in range(n-1):
  for j in range(i+1,n):
    A[i,j] = np.random.binomial(1,np.sum(m[c[i]]*m[c[j]]))
    A[j,i] = A[i,j]

## Spectral decomposition --> n-dimensional embeddings
w,v = eig(A)
w_mag = (-w).argsort()
m = 10 ##int(np.floor(np.sqrt(n)))
X = np.dot(v[:,w_mag[:m]],np.diag(abs(w[w_mag[:m]]))**.5)

## Set number of clusters, latent dimension and initial cluster allocation z
K = 5
d = 5 #2
z = np.copy(c)

## Set the hyperparameters
alpha = 1.0
nu0 = 1.0
kappa0 = 1.0
## Posterior values for the marginalised 'garbage' Gaussian
kappan = kappa0 + n 
nun = nu0 + n
## Initialise means
mean0 = np.zeros(m) ##np.mean(X,axis=0)
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
Delta0 = np.diag(np.ones(m)) ## np.diag(np.diag(np.cov(X.T)))
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

## Calculate the determinants sequentially for the prior
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
dd = np.zeros(100)
for _ in range(100): print _; dimension_change(); dd[_] = d







#!/usr/bin/env python
import sys
import argparse
import numpy as np
from numpy.linalg import eig,slogdet,inv
from numpy import pi,log,exp,sqrt
from collections import Counter, OrderedDict
from operator import itemgetter
import matplotlib.pyplot as plt
import mcmc_sampler_undirected as mcmc_sampler_class

## Set the number of nodes
n = 500
## Randomly allocate to clusters
c = np.random.choice(5,size=n)
## Vector of latent positions
m = np.array([[.7,.4],[.3,.1],[.4,.8],[.1,.2],[.9,.2]])
## Construct the adajcency matrix
A = np.zeros((n,n))
for i in range(n-1):
  for j in range(i+1,n):
    A[i,j] = np.random.binomial(1,np.sum(m[c[i]]*m[c[j]]))
    A[j,i] = A[i,j]

## Spectral decomposition --> n-dimensional embeddings
w,v = eig(A)
w_mag = (-abs(w)).argsort()
m = 100 ##int(np.floor(np.sqrt(n)))
X = 10*np.dot(v[:,w_mag[:m]],np.diag(abs(w[w_mag[:m]]))**.5)

g = mcmc_sampler_class.gibbs_undirected(X)
g.init_dim(d=10,delta=0.1)
g.init_cluster(z=c)##np.random.choice([0,1],size=g.n),K=2,alpha=1.0,omega=0.1)
### g.init_cluster(z=c,K=6,alpha=1.0,omega=0.1)
g.prior_gauss(mean0=np.zeros(g.m),Delta0=np.diag(np.diag(np.cov(X.T)))/g.K,kappa0=1e-14,nu0=1.0)## ,covstrut='diagonal',meanstrut='known')
## g.prior_gauss(mean0=np.zeros(g.m),Delta0=.01*np.diag(np.ones(g.m)),kappa0=1e-8,nu0=.0)##,covstrut='diagonal',meanstrut='known')
g.marginal_likelihoods_dimension()
for _ in range(10): g.gibbs_communities()




plt.plot(g.mlik)
plt.show()



















