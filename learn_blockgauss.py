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
import mcmc_sampler_undirected as mcmc_sampler_class

###################################
## Simulate from the prior model ##
###################################

## Number of nodes
n = 500
## Community allocation
c = np.random.choice(5,size=n)
## Size of the dataset
m = 25
## Latent dimension
d = 2
K = 5
## Simulate inverse Wishart for 'garbage' part
nu0 = 2.0
kappa0 = 0.01
np.random.seed(1700)
Wr = iw.rvs(df=nu0+m-d-1,scale=.01*np.diag(np.ones(m-d)))
vr = rmvn(mean=np.zeros(m-d),cov=Wr/kappa0)
W = iw.rvs(df=nu0+d-1,scale=.01*np.diag(np.ones(d)),size=K)
v = np.zeros((K,d))
for k in range(K):
    v[k] = rmvn(mean=np.zeros(d),cov=W[k]/kappa0)

X = np.zeros((n,m))
for i in range(n):
    X[i,:d] = rmvn(mean=v[c[i]],cov=W[c[i]])
    X[i,d:] = rmvn(mean=vr,cov=Wr) ## rmvn(mean=np.zeros(m-d),cov=.01*np.diag(np.ones(m-d)))

g = mcmc_sampler_class.gibbs_undirected(X)
g.init_dim(d=10,delta=0.1)
g.init_cluster(z=np.random.choice([0,1],size=g.n),K=2,alpha=1.0,omega=0.1)
### g.init_cluster(z=c,K=6,alpha=1.0,omega=0.1)
g.prior_gauss(mean0=np.zeros(g.m),Delta0=.01*np.diag(np.ones(g.m)),kappa0=0.1,nu0=2.0)##,covstrut='diagonal',meanstrut='known')
g.marginal_likelihoods_dimension()
plt.plot(g.mlik)

for _ in range(100): g.gibbs_communities(l=g.n)
for _ in range(100): g.dimension_change()
g.split_merge()

g.propose_empty()



XX = np.copy(X)

m = 500
X = XX[:,:m] # X[:,:m]

## Plot result
##plt.scatter(X[:,0],X[:,1],c=c)
##plt.show()

c = np.copy(z)

###### Calculate marginal likelihoods for all the possible values of d
mlik = np.zeros(m+1)
K = 5
nu0 = 2.0
kappa0 = .1 #### 1.0
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
Delta0 = .01*np.diag(np.ones(m)) ### np.diag(np.diag(np.cov(X.T)))
 ##np.cov(X.T) ## np.diag(.1*np.ones(m))

def mlikf(k):
    global nk, Deltank, d, Delta0_det, kappa0, nu0
    l = -.5*d*nk[k]*log(np.pi) + .5*d*(log(kappa0) - log(kappa0 + nk[k])) 
    l += .5*(nu0+d-1)*slogdet(Delta0)[1]
    l -= .5*(nu0+nk[k]+d-1)*Deltank_det[k,d]
    l += np.sum([gammaln(.5*(nu0+nk[k]+d-i)) for i in range(1,d+1)])
    l -= np.sum([gammaln(.5*(nu0+d-i)) for i in range(1,d+1)])
    return l

def lik_full():
    global n, post_Delta_tot, d, Delta0_det, kappa0, nu0
    l = -.5*d*n*log(np.pi) + .5*d*(log(kappa0) - log(kappa0 + n)) 
    l += .5*(nu0+d-1)*slogdet(Delta0)[1]
    l -= .5*(nu0+n+d-1)*slogdet(post_Delta_tot)[1]
    l += np.sum([gammaln(.5*(nu0+n+d-i)) for i in range(1,d+1)])
    l -= np.sum([gammaln(.5*(nu0+d-i)) for i in range(1,d+1)])
    return l

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

mlik = np.zeros(m+1)
for d in range(m+1):
    ## Calculate the marginal likelihood for the garbage Gaussian
    if d != m:
        mlik[d] = -.5*n*(m-d)*log(np.pi) + .5*(m-d)*(log(kappa0) - log(kappan)) + .5*(nu0+m-d-1)*Delta0_det_r[d] - .5*(nu0+n+m-d-1)*Deltar_det[d]
        mlik[d] += np.sum([(gammaln(.5*(nun+m-d-i)) - gammaln(.5*(nu0+m-d-i))) for i in range(1,m-d+1)])
    else:
        mlik[d] = 0
    ## Calculate the marginal likelihood for the community allocations
    if d != 0: 
        for k in range(K):
            mlik[d] += -.5*nk[k]*d*log(np.pi) + .5*d*(log(kappa0) - log(nk[k] + kappa0)) + .5*(nu0+d-1)*Delta0_det[d] - .5*(nu0+nk[k]+d-1)*Deltank_det[k,d]
            mlik[d] += np.sum([(gammaln(.5*(nu0+nk[k]+d-i)) - gammaln(.5*(nu0+d-i))) for i in range(1,d+1)]) 

d = 2
lll = np.sum(-.5*n*d*log(np.pi) + .5*d*(log(kappa0) - log(kappa0 + nk)) + .5*((nu0+d-1)*Delta0_det[d] - (nu0+nk+d-1)*Deltank_det[:,d]))
lll += np.sum([np.sum(gammaln(.5*(nu0 + nk + d - i)) - gammaln(.5*(nu0 + d - i))) for i in range(1,d+1)]) 

lll2 = -.5*n*d*log(np.pi) + .5*d*(log(kappa0) - log(kappa0 + n)) + .5*((nu0+d-1)*Delta0_det[d] - (nu0+n+d-1)*Deltar_det[0])
lll2 += np.sum([gammaln(.5*(nu0 + n + d - i)) - gammaln(.5*(nu0 + d - i)) for i in range(1,d+1)])


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
##kappan = kappa0 + n 
##nun = nu0 + n
## Initialise means
##mean0 = np.mean(X,axis=0)
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
## Delta0 = np.diag(np.diag(np.cov(X)))
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







i,j = np.random.choice(n,size=2,replace=False)
#i = 201
#j = 158
print str(i)
print str(j)
print str(z[i] == z[j])
## Propose a split or merge move according to the sampled values
if z[i] == z[j]:
    split = True
    zsplit = z[i]
    ## Obtain the indices that must be re-allocated
    S = np.delete(range(n),[i,j])[np.delete(z == zsplit,[i,j])]
    z_prop = np.copy(z)
    ## Move j to a new cluster
    z_prop[j] = K
else:
    split = False
    ## Choose to merge to the cluster with minimum index (zmerge) and remove the cluster with maximum index (zlost)
    zmerge = min([z[i],z[j]])
    zj = z[j]
    zlost = max([z[i],z[j]])
    z_prop = np.copy(z)
    z_prop[z == zlost] = zmerge
    if zlost != K-1:
        for k in range(zlost,K-1):
            z_prop[z == k+1] = k
    ## Set of observations for the imaginary split (Dahl, 2003)
    S = np.delete(range(n),[i,j])[np.delete((z == zmerge) + (z == zlost),[i,j])]

## Initialise the proposal ratio
prop_ratio = 0
## Construct vectors for sequential posterior predictives
nk_rest = np.ones(2,int)
nunk_rest = nu0 + np.ones(2,int)
kappa_rest = kappa0 + np.ones(2,int)
sum_rest = X[[i,j],:d]
mean_rest = (prior_sum[:d] + sum_rest) / (kappa0 + 1.0)
squared_sum_restricted = full_outer_x[[i,j],:d,:d]
Delta_restricted = {}; Delta_restricted_inv = {}; Delta_restricted_det = np.zeros(2)
for q in [0,1]:
    Delta_restricted[q] = Delta0[:d,:d] + squared_sum_restricted[q] + prior_outer[:d,:d] - kappa_rest[q] * np.outer(mean_rest[q],mean_rest[q])
    Delta_restricted_inv[q] = kappa_rest[q] / (kappa_rest[q] + 1) * inv(Delta_restricted[q])
    Delta_restricted_det[q] = slogdet(Delta_restricted[q])[1]

## Randomly permute the indices in S and calculate the sequential allocations
for h in np.random.permutation(S):
    ## Calculate the predictive probability
    position = X[h,:d]
    out_position = np.outer(position,position)
    pred_prob = exp(np.array([dmvt_efficient(x=position,mu=mean_rest[q],Sigma_inv=Delta_restricted_inv[q], \
        Sigma_logdet=d*log((kappa_rest[q] + 1.0) / (kappa_rest[q] * nunk_rest[q]))+Delta_restricted_det[q], nu=nunk_rest[q]) for q in [0,1]]) + \
        log(nunk_rest[q] + alpha/2.0))
    pred_prob /= sum(pred_prob)
    print pred_prob
    if np.isnan(pred_prob).any():
        break
    if split:
        ## Sample the new value
        znew = np.random.choice(2,p=pred_prob)
        ## Update proposal ratio
        prop_ratio += log(pred_prob[znew])
        ## Update proposed z
        z_prop[h] = [zsplit,K][znew]
    else:
        ## Determine the new value deterministically
        znew = int(z[h] == zj)
        ## Update proposal ratio in the imaginary split
        prop_ratio += log(pred_prob[znew])
    
    ## Update parameters 
    nk_rest[znew] += 1
    nunk_rest[znew] += 1
    kappa_rest[znew] += 1
    sum_rest[znew] += position
    mean_rest[znew] = (prior_sum[:d] + sum_rest[znew]) / kappa_rest[znew]
    squared_sum_restricted[znew] += out_position
    Delta_restricted[znew] = Delta0[:d,:d] + squared_sum_restricted[znew] + prior_outer[:d,:d] - kappa_rest[znew] * np.outer(mean_rest[znew],mean_rest[znew])
    Delta_restricted_inv[znew] = kappa_rest[znew] / (kappa_rest[znew] + 1.0) * inv(Delta_restricted[znew])
    sign, Delta_restricted_det[znew] = slogdet(Delta_restricted[znew])
    if sign < 0:
        break

## Calculate the acceptance probability
if split:
    ## Calculate the acceptance ratio
    accept_ratio = .5*d*(log(kappa0) + log(kappank[zsplit]) - np.sum(log(kappa_rest))) 
    accept_ratio += .5*(nu0+d-1)*Delta0_det[d] + .5*(nunk[zsplit]+d-1)*Delta_k_det[zsplit] - .5*np.sum((nunk_rest+d-1)*Delta_restricted_det)
    accept_ratio += np.sum(gammaln(.5*(np.subtract.outer(nunk_rest+d,np.arange(d)+1))))
    accept_ratio -= np.sum(gammaln(.5*(nu0+d-np.arange(d)+1))) + np.sum(gammaln(.5*(nunk[zsplit]+d-np.arange(d)+1))) 
    accept_ratio += K*gammaln(float(alpha)/K) - (K+1)*gammaln(alpha/(K+1)) - np.sum(gammaln(nk + float(alpha)/K))
    accept_ratio += np.sum(gammaln(np.delete(nk,zsplit) + alpha/(K+1.0))) + np.sum(gammaln(nk_rest + alpha/(K+1.0)))
    accept_ratio += log(1.0-omega) - prop_ratio
else:
    ## Merge the two clusters and calculate the acceptance ratio
    nk_sum = np.sum(nk[[zmerge,zlost]])
    nunk_sum = nu0 + nk_sum
    kappank_sum = kappa0 + nk_sum
    sum_x_sum = sum_x[zmerge] + sum_x[zlost]
    mean_k_sum = (prior_sum[:d] + sum_x_sum) / kappank_sum
    squared_sum_x_sum = squared_sum_x[zlost] + squared_sum_x[zlost]
    Delta_det_merged = slogdet(Delta0[:d,:d] + squared_sum_x_sum + prior_outer[:d,:d] - kappank_sum * np.outer(mean_k_sum,mean_k_sum))[1]
    accept_ratio = -.5*d*(log(kappa0) + np.sum(log(kappank[[zmerge,zlost]])) - log(np.sum(kappa0 + nk[[zmerge,zlost]])))
    accept_ratio -= .5*(nu0+d-1)*Delta0_det[d] + .5*(nu0+np.sum(nk[[zmerge,zlost]])+d-1)*Delta_det_merged 
    accept_ratio += .5*np.sum((nunk[[zmerge,zlost]]+d-1)*Delta_k_det[[zmerge,zlost]])
    accept_ratio += np.sum(gammaln(.5*(nu0+d-np.arange(d)+1))) + np.sum(gammaln(.5*(nu0+np.sum(nk[[zmerge,zlost]])+d-np.arange(d)+1)))
    accept_ratio -= np.sum(gammaln(.5*(np.subtract.outer(nunk[[zmerge,zlost]]+d,np.arange(d)+1))))
    accept_ratio += K*gammaln(float(alpha)/K) - (K-1)*gammaln(alpha/(K-1.0)) - np.sum(gammaln(nk + float(alpha)/K))
    accept_ratio += np.sum(gammaln(np.delete(nk,[zmerge,zlost]) + float(alpha)/(K-1.0))) + gammaln(np.sum(nk_rest) + float(alpha)/(K-1.0))
    accept_ratio -= log(1.0-omega) - prop_ratio

plt.scatter(X[:,0],X[:,1],c=z_prop)
accept_ratio






## Accept or reject the proposal
print str(exp(accept_ratio))
accept = (-np.random.exponential(1) < accept_ratio)
if accept:
    ## Update the stored values
    if split:
            z = np.copy(z_prop)
            nk[zsplit] = nk_rest[0]
            nk = np.append(nk,nk_rest[1])
            nunk = nu0 + nk 
            kappank = kappa0 + nk
            sum_x[zsplit] = sum_rest[0]
            sum_x = np.append(sum_x,sum_rest)
            mean_x = (prior_sum[:d] + sum_x) / kappank
            squared_sum_x[zsplit] = squared_sum_restricted[0]
            squared_sum_x[K] = squared_sum_restricted[1]
            Delta_k[zsplit] = Delta_restricted[0]
            Delta_k[K] = Delta_restricted[1]
            Delta_k_inv[zsplit] = Delta_restricted_inv[0]
            Delta_k_inv[K] = Delta_restricted_inv[1]
            Delta_k_det[zsplit] = Delta_restricted_det[0]
            Delta_k_det = np.append(Delta_k_det,Delta_restricted_det[1])
            ## Update K 
            K += 1
    else:
            z = np.copy(z_prop)
            nk[zmerge] += nk[zlost]
            nunk[zmerge] = nu0 + nk[zmerge]
            kappank[zmerge] = kappa0 + nk[zmerge]
            sum_x[zmerge] += sum_x[zlost]
            mean_k[zmerge] = (prior_sum[:d] + sum_x[zmerge]) / kappank[zmerge]
            squared_sum_x[zmerge] += squared_sum_x[zlost]
            Delta_k[zmerge] = Delta0[:d,:d] + squared_sum_x[zmerge] + prior_outer[:d,:d] - kappank[zmerge] * np.outer(mean_k[zmerge],mean_k[zmerge])
            Delta_k_inv[zmerge] = kappank[zmerge] / (kappank[zmerge] + 1.0) * inv(Delta_k[zmerge])
            Delta_k_det[zmerge] = slogdet(Delta_k[zmerge])[1]
            ## Delete components from vectors and dictionaries
            nk = np.delete(nk,zlost)
            nunk = np.delete(nunk,zlost)
            kappank = np.delete(kappank,zlost)
            sum_x = np.delete(sum_x,zlost,axis=0)
            mean_k = np.delete(mean_k,zlost,axis=0)
            ## Remove the element corresponding to the empty cluster
            del squared_sum_x[zlost]
            del Delta_k[zlost]
            del Delta_k_inv[zlost]
            Delta_k_det = np.delete(Delta_k_det,zlost)
            ## Update the dictionaries and the allocations z
            if zlost != K-1:
                for k in range(zlost,K-1):
                    squared_sum_x[k] = squared_sum_x[k+1]
                    Delta_k[k] = Delta_k[k+1]
                    Delta_k_inv[k] = Delta_k_inv[k+1]
                ## Remove the final term
                del squared_sum_x[K-1]
                del Delta_k[K-1]
                del Delta_k_inv[K-1] 
            ## Update K
            K -= 1















