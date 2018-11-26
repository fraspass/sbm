#!/usr/bin/env python
import sys
import argparse
from scipy.stats import t
from scipy.special import gammaln
import numpy as np
from numpy import pi,log,exp,sqrt
from numpy.linalg import slogdet,inv
from collections import Counter, OrderedDict
from operator import itemgetter
import pandas as pd

#############################################
### Utility functions used in the sampler ###
#############################################

################################
### Multivariate Student's t ###
################################

### Multivariate Student's t density (log)
def dmvt(x,mu,Sigma,nu):
	##### IMPORTANT: also the scalars MUST be np arrays
	## Check that mu and Sigma have the appropriate dimension
	if mu.shape[0] != Sigma.shape[0]:
		raise ValueError("The arrays mu and Sigma must have compatible dimension.")
	if Sigma.shape[0] != 1:
		if Sigma.shape[0] != Sigma.shape[1]:
			raise ValueError("Sigma must be a squared matrix.")
	if mu.shape[0] == 1:
		## Exception when the PDF is unidimensional 
		return t.logpdf(x,df=nu,loc=mu,scale=sqrt(Sigma))
	else:
		# Calculate the number of dimensions
		d = mu.shape[0] 
		# Calculate the ratio of Gamma functions
		gamma_ratio = gammaln(nu/2.0 + d/2.0) - gammaln(nu/2.0)
		# Calculate the logarithm of the determinant 
		logdet = -.5 * slogdet(Sigma)[1] - .5 * d * (log(pi) + log(nu))
		# Invert the scale matrix, centre the vector and calculate the main expression
		Sigma_inv = inv(Sigma)
		x_center = x - mu
		main_exp = -.5 * (nu + d) * log(1 + (x_center.dot(Sigma_inv)).dot(x_center) / nu)
		# Return the result
		return gamma_ratio + logdet + main_exp ## log-density of the Student's t

### Multivariate Student's t density (log) -- computed efficiently (determinants and inverses are stored)
def dmvt_efficient(x,mu,Sigma_inv,Sigma_logdet,nu):
	##### IMPORTANT: the scalars MUST be numpy arrays
	## Sigma_inv must be scaled by the degrees of freedom
	## Check that mu and Sigma have the appropriate dimension
	if mu.shape[0] != Sigma_inv.shape[0]:
		raise ValueError("The arrays mu and Sigma must have compatible dimension.")
	if Sigma_inv.shape[0] != 1:
		if Sigma_inv.shape[0] != Sigma_inv.shape[1]:
			raise ValueError("Sigma must be a squared matrix.")
	if mu.shape[0] == 1:
		## Exception when the PDF is unidimensional 
		return t.logpdf(x,df=nu,loc=mu,scale=1/sqrt(Sigma_inv))
	else:
		# Calculate the number of dimensions
		d = mu.shape[0]
		# Calculate the ratio of Gamma functions
		gamma_ratio = gammaln(.5*(nu + d)) - gammaln(.5*nu)
		# Calculate the logarithm of the determinant 
		logdet = -.5 * Sigma_logdet - .5 * d * (log(pi) + log(nu))
		# Centre the vector and calculate the main expression
		### IMPORTANT: Sigma_inv MUST BE SCALED by the degrees of freedom in input
		x_center = x - mu
		main_exp = -.5 * (nu + d) * log(1 + (x_center.dot(Sigma_inv)).dot(x_center))
		# Return the result
		return gamma_ratio + logdet + main_exp ## log-density of the Student's t

########################################################
### a. Resample the allocations using Gibbs sampling ###
########################################################

def gibbs_communities(l=50):
	## Import global parameters
	global X, m, n, K, d, z, alpha, prior_sum, prior_outer, nu0, kappa0, Delta0, sum_x, mean_k, squared_sum_x, Delta_k, Delta_k_inv, Delta_k_det, \
				nk, nunk, kappank
	## Change the value of l when too large
	if l > n:
		l = n
	## Update the latent allocations in randomised order
	## Loop over the indices
	for j in np.random.permutation(range(n)):
		zold = z[j] 
		## Update Student's t parameters
		position = X[j,:d]
		out_position = np.outer(position,position)
		sum_x[zold] -= position
		squared_sum_x[zold] -= out_position
		nk[zold] -= 1.0
		nunk[zold] -= 1.0
		kappank[zold] -= 1.0
		mk_old = mean_k[zold]
		mean_k[zold] = (prior_sum[:d] + sum_x[zold]) / kappank[zold]
		## Update Delta and the parameters for the multivariate Student's t (store the old values in case the allocation does not change)
		Delta_old = Delta_k[zold]
		Delta_inv_old = Delta_k_inv[zold]
		Delta_det_old = Delta_k_det[zold]
		# Delta_k[zold] -= (kappank[zold] + 1.0) / kappank[zold] * np.outer(mk_old-position,mk_old-position)
		Delta_k[zold] = Delta0[:d,:d] + squared_sum_x[zold] + prior_outer[:d,:d] - kappank[zold] * np.outer(mean_k[zold],mean_k[zold])
		Delta_k_inv[zold] = kappank[zold] / (kappank[zold] + 1.0) * inv(Delta_k[zold])
		sign_det, Delta_k_det[zold] = slogdet(Delta_k[zold])
		## Raise error if the matrix is not positive definite
		if sign_det <= 0.0: 
			raise ValueError("Covariance matrix is negative definite.")
		## Calculate the probability of allocation within each community
		community_probs = np.array([dmvt_efficient(x=position,mu=mean_k[i],Sigma_inv=Delta_k_inv[i], \
			Sigma_logdet=d*log((kappank[i] + 1.0)/ (kappank[i] * nunk[i])) + Delta_k_det[i],nu=nunk[i]) for i in range(K)])
		## Raise error if nan probabilities are computed
		if np.isnan(community_probs).any():
			raise ValueError("Error in the allocation probabilities. Check invertibility of the covariance matrices.")
		community_probs += log(nk + float(alpha)/K)
		community_probs = exp(community_probs)
		community_probs /= sum(community_probs)
		## Sample the new community allocation
		znew = int(np.random.choice(K,p=community_probs))
		z[j] = znew
		## Update the Student's t parameters accordingly
		sum_x[znew] += position
		squared_sum_x[znew] += out_position
		nk[znew] += 1.0
		nunk[znew] += 1.0
		kappank[znew] += 1.0
		mean_k[znew] = (prior_sum[:d] + sum_x[znew]) / kappank[znew]
		## If znew == zold, do not recompute inverses and determinants but just copy the old values
		if znew != zold:
			# Delta_k[znew] = Delta0[:d,:d] + squared_sum_x[znew] + prior_outer[:d,:d] - kappank[znew] * np.outer(mean_k[znew],mean_k[znew])
			Delta_k[znew] = Delta0[:d,:d] + squared_sum_x[znew] + prior_outer[:d,:d] - kappank[znew] * np.outer(mean_k[znew],mean_k[znew])
			Delta_k_inv[znew] = kappank[znew] / (kappank[znew] + 1.0) * inv(Delta_k[znew])
			sign_det, Delta_k_det[znew] = slogdet(Delta_k[znew])
			## Raise error if the matrix is not positive definite
			if sign_det <= 0.0: 
				raise ValueError("Covariance matrix is negative definite.")
		else:
			Delta_k[znew] = Delta_old
			Delta_k_inv[znew] = Delta_inv_old
			Delta_k_det[znew] = Delta_det_old
	return None ## the global variables are updated within the function, no need to return anything

###################################################
### b. Propose a change in the latent dimension ###
###################################################

def dimension_change():
	## Import global parameters
	global X, m, n, K, d, z, alpha, prior_outer, nu0, kappa0, Delta0, sum_x, mean_k, squared_sum_x, Delta_k, Delta_k_inv, Delta_k_det, \
				nk, nunk, kappank, mean_r, Delta_r, Delta_r_det, Delta0_det, Delta0_det_r, post_Delta_tot, post_Delta_tot_det, \
				full_outer_x, post_mean_tot
	## Propose a new balue of d
	if d == 1:
		d_prop = 2
	elif d == m:
		d_prop = m-1
	else:
		d_prop = np.random.choice([d-1,d+1])
	## Calculate likelihood for the current value of d 
	squared_sum_x_prop = {}
	Delta_k_prop = {}
	Delta_k_det_prop = np.zeros(K)
	## Different proposed quantities according to the sampled value of d
	if d_prop > d:
		sum_x_prop = np.hstack((sum_x,np.array([np.sum(X[z == i,d_prop-1]) for i in range(K)],ndmin=2).T))
		mean_k_prop = np.divide((prior_sum[:d_prop] + sum_x_prop).T, kappank).T
		for i in range(K):
			squared_sum_x_prop[i] = full_outer_x[z == i,:d_prop,:d_prop].sum(axis=0)
			Delta_k_prop[i] = Delta0[:d_prop,:d_prop] + squared_sum_x_prop[i] + prior_outer[:d_prop,:d_prop] - kappank[i] * np.outer(mean_k_prop[i],mean_k_prop[i])
			sign_det, Delta_k_det_prop[i] = slogdet(Delta_k_prop[i])
			if sign_det <= 0.0:
				raise ValueError("Covariance matrix for d_prop is not invertible. Check conditions.")
		mean_r_prop = mean_r[1:]
		Delta_r_prop = Delta_r[1:,1:]
		Delta_r_det_prop = post_Delta_tot_det[d_prop] # slogdet(Delta_r_prop)[1]
		## Calculate acceptance ratio for proposal d+1
		accept_ratio = 0
		accept_ratio += .5*log(kappan) + .5*(K-1)*log(kappa0) - gammaln(.5*(nun+m-d-1)) + gammaln(.5*(nu0+m-d-1)) - K*gammaln(.5*(nu0+d)) + \
						np.sum(gammaln(.5*(nunk+d)) - .5*log(kappank))
		accept_ratio += .5*(nu0+m-d-2)*Delta0_det_r[d+1] - .5*(nu0+m-d-1)*Delta0_det_r[d] + .5*K*(nu0+d)*Delta0_det[d+1] - .5*K*(nu0+d-1)*Delta0_det[d]
		accept_ratio += .5*(nun+m-d-1)*Delta_r_det - .5*(nun+m-d-2)*Delta_r_det_prop 
		accept_ratio += np.sum(.5*(nunk+d-1)*Delta_k_det - .5*(nunk+d)*Delta_k_det_prop)
		accept_ratio += log(1.0-delta) + log(.5)*int(d == 1) - log(.5)*int(d_prop == m)
	else:
		sum_x_prop = sum_x[:,:-1]
		mean_k_prop = mean_k[:,:-1]
		for i in range(K):
			squared_sum_x_prop[i] = squared_sum_x[i][:-1,:-1]
			Delta_k_prop[i] = Delta0[:d_prop,:d_prop] + squared_sum_x_prop[i] + prior_outer[:d_prop,:d_prop] - kappank[i] * np.outer(mean_k_prop[i],mean_k_prop[i])
			Delta_k_det_prop[i] = slogdet(Delta_k_prop[i])[1]
		mean_r_prop = np.insert(mean_r,0,post_mean_tot[d_prop])
		Delta_r_prop = post_Delta_tot[d_prop:,d_prop:] 
		## Alternatively: Delta0[d_prop:,d_prop:] + full_outer_x[:,d_prop:,d_prop:].sum(axis=0) + prior_outer[d_prop:,d_prop:] - kappan * np.outer(mean_r_prop,mean_r_prop)
		Delta_r_det_prop = post_Delta_tot_det[d_prop] ## Alternatively: logdet(Delta_r_prop)
		## Calculate acceptance ratio for proposal d-1
		accept_ratio = 0
		accept_ratio -= .5*log(kappan) + .5*(K-1)*log(kappa0) - gammaln(.5*nun) - (K-1)*gammaln(.5*nu0) + .5*(nu0+m-d-1)*Delta0_det_r[d] - \
						.5*(nu0+m-d)*Delta0_det_r[d-1] + .5*K*(nu0+d-1)*Delta0_det[d] - .5*K*(nu0+d-2)*Delta0_det[d-1]
		accept_ratio -= .5*(nun+m-d)*Delta_r_det - .5*(nun+m-d-1)*Delta_r_det_prop 
		accept_ratio -= np.sum(gammaln(.5*nunk) - .5*log(kappank) + .5*(nunk+d-2)*Delta_k_det - .5*(nunk+d-1)*Delta_k_det_prop)
		accept_ratio -= log(1.0-delta) + log(.5)*int(d_prop == 1) - log(.5)*int(d == m)
	## Accept or reject the proposal
	accept = (-np.random.exponential(1) < accept_ratio)
	## If the proposal is accepted, update the parameters
	if accept:
		d = d_prop
		sum_x = sum_x_prop
		mean_k = mean_k_prop
		mean_r = mean_r_prop
		squared_sum_x = squared_sum_x_prop
		Delta_k = Delta_k_prop
		for k in range(K):
			Delta_k_inv[k] = kappank[k] / (kappank[k] + 1.0) * inv(Delta_k[k])
		Delta_k_det = Delta_k_det_prop
		Delta_r = Delta_r_prop
		Delta_r_det = Delta_r_det_prop
		Delta_k_inv = {}
	return accept

#################################################
### c. Propose to split (or merge) two groups ###
#################################################

### Split-merge move
def split_merge():
	## Import global parameters
	global X, m, n, K, d, z, sum_x, squared_sum_x, mean_k, Delta_k, Delta_k_inv, Delta_k_det, nk, nunk, kappank, kappa0, kappan, nu0, nun, Delta0_det, omega
	## Randomly choose two indices
	i,j = np.random.choice(n,size=2,replace=False)
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
		Delta_restricted[znew] = Delta0[:d,:d] + squared_sum_restricted[q] + prior_outer[:d,:d] - kappa_rest[znew] * np.outer(mean_rest[znew],mean_rest[znew])
		Delta_restricted_inv[znew] = kappa_rest[znew] / (kappa_rest[znew] + 1.0) * inv(Delta_restricted[znew])
		Delta_restricted_det[znew] = slogdet(Delta_restricted[znew])[1]
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
		mean_k_sum = (prior_mean[:d] + sum_x_sum) / kappank_sum
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
	## Accept or reject the proposal
	print str(exp(accept_ratio))
	accept = (-np.random.exponential(1) < accept_ratio)
	if accept:
		## Update the stored values
		if split:
			z = np.copy(zprop)
			nk[zsplit] = nk_rest[0]
			nk = np.append(nk,nk_rest[1])
			nunk = nu0 + nk 
			kappank = kappa0 + nk
			sum_x[zsplit] = sum_rest[0]
			sum_x = np.append(sum_x,sum_rest)
			mean_x = (prior_mean[:d] + sum_x) / kappank
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
			z = np.copy(zprop)
			nk[zmerge] += nk[zlost]
			nunk[zmerge] = nu0 + nk[zmerge]
			kappank[zmerge] = kappa0 + nk[zmerge]
			sum_x[zmerge] += sum_rest[zlost]
			mean_k[zmerge] = (prior_mean[:d] + sum_x[zmerge]) / kappank[zmerge]
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
			if ind_delete != K-1:
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
	return accept

########################################################
### d. Propose to add (or delete) an empty component ###
########################################################

def propose_empty():
	## Import global parameters
	global X, m, n, K, d, z, sum_x, squared_sum_x, mean_k, Delta_k, nk, nunk, kappank, kappa0, kappan, nu0, nun, Delta0_det
	## Propose to add or remove an empty cluster
	if K == 1:
		K_prop = 2
	elif K == n:
		K_prop = n-1
	else:
		K_prop = np.random.choice([K-1,K+1])
	if np.sum(nuk == 0) == 0 and K_prop < K:
		return False
	## Common term for the acceptance probability
	accept_ratio = K*gammaln(float(alpha)/K) - K_prop*gammaln(float(alpha)/K_prop) + np.sum(gammaln(nuk + float(alpha)/K_prop)) - np.sum(gammaln(nuk + float(alpha)/K))
	## Add or subtract the term for the empty cluster according to the sampled value
	if K_prop > K:
		accept_ratio += gammaln(float(alpha)/K_prop) + log(.5)*int(K == 1) - log(.5)*int(K_prop == n)
	else:
		accept_ratio -= gammaln(float(alpha)/K_prop) + log(.5)*int(K_prop == 1) - log(.5)*int(K == n)
	## Accept or reject the proposal
	accept = (-np.random.exponential(1) < accept_ratio)
	## Scale all the values if an empty cluster is added
	if accept:
		if K_prop > K:
			nuk = np.append(nuk,0)
			sum_x = np.vstack((sum_x,np.zeros((1,d))))
			mean_k = np.vstack((mean_k,prior_mean[:d]/kappa0))
			squared_sum_x[K_prop] = np.zeros((d,d)) 
			Delta_k[K_prop] = prior_covariance[:d,:d]
			Delta_k_inv[K_prop] = prior_covariance_inverse[d]
			Delta_k_det[K_prop] = prior_determinant[d]
		else:
			## Delete empty cluster with largest index
			ind_delete = np.where(nunk - nu0 == 0)[0][-1]
			nunk_prop = np.delete(nuk,ind_delete)
			sum_x = np.delete(sum_x,ind_delete,axis=0)
			mean_k = np.delete(mean_k,ind_delete,axis=0)
			## Remove the element corresponding to the empty cluster
			del squared_sum_x[ind_delete]
			del Delta_k[ind_delete]
			del Delta_k_inv[ind_delete]
			Delta_k_det = np.delete(Delta_k_det,ind_delete)
			## Update the dictionaries and the allocations z
			if ind_delete != K_prop:
				for k in range(ind_delete,K_prop):
					squared_sum_x[k] = squared_sum_x[k+1]
					Delta_k[k] = Delta_k[k+1]
					Delta_k_inv[k] = Delta_k_inv[k+1]
					z[z == k+1] = k
				## Remove the final term
				del squared_sum_x[K_prop]
				del Delta_k[K_prop]
				del Delta_k_inv[K_prop] 
	return accept
