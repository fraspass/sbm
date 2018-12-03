#!/usr/bin/env python
import sys
import argparse
from scipy.stats import t
from scipy.special import gammaln
import numpy as np
from numpy import pi,log,exp,sqrt
from numpy.linalg import slogdet,inv
import mvt

###############################################################
### Define class for Gibbs sampling on the undirected graph ###
###############################################################

g = gibbs_undirected(X)
g.init_dim(d=2)
g.init_cluster(z=np.copy(c))
g.prior_gauss(mean0=np.zeros(g.m),kappa0=0.1,nu0=1.0,Delta0=.1*np.diag(np.ones(g.m)))
for _ in range(100): print _; g.gibbs_communities(l=50)
for _ in range(100): print _; g.dimension_change()

class gibbs_undirected:
	
	## Initialise the class from the embeddings X
	def __init__(self,X):
		self.X = np.copy(X)
		self.n = self.X.shape[0]
		self.m = self.X.shape[1]
		## Calculate an array of outer products for the entire node set
		self.full_outer_x = np.zeros((self.n,self.m,self.m))
		for x in range(self.n):
			self.full_outer_x[x] = np.outer(X[x],X[x])
	
	## Initialise the optimal dimension d
	def init_dim(self,d,delta=0.1):
		self.d = d 
		self.delta = delta
	
	## Initialise the cluster structure (assuming that allocations in z range from 0 to K-1)
	def init_cluster(self,z,alpha=1.0,omega=0.1):
		self.z = z
		self.K = len(np.unique(z))
		self.nk = np.array([np.sum(self.z == k) for k in range(self.K)])
		self.alpha = alpha
		self.omega = omega
	
	## Add the prior parameters for the Gaussian components of the matrix
	def prior_gauss(self,mean0,Delta0,kappa0=1.0,nu0=1.0):
		self.mean0 = np.copy(mean0)
		self.kappa0 = kappa0
		self.nu0 = nu0
		self.Delta0 = np.copy(Delta0)
		## Compute kappan and nun
		self.kappan = kappa0 + self.n
		self.nun = nu0 + self.n
		## Compute kappank and nunk
		self.kappank = kappa0 + self.nk
		self.nunk = nu0 + self.nk
		## Calculate the prior sum
		self.prior_sum = kappa0 * mean0
		## Prior outer product, scaled by kappa0
		self.prior_outer = kappa0 * np.outer(mean0,mean0)
		## Posterior mean for the entire matrix
		self.post_mean_tot = (self.prior_sum + np.sum(self.X,axis=0)) / self.kappan
		## Posterior mean for the right hand side of the matrix
		self.mean_r = self.post_mean_tot[self.d:]
		## Initialise the sums
		self.sum_x = np.zeros((self.K,self.d))
		self.mean_k = np.zeros((self.K,self.d))
		self.squared_sum_x = {}
		for k in range(self.K):
			x = self.X[self.z == k,:self.d]
			self.sum_x[k] = x.sum(axis=0)
			self.mean_k[k] = (self.sum_x[k] + self.prior_sum[:self.d]) / (self.nk[k] + kappa0)
			self.squared_sum_x[k] = np.dot(x.T,x)
		## Calculate marginal posterior covariance for the entire matrix and its determinant (sequentially)
		self.post_Delta_tot = Delta0 + np.dot(self.X.T,self.X) + self.prior_outer - self.kappan * np.outer(self.post_mean_tot,self.post_mean_tot)
		self.Delta_r = self.post_Delta_tot[self.d:,self.d:]
		self.post_Delta_tot_det = np.zeros(self.m+1)
		for i in range(self.m+1):
			self.post_Delta_tot_det[i] = slogdet(self.post_Delta_tot[i:,i:])[1]
		self.Delta_r_det = self.post_Delta_tot_det[self.d]
		## Initialise the Deltas 
		self.Delta_k = {}
		self.Delta_k_inv = {}
		self.Delta_k_det = np.zeros(self.K)
		for k in range(self.K):
			self.Delta_k[k] = self.Delta0[:self.d,:self.d] + self.squared_sum_x[k] + self.prior_outer[:self.d,:self.d] - self.kappank[k] * \
				np.outer(self.mean_k[k],self.mean_k[k])
			self.Delta_k_inv[k] = self.kappank[k] / (self.kappank[k] + 1.0) * inv(self.Delta_k[k])
			self.Delta_k_det[k] = slogdet(self.Delta_k[k])[1]
		## Calculate the determinants sequentially for the prior
		self.Delta0_det = np.zeros(self.m+1)
		self.Delta0_det_r = np.zeros(self.m+1)
		for i in range(self.m+1):
			self.Delta0_det[i] = slogdet(self.Delta0[:i,:i])[1]
			self.Delta0_det_r[i] = slogdet(self.Delta0[i:,i:])[1]

	########################################################
	### a. Resample the allocations using Gibbs sampling ###
	########################################################
	def gibbs_communities(self,l=50):
		## Change the value of l when too large
		if l > self.n:
			l = self.n
		## Update the latent allocations in randomised order
		## Loop over the indices
		for j in np.random.permutation(range(self.n)):
			zold = self.z[j] 
			## Update Student's t parameters
			position = self.X[j,:self.d]
			out_position = np.outer(position,position)
			self.sum_x[zold] -= position
			self.squared_sum_x[zold] -= out_position
			self.nk[zold] -= 1.0
			self.nunk[zold] -= 1.0
			self.kappank[zold] -= 1.0
			mk_old = self.mean_k[zold]
			self.mean_k[zold] = (self.prior_sum[:self.d] + self.sum_x[zold]) / self.kappank[zold]
			## Update Delta and the parameters for the multivariate Student's t (store the old values in case the allocation does not change)
			Delta_old = self.Delta_k[zold]
			Delta_inv_old = self.Delta_k_inv[zold]
			Delta_det_old = self.Delta_k_det[zold]
			# Delta_k[zold] -= (kappank[zold] + 1.0) / kappank[zold] * np.outer(mk_old-position,mk_old-position)
			self.Delta_k[zold] = self.Delta0[:self.d,:self.d] + self.squared_sum_x[zold] + self.prior_outer[:self.d,:self.d] - self.kappank[zold] * \
				np.outer(self.mean_k[zold],self.mean_k[zold])
			self.Delta_k_inv[zold] = self.kappank[zold] / (self.kappank[zold] + 1.0) * inv(self.Delta_k[zold])
			sign_det, self.Delta_k_det[zold] = slogdet(self.Delta_k[zold])
			## Raise error if the matrix is not positive definite
			if sign_det <= 0.0: 
				raise ValueError("Covariance matrix is negative definite.")
			## Calculate the probability of allocation within each community
			community_probs = np.array([mvt.dmvt_efficient(x=position,mu=self.mean_k[i],Sigma_inv=self.Delta_k_inv[i], \
				Sigma_logdet=self.d*log((self.kappank[i] + 1.0)/ (self.kappank[i] * self.nunk[i])) + self.Delta_k_det[i],nu=self.nunk[i]) for i in range(self.K)])
			## Raise error if nan probabilities are computed
			if np.isnan(community_probs).any():
				raise ValueError("Error in the allocation probabilities. Check invertibility of the covariance matrices.")
			community_probs += log(self.nk + float(self.alpha)/self.K)
			community_probs = exp(community_probs)
			community_probs /= sum(community_probs)
			## Sample the new community allocation
			znew = int(np.random.choice(self.K,p=community_probs))
			self.z[j] = znew
			## Update the Student's t parameters accordingly
			self.sum_x[znew] += position
			self.squared_sum_x[znew] += out_position
			self.nk[znew] += 1.0
			self.nunk[znew] += 1.0
			self.kappank[znew] += 1.0
			self.mean_k[znew] = (self.prior_sum[:self.d] + self.sum_x[znew]) / self.kappank[znew]
			## If znew == zold, do not recompute inverses and determinants but just copy the old values
			if znew != zold:
				# Delta_k[znew] = Delta0[:d,:d] + squared_sum_x[znew] + prior_outer[:d,:d] - kappank[znew] * np.outer(mean_k[znew],mean_k[znew])
				self.Delta_k[znew] = self.Delta0[:self.d,:self.d] + self.squared_sum_x[znew] + self.prior_outer[:self.d,:self.d] - self.kappank[znew] * \
					np.outer(self.mean_k[znew],self.mean_k[znew])
				self.Delta_k_inv[znew] = self.kappank[znew] / (self.kappank[znew] + 1.0) * inv(self.Delta_k[znew])
				sign_det, self.Delta_k_det[znew] = slogdet(self.Delta_k[znew])
				## Raise error if the matrix is not positive definite
				if sign_det <= 0.0: 
					raise ValueError("Covariance matrix is negative definite.")
			else:
				self.Delta_k[znew] = Delta_old
				self.Delta_k_inv[znew] = Delta_inv_old
				self.Delta_k_det[znew] = Delta_det_old
		return None ## the global variables are updated within the function, no need to return anything

	###################################################
	### b. Propose a change in the latent dimension ###
	###################################################
	def dimension_change(self):
		## Propose a new value of d
		if self.d == 0:
			d_prop = 1
		elif self.d == self.m:
			d_prop = self.m-1
		else:
			d_prop = np.random.choice([self.d-1,self.d+1])
		## Calculate likelihood for the current value of d 
		squared_sum_x_prop = {}
		Delta_k_prop = {}
		Delta_k_det_prop = np.zeros(self.K)
		## Different proposed quantities according to the sampled value of d
		if d_prop > self.d:
			sum_x_prop = np.hstack((self.sum_x,np.array([np.sum(self.X[self.z == i,d_prop-1]) for i in range(self.K)],ndmin=2).T))
			mean_k_prop = np.divide((self.prior_sum[:d_prop] + sum_x_prop).T, self.kappank).T
			for i in range(self.K):
				squared_sum_x_prop[i] = self.full_outer_x[self.z == i,:d_prop,:d_prop].sum(axis=0)
				Delta_k_prop[i] = self.Delta0[:d_prop,:d_prop] + squared_sum_x_prop[i] + self.prior_outer[:d_prop,:d_prop] - \
					self.kappank[i] * np.outer(mean_k_prop[i],mean_k_prop[i])
				sign_det, Delta_k_det_prop[i] = slogdet(Delta_k_prop[i])
				if sign_det <= 0.0:
					raise ValueError("Covariance matrix for d_prop is not invertible. Check conditions.")
			mean_r_prop = self.mean_r[1:]
			Delta_r_prop = self.Delta_r[1:,1:]
			Delta_r_det_prop = self.post_Delta_tot_det[d_prop] # slogdet(Delta_r_prop)[1]
		else:
			sum_x_prop = self.sum_x[:,:-1]
			mean_k_prop = self.mean_k[:,:-1]
			for i in range(self.K):
				squared_sum_x_prop[i] = self.squared_sum_x[i][:-1,:-1]
				Delta_k_prop[i] = self.Delta0[:d_prop,:d_prop] + squared_sum_x_prop[i] + self.prior_outer[:d_prop,:d_prop] - \
					self.kappank[i] * np.outer(mean_k_prop[i],mean_k_prop[i])
				Delta_k_det_prop[i] = slogdet(Delta_k_prop[i])[1]
			mean_r_prop = np.insert(self.mean_r,0,self.post_mean_tot[d_prop])
			Delta_r_prop = self.post_Delta_tot[d_prop:,d_prop:] 
			## Alternatively: Delta0[d_prop:,d_prop:] + full_outer_x[:,d_prop:,d_prop:].sum(axis=0) + prior_outer[d_prop:,d_prop:] - kappan * np.outer(mean_r_prop,mean_r_prop)
			Delta_r_det_prop = self.post_Delta_tot_det[d_prop] ## Alternatively: logdet(Delta_r_prop)
		# Calculate old likelihood
		old_lik = 0
		if self.d != m:
			old_lik = .5*(self.m-self.d)*(log(self.kappa0) - log(self.kappan)) + .5*(self.nu0+self.m-self.d-1)*self.Delta0_det_r[self.d] - \
				.5*(self.nun+self.m-self.d-1)*self.Delta_r_det 
		# Add the community specific components to the log-likelihood
		if self.d != 0:
			for k in range(self.K):
				old_lik += .5*self.d*(log(self.kappa0) - log(self.kappank[k])) + .5*(self.nu0+self.d-1)*self.Delta0_det[self.d] - \
					.5*(self.nunk[k]+self.d-1)*self.Delta_k_det[k]
		# Calculate new likelihood
		new_lik = 0
		if d_prop != self.m:
			new_lik += .5*(self.m-d_prop)*(log(self.kappa0) - log(self.kappan)) + .5*(self.nu0+self.m-d_prop-1)*self.Delta0_det_r[d_prop] - \
				.5*(self.nun+self.m-d_prop-1)*Delta_r_det_prop 
		# Add the community specific components to the log-likelihood
		if d_prop != 0: 
			for k in range(self.K):
				new_lik += .5*d_prop*(log(self.kappa0) - log(self.kappank[k])) + .5*(self.nu0+d_prop-1)*self.Delta0_det[d_prop] - \
					.5*(self.nunk[k]+d_prop-1)*Delta_k_det_prop[k]
		## Add the remaining component of the likelihood ratio
		if d_prop > self.d:
			new_lik += np.sum(gammaln(.5*(self.nunk+d_prop-1)) - gammaln(.5*(self.nu0+d_prop-1)))
			old_lik += gammaln(.5*(self.nun+self.m-self.d-1)) - gammaln(.5*(self.nu0+self.m-self.d-1))
		else:
			old_lik += np.sum(gammaln(.5*(self.nunk+self.d-1)) - gammaln(.5*(self.nu0+self.d-1)))
			new_lik += gammaln(.5*(self.nun+self.m-d_prop-1)) - gammaln(.5*(self.nu0+self.m-d_prop-1))
		## Acceptance ratio
		accept_ratio = new_lik - old_lik + (d_prop - self.d) * log(1 - self.delta)
		accept = (-np.random.exponential(1) < accept_ratio)
		## If the proposal is accepted, update the parameters
		if accept:
			self.d = d_prop
			self.sum_x = sum_x_prop
			self.mean_k = mean_k_prop
			self.mean_r = mean_r_prop
			self.squared_sum_x = squared_sum_x_prop
			self.Delta_k = Delta_k_prop
			for k in range(self.K):
				self.Delta_k_inv[k] = self.kappank[k] / (self.kappank[k] + 1.0) * inv(self.Delta_k[k])
			self.Delta_k_det = Delta_k_det_prop
			self.Delta_r = Delta_r_prop
			self.Delta_r_det = Delta_r_det_prop
		return None




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
		pred_prob = exp(np.array([mvt.dmvt_efficient(x=position,mu=mean_rest[q],Sigma_inv=Delta_restricted_inv[q], \
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









		## Calculate acceptance ratio for proposal d-1
		#accept_ratio = 0
		#accept_ratio -= .5*log(kappan) + .5*(K-1)*log(kappa0) - gammaln(.5*(nun+m-d-1)) + gammaln(.5*(nu0+m-d-1)) - K*gammaln(.5*(nu0+d-1)) + \
		#				.5*(nu0+m-d-1)*Delta0_det_r[d] - .5*(nu0+m-d)*Delta0_det_r[d-1] + .5*K*(nu0+d-1)*Delta0_det[d] - \
		#				.5*K*(nu0+d-2)*Delta0_det[d-1]
		#accept_ratio -= .5*(nun+m-d)*Delta_r_det - .5*(nun+m-d-1)*Delta_r_det_prop 
		#accept_ratio -= np.sum(gammaln(.5*(nunk+d)) - .5*log(kappank) + .5*(nunk+d-2)*Delta_k_det - .5*(nunk+d-1)*Delta_k_det_prop)
		#accept_ratio -= log(1.0-delta) + log(.5)*int(d_prop == 1) - log(.5)*int(d == m)





				## Calculate acceptance ratio for proposal d+1
		#accept_ratio = 0
		#accept_ratio += .5*log(kappan) + .5*(K-1)*log(kappa0) - gammaln(.5*(nun+m-d-1)) + gammaln(.5*(nu0+m-d-1)) - K*gammaln(.5*(nu0+d)) + \
		#				np.sum(gammaln(.5*(nunk+d)) - .5*log(kappank))
		#accept_ratio += .5*(nu0+m-d-2)*Delta0_det_r[d+1] - .5*(nu0+m-d-1)*Delta0_det_r[d] + .5*K*(nu0+d)*Delta0_det[d+1] - .5*K*(nu0+d-1)*Delta0_det[d]
		#accept_ratio += .5*(nun+m-d-1)*Delta_r_det - .5*(nun+m-d-2)*Delta_r_det_prop 
		#accept_ratio += np.sum(.5*(nunk+d-1)*Delta_k_det - .5*(nunk+d)*Delta_k_det_prop)
		#accept_ratio += log(1.0-delta) + log(.5)*int(d == 1) - log(.5)*int(d_prop == m)



		        # for k in range(K):
        #    new_lik += .5*d_prop*(log(kappa0) - log(kappank[k])) + .5*(nu0+d_prop-1)*(Delta0_det[d_prop] - Delta_k_det_prop[k]) # new_lik += np.sum([gammaln(.5*(nunk[k]+d_prop-i)) - gammaln(.5*(nu0+d_prop-i)) for i in range(1,d_prop+1)])



        # for k in range(K):
        #     old_lik += .5*d*(log(kappa0) - log(kappank[k])) + .5*(nu0+d-1)*(Delta0_det[d] - Delta_k_det[k]) # old_lik += np.sum([gammaln(.5*(nunk[k]+d-i)) - gammaln(.5*(nu0+d-i)) for i in range(1,d+1)])
	
