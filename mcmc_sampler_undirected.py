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
	def init_cluster(self,z,K=0,alpha=1.0,omega=0.1):
		self.z = np.copy(z)
		if K != 0 and K >= len(np.unique(z)):
			self.K = K
		else:
			self.K = len(np.unique(z))
		self.nk = np.array([np.sum(self.z == k) for k in range(self.K)])
		self.alpha = alpha
		self.omega = omega
	
	## Add the prior parameters for the Gaussian components of the matrix
	def prior_gauss(self,mean0,Delta0,kappa0=1.0,nu0=1.0,covstrut='full',meanstrut='unknown'):
		## Covariance and mean structure
		self.covstrut = covstrut
		self.meanstrut = meanstrut
		## Intialise the parameters of the NIW distribution
		self.mean0 = np.copy(mean0)
		self.kappa0 = kappa0
		self.nu0 = nu0
		if covstrut == 'diagonal':
			self.Delta0 = np.diag(np.diag(Delta0))
		else:
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
		## Posterior mean for the entire matrix and for the right hand side of the matrix
		if meanstrut == 'unknown':
			self.post_mean_tot = (self.prior_sum + np.sum(self.X,axis=0)) / self.kappan
			self.mean_r = self.post_mean_tot[self.d:]
		else:
			self.post_mean_tot = np.copy(mean0)
			self.mean_r = mean0[self.d:]
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
		if covstrut == 'diagonal':
			self.post_Delta_tot = np.diag(np.diag(Delta0 + np.dot(self.X.T,self.X) + self.prior_outer - self.kappan * np.outer(self.post_mean_tot,self.post_mean_tot)))
		else:
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
	def dimension_change(self,verbose=False):
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
		if self.d != self.m:
			## According to the mean structure define the old log-likelihood
			if self.meanstrut == 'unknown':
				old_lik = .5*(self.m-self.d)*(log(self.kappa0) - log(self.kappan)) 
			## According to the covariance structure define the old log-likelihood
			if self.covstrut == 'diagonal':
				old_lik += .5*self.Delta0_det_r[self.d] - .5*self.Delta_r_det
			else:
				old_lik += .5*(self.nu0+self.m-self.d-1)*self.Delta0_det_r[self.d] - .5*(self.nun+self.m-self.d-1)*self.Delta_r_det
		# Add the community specific components to the log-likelihood
		if self.d != 0:
			for k in range(self.K):
				old_lik += .5*self.d*(log(self.kappa0) - log(self.kappank[k])) + .5*(self.nu0+self.d-1)*self.Delta0_det[self.d] - \
					.5*(self.nunk[k]+self.d-1)*self.Delta_k_det[k]
		# Calculate new likelihood
		new_lik = 0
		if d_prop != self.m:
			## According to the mean structure define the old log-likelihood
			if self.meanstrut == 'unknown':			
				new_lik += .5*(self.m-d_prop)*(log(self.kappa0) - log(self.kappan)) 
			## According to the covariance structure define the old log-likelihood
			if self.covstrut == 'diagonal':
				new_lik += .5*self.Delta0_det_r[d_prop] - .5*Delta_r_det_prop
			else:
				new_lik += .5*(self.nu0+self.m-d_prop-1)*self.Delta0_det_r[d_prop] - .5*(self.nun+self.m-d_prop-1)*Delta_r_det_prop 
		# Add the community specific components to the log-likelihood
		if d_prop != 0: 
			for k in range(self.K):
				new_lik += .5*d_prop*(log(self.kappa0) - log(self.kappank[k])) + .5*(self.nu0+d_prop-1)*self.Delta0_det[d_prop] - \
					.5*(self.nunk[k]+d_prop-1)*Delta_k_det_prop[k]
		## Add the remaining component of the likelihood ratio
		if d_prop > self.d:
			new_lik += np.sum(gammaln(.5*(self.nunk+d_prop-1)) - gammaln(.5*(self.nu0+d_prop-1)))
			if self.covstrut == 'diagonal':
				old_lik += gammaln(.5*self.nun) - gammaln(.5*self.nu0)
			else:
				old_lik += gammaln(.5*(self.nun+self.m-self.d-1)) - gammaln(.5*(self.nu0+self.m-self.d-1))
		else:
			old_lik += np.sum(gammaln(.5*(self.nunk+self.d-1)) - gammaln(.5*(self.nu0+self.d-1)))
			if self.covstrut == 'diagonal':
				new_lik += gammaln(.5*self.nun) - gammaln(.5*self.nu0)
			else:
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
		if verbose:
			print 'Proposal: '+str(d_prop)+'\t'+'Accepted: '+str(accept)
		return None

	#################################################
	### c. Propose to split (or merge) two groups ###
	#################################################
	### Split-merge move
	def split_merge(self,verbose=False):
		## Randomly choose two indices
		i,j = np.random.choice(self.n,size=2,replace=False)
		## Propose a split or merge move according to the sampled values
		if self.z[i] == self.z[j]:
			split = True
			zsplit = self.z[i]
			## Obtain the indices that must be re-allocated
			S = np.delete(range(self.n),[i,j])[np.delete(self.z == zsplit,[i,j])]
			z_prop = np.copy(self.z)
			## Move j to a new cluster
			z_prop[j] = self.K
		else:
			split = False
			## Choose to merge to the cluster with minimum index (zmerge) and remove the cluster with maximum index (zlost)
			zmerge = min([self.z[i],self.z[j]])
			zj = self.z[j]
			zlost = max([self.z[i],self.z[j]])
			z_prop = np.copy(self.z)
			z_prop[self.z == zlost] = zmerge
			if zlost != self.K-1:
				for k in range(zlost,self.K-1):
					z_prop[self.z == k+1] = k
			## Set of observations for the imaginary split (Dahl, 2003)
			S = np.delete(range(self.n),[i,j])[np.delete((self.z == zmerge) + (self.z == zlost),[i,j])]
		## Initialise the proposal ratio
		prop_ratio = 0
		## Construct vectors for sequential posterior predictives
		nk_rest = np.ones(2,int)
		nunk_rest = self.nu0 + np.ones(2,int)
		kappa_rest = self.kappa0 + np.ones(2,int)
		sum_rest = self.X[[i,j],:self.d]
		mean_rest = (self.prior_sum[:self.d] + sum_rest) / (self.kappa0 + 1.0)
		squared_sum_restricted = self.full_outer_x[[i,j],:self.d,:self.d]
		Delta_restricted = {}; Delta_restricted_inv = {}; Delta_restricted_det = np.zeros(2)
		for q in [0,1]:
			Delta_restricted[q] = self.Delta0[:self.d,:self.d] + squared_sum_restricted[q] + self.prior_outer[:self.d,:self.d] - \
				kappa_rest[q] * np.outer(mean_rest[q],mean_rest[q])
			Delta_restricted_inv[q] = kappa_rest[q] / (kappa_rest[q] + 1) * inv(Delta_restricted[q])
			Delta_restricted_det[q] = slogdet(Delta_restricted[q])[1]
		## Randomly permute the indices in S and calculate the sequential allocations
		for h in np.random.permutation(S):
			## Calculate the predictive probability
			position = self.X[h,:self.d]
			out_position = np.outer(position,position)
			pred_prob = exp(np.array([mvt.dmvt_efficient(x=position,mu=mean_rest[q],Sigma_inv=Delta_restricted_inv[q], \
				Sigma_logdet=self.d*log((kappa_rest[q] + 1.0) / (kappa_rest[q] * nunk_rest[q]))+Delta_restricted_det[q], nu=nunk_rest[q]) for q in [0,1]]) + \
				log(nunk_rest[q] + self.alpha/2.0))
			pred_prob /= sum(pred_prob)
			if split:
				## Sample the new value
				znew = np.random.choice(2,p=pred_prob)
				## Update proposal ratio
				prop_ratio += log(pred_prob[znew])
				## Update proposed z
				z_prop[h] = [zsplit,self.K][znew]
			else:
				## Determine the new value deterministically
				znew = int(self.z[h] == zj)
				## Update proposal ratio in the imaginary split
				prop_ratio += log(pred_prob[znew])
			## Update parameters 
			nk_rest[znew] += 1
			nunk_rest[znew] += 1
			kappa_rest[znew] += 1
			sum_rest[znew] += position
			mean_rest[znew] = (self.prior_sum[:self.d] + sum_rest[znew]) / kappa_rest[znew]
			squared_sum_restricted[znew] += out_position
			Delta_restricted[znew] = self.Delta0[:self.d,:self.d] + squared_sum_restricted[znew] + self.prior_outer[:self.d,:self.d] - \
				kappa_rest[znew] * np.outer(mean_rest[znew],mean_rest[znew])
			Delta_restricted_inv[znew] = kappa_rest[znew] / (kappa_rest[znew] + 1.0) * inv(Delta_restricted[znew])
			Delta_restricted_det[znew] = slogdet(Delta_restricted[znew])[1]
		## Calculate the acceptance probability
		if split:
			## Calculate the acceptance ratio
			accept_ratio = .5*self.d*(log(self.kappa0) + log(self.kappank[zsplit]) - np.sum(log(kappa_rest))) 
			accept_ratio += .5*(self.nu0+self.d-1)*self.Delta0_det[self.d] + .5*(self.nunk[zsplit]+self.d-1)*self.Delta_k_det[zsplit] -\
				.5*np.sum((nunk_rest+self.d-1)*Delta_restricted_det)
			accept_ratio += np.sum(gammaln(.5*(np.subtract.outer(nunk_rest+self.d,np.arange(self.d)+1))))
			accept_ratio -= np.sum(gammaln(.5*(self.nu0+self.d-np.arange(self.d)+1))) + np.sum(gammaln(.5*(self.nunk[zsplit]+self.d-np.arange(self.d)+1))) 
			accept_ratio += self.K*gammaln(float(self.alpha)/self.K) - (self.K+1)*gammaln(self.alpha/(self.K+1)) - np.sum(gammaln(self.nk + float(self.alpha)/self.K))
			accept_ratio += np.sum(gammaln(np.delete(self.nk,zsplit) + self.alpha/(self.K+1.0))) + np.sum(gammaln(nk_rest + self.alpha/(self.K+1.0)))
			accept_ratio += log(1.0-self.omega) - prop_ratio
		else:
			## Merge the two clusters and calculate the acceptance ratio
			nk_sum = np.sum(self.nk[[zmerge,zlost]])
			nunk_sum = self.nu0 + nk_sum
			kappank_sum = self.kappa0 + nk_sum
			sum_x_sum = self.sum_x[zmerge] + self.sum_x[zlost]
			mean_k_sum = (self.prior_sum[:self.d] + sum_x_sum) / kappank_sum
			squared_sum_x_sum = self.squared_sum_x[zlost] + self.squared_sum_x[zlost]
			Delta_det_merged = slogdet(self.Delta0[:self.d,:self.d] + squared_sum_x_sum + self.prior_outer[:self.d,:self.d] - kappank_sum * np.outer(mean_k_sum,mean_k_sum))[1]
			accept_ratio = -.5*self.d*(log(self.kappa0) + np.sum(log(self.kappank[[zmerge,zlost]])) - log(np.sum(self.kappa0 + self.nk[[zmerge,zlost]])))
			accept_ratio -= .5*(self.nu0+self.d-1)*self.Delta0_det[self.d] + .5*(self.nu0+np.sum(self.nk[[zmerge,zlost]])+self.d-1)*Delta_det_merged 
			accept_ratio += .5*np.sum((self.nunk[[zmerge,zlost]]+self.d-1)*self.Delta_k_det[[zmerge,zlost]])
			accept_ratio += np.sum(gammaln(.5*(self.nu0+self.d-np.arange(self.d)+1))) + np.sum(gammaln(.5*(self.nu0+np.sum(self.nk[[zmerge,zlost]])+self.d-np.arange(self.d)+1)))
			accept_ratio -= np.sum(gammaln(.5*(np.subtract.outer(self.nunk[[zmerge,zlost]]+self.d,np.arange(self.d)+1))))
			accept_ratio += self.K*gammaln(float(self.alpha)/self.K) - (self.K-1)*gammaln(self.alpha/(self.K-1.0)) - np.sum(gammaln(self.nk + float(self.alpha)/self.K))
			accept_ratio += np.sum(gammaln(np.delete(self.nk,[zmerge,zlost]) + float(self.alpha)/(self.K-1.0))) + gammaln(np.sum(nk_rest) + float(self.alpha)/(self.K-1.0))
			accept_ratio -= log(1.0-self.omega) - prop_ratio
		## Accept or reject the proposal
		accept = (-np.random.exponential(1) < accept_ratio)
		if accept:
			## Update the stored values
			if split:
				self.z = np.copy(z_prop)
				self.nk[zsplit] = nk_rest[0]
				self.nk = np.append(self.nk,nk_rest[1])
				self.nunk = self.nu0 + self.nk 
				self.kappank = self.kappa0 + self.nk
				self.sum_x[zsplit] = sum_rest[0]
				self.sum_x = np.vstack((self.sum_x,sum_rest[1]))
				self.mean_k = np.divide((self.prior_sum[:self.d] + self.sum_x).T,self.kappank).T
				self.squared_sum_x[zsplit] = squared_sum_restricted[0]
				self.squared_sum_x[self.K] = squared_sum_restricted[1]
				self.Delta_k[zsplit] = Delta_restricted[0]
				self.Delta_k[self.K] = Delta_restricted[1]
				self.Delta_k_inv[zsplit] = Delta_restricted_inv[0]
				self.Delta_k_inv[self.K] = Delta_restricted_inv[1]
				self.Delta_k_det[zsplit] = Delta_restricted_det[0]
				self.Delta_k_det = np.append(self.Delta_k_det,Delta_restricted_det[1])
				## Update K 
				self.K += 1
			else:
				self.z = np.copy(z_prop)
				self.nk[zmerge] += self.nk[zlost]
				self.nunk[zmerge] = self.nu0 + self.nk[zmerge]
				self.kappank[zmerge] = self.kappa0 + self.nk[zmerge]
				self.sum_x[zmerge] += self.sum_x[zlost]
				self.mean_k[zmerge] = (self.prior_sum[:self.d] + self.sum_x[zmerge]) / self.kappank[zmerge]
				self.squared_sum_x[zmerge] += self.squared_sum_x[zlost]
				self.Delta_k[zmerge] = self.Delta0[:self.d,:self.d] + self.squared_sum_x[zmerge] + self.prior_outer[:self.d,:self.d] - \
					self.kappank[zmerge] * np.outer(self.mean_k[zmerge],self.mean_k[zmerge])
				self.Delta_k_inv[zmerge] = self.kappank[zmerge] / (self.kappank[zmerge] + 1.0) * inv(self.Delta_k[zmerge])
				self.Delta_k_det[zmerge] = slogdet(self.Delta_k[zmerge])[1]
				## Delete components from vectors and dictionaries
				self.nk = np.delete(self.nk,zlost)
				self.nunk = np.delete(self.nunk,zlost)
				self.kappank = np.delete(self.kappank,zlost)
				self.sum_x = np.delete(self.sum_x,zlost,axis=0)
				self.mean_k = np.delete(self.mean_k,zlost,axis=0)
				## Remove the element corresponding to the empty cluster
				del self.squared_sum_x[zlost]
				del self.Delta_k[zlost]
				del self.Delta_k_inv[zlost]
				self.Delta_k_det = np.delete(self.Delta_k_det,zlost)
				## Update the dictionaries and the allocations z
				if zlost != self.K-1:
					for k in range(zlost,self.K-1):
						self.squared_sum_x[k] = self.squared_sum_x[k+1]
						self.Delta_k[k] = self.Delta_k[k+1]
						self.Delta_k_inv[k] = self.Delta_k_inv[k+1]
					## Remove the final term
					del self.squared_sum_x[self.K-1]
					del self.Delta_k[self.K-1]
					del self.Delta_k_inv[self.K-1] 
				## Update K
				self.K -= 1
		if verbose:
			print 'Proposal: '+['MERGE','SPLIT'][split]+'\t'+'Accepted: '+str(accept)
		return None

	########################################################
	### d. Propose to add (or delete) an empty component ###
	########################################################
	def propose_empty(self,verbose=False):
		## Propose to add or remove an empty cluster
		if self.K == 1:
			K_prop = 2
		elif self.K == self.n:
			K_prop = self.n-1
		else:
			K_prop = np.random.choice([self.K-1,self.K+1])
		## Assign values to the variable remove
		if K_prop < self.K:
			remove = True
		else:
			remove = False
		## If there are no empty clusters and K_prop = K-1, reject the proposal
		if not (self.nk == 0).any() and K_prop < self.K:
			return None
		## Propose a new vector of cluster allocations
		if K_prop < self.K:
			## Delete empty cluster with largest index
			ind_delete = np.where(self.nk == 0)[0][-1]
			nk_prop = np.delete(self.nk,ind_delete)
		else:
			nk_prop = np.append(self.nk,0)
		## Common term for the acceptance probability
		accept_ratio = self.K*gammaln(float(self.alpha)/self.K) - K_prop*gammaln(float(self.alpha)/K_prop) + np.sum(gammaln(nk_prop + float(self.alpha)/K_prop)) - \
			np.sum(gammaln(self.nk + float(self.alpha)/self.K)) + (K_prop - self.K)*log(1-self.omega) * log(.5)*int(self.K == 1) - log(.5)*int(self.K == self.n)
		## Accept or reject the proposal
		accept = (-np.random.exponential(1) < accept_ratio)
		## Scale all the values if an empty cluster is added
		if accept:
			if K_prop > self.K:
				self.nk = nk_prop
				self.kappank = np.append(self.kappank,self.kappa0)
				self.nunk = np.append(self.nunk,self.nu0)
				self.sum_x = np.vstack((self.sum_x,np.zeros((1,self.d))))
				self.mean_k = np.vstack((self.mean_k,self.prior_sum[:self.d]/self.kappa0))
				self.squared_sum_x[self.K] = np.zeros((self.d,self.d)) 
				self.Delta_k[self.K] = self.Delta0[:self.d,:self.d] + self.prior_outer[:self.d,:self.d]
				self.Delta_k_inv[self.K] = self.kappank[self.K] / (self.kappank[self.K] + 1.0) * inv(self.Delta_k[self.K])
				self.Delta_k_det = np.append(self.Delta_k_det,slogdet(self.Delta_k[self.K])[1])
			else:
				self.nk = nk_prop
				self.kappank = np.delete(self.kappank,ind_delete)
				self.nunk = np.delete(self.nunk,ind_delete)
				self.sum_x = np.delete(self.sum_x,ind_delete,axis=0)
				self.mean_k = np.delete(self.mean_k,ind_delete,axis=0)
				## Remove the element corresponding to the empty cluster
				del self.squared_sum_x[ind_delete]
				del self.Delta_k[ind_delete]
				del self.Delta_k_inv[ind_delete]
				self.Delta_k_det = np.delete(self.Delta_k_det,ind_delete)
				## Update the dictionaries and the allocations z
				if ind_delete != K_prop:
					for k in range(ind_delete,K_prop):
						self.squared_sum_x[k] = self.squared_sum_x[k+1]
						self.Delta_k[k] = self.Delta_k[k+1]
						self.Delta_k_inv[k] = self.Delta_k_inv[k+1]
						self.z[self.z == k+1] = k
					## Remove the final term
					del self.squared_sum_x[K_prop]
					del self.Delta_k[K_prop]
					del self.Delta_k_inv[K_prop] 
			self.K = K_prop
		if verbose:
			print 'Proposal: '+['ADD','REMOVE'][remove]+'\t'+'Accepted: '+str(accept)
		return None

	## Utility function: calculates the marginal likelihood for all the possible values of d given a set of allocations z
	def marginal_likelihoods_dimension(self):
		## Initialise sum, mean and sum of squares
		sums = np.zeros((self.K,self.m))
		means = np.zeros((self.K,self.m))
		Deltank = {}
		Deltank_det = np.zeros((self.K,self.m+1))
		for k in range(self.K):
			x = self.X[self.z==k]
			sums[k] = x.sum(axis=0)
			means[k] = (sums[k] + self.prior_sum) / (self.nk[k] + self.kappa0)
			Deltank[k] = self.Delta0 + np.dot(x.T,x) + self.prior_outer - (self.kappa0 + self.nk[k]) * np.outer(means[k],means[k])
			for d in range(self.m+1):
				Deltank_det[k,d] = slogdet(Deltank[k][:d,:d])[1]
		## Initialise determinant for the right hand side matrices
		Deltar_det = np.zeros(self.m+1)
		for d in range(self.m+1):
			Deltar_det[d] = slogdet(self.post_Delta_tot[d:,d:])[1] 
		## Calculate the determinants sequentialy for the prior
		Delta0_det = np.zeros(self.m+1)
		Delta0_det_r = np.zeros(self.m+1)
		for i in range(self.m+1):
			Delta0_det[i] = slogdet(self.Delta0[:i,:i])[1]
			Delta0_det_r[i] = slogdet(self.Delta0[i:,i:])[1]
		## Calculate the marginal likelihoods sequentially
		self.mlik = np.zeros(self.m+1)
		for d in range(self.m+1):
		## Calculate the marginal likelihood for the garbage Gaussian
			if d != self.m:
				self.mlik[d] = -.5*self.n*(self.m-d)*log(np.pi) 
				self.mlik[d] += .5*(self.nu0+self.m-d-1)*Delta0_det_r[d] - .5*(self.nu0+self.n+self.m-d-1)*Deltar_det[d]
				if self.covstrut == 'diagonal':
					self.mlik[d] += (self.m-d)*(gammaln(.5*self.nun) - gammaln(.5*self.nu0))
				else:
					self.mlik[d] += np.sum([(gammaln(.5*(self.nun+self.m-d-i)) - gammaln(.5*(self.nu0+self.m-d-i))) for i in range(1,self.m-d+1)])
				if self.meanstrut == 'unknown':
					self.mlik[d] += .5*(self.m-d)*(log(self.kappa0) - log(self.kappan)) 
			else:
				self.mlik[d] = 0
			## Calculate the marginal likelihood for the community allocations
			if d != 0: 
				for k in range(self.K):
					self.mlik[d] += -.5*self.nk[k]*d*log(np.pi) + .5*d*(log(self.kappa0) - log(self.nk[k] + self.kappa0)) + .5*(self.nu0+d-1)*Delta0_det[d] - \
						.5*(self.nu0+self.nk[k]+d-1)*Deltank_det[k,d]
					self.mlik[d] += np.sum([(gammaln(.5*(self.nu0+self.nk[k]+d-i)) - gammaln(.5*(self.nu0+d-i))) for i in range(1,d+1)]) 

