#!/usr/bin/env python
import sys
import argparse
from scipy.stats import t
from scipy.special import gammaln
import numpy as np
from numpy import pi,log,exp,sqrt
from numpy.linalg import slogdet,inv
import mvt
import warnings
from collections import Counter

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
	def prior_gauss_left(self,mean0,Delta0,kappa0=1.0,nu0=1.0):
		## Intialise the parameters of the NIW distribution
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
		## Initialise the sums
		self.sum_x = np.zeros((self.K,self.d))
		self.mean_k = np.zeros((self.K,self.d))
		self.squared_sum_x = {}
		for k in range(self.K):
			x = self.X[self.z == k,:self.d]
			self.sum_x[k] = x.sum(axis=0)
			self.mean_k[k] = (self.sum_x[k] + self.prior_sum[:self.d]) / (self.nk[k] + kappa0)
			self.squared_sum_x[k] = np.dot(x.T,x)
		## Initialise the Deltas 
		self.Delta_k = {}
		self.Delta_k_inv = {}
		self.Delta_k_det = np.zeros(self.K)
		for k in range(self.K):
			self.Delta_k[k] = self.Delta0[:self.d,:self.d] + self.squared_sum_x[k] + self.prior_outer[:self.d,:self.d] - \
				self.kappank[k] * np.outer(self.mean_k[k],self.mean_k[k])
			self.Delta_k_inv[k] = self.kappank[k] / (self.kappank[k] + 1.0) * inv(self.Delta_k[k])
			self.Delta_k_det[k] = slogdet(self.Delta_k[k])[1]
		## Calculate the determinants sequentially for the prior
		self.Delta0_det = np.zeros(self.m+1)
		for i in range(self.m+1):
			self.Delta0_det[i] = slogdet(self.Delta0[:i,:i])[1]

	def prior_gauss_right(self,sigma0,lambda0=1.0):
		## sigma0 is a n-dimensional vector of variences
		self.sigma0 = sigma0
		self.lambda0 = lambda0
		## Calculate the updated values
		self.prior_sigma = lambda0 * sigma0
		self.lambdank = lambda0 + self.nk
		self.sigmank = np.zeros((self.K,self.m-self.d)) 
		for k in range(self.K):
			self.sigmank[k] = self.prior_sigma[self.d:] + np.sum(self.X[self.z == k,self.d:] ** 2,axis=0)
		## Initialise equal_var to false 
		if not hasattr(self,'equal_var'):
			self.equal_var = False

    ### Must be initialised AFTER prior_gauss_right
	def init_group_variance(self,v,H=0,beta=1.0,csi=0.1):
		self.v = np.copy(v)
		if len(v) != self.K:
			raise ValueError('v must have dimension K')
		if H != 0 and H >= len(np.unique(v)):
			self.H = H
		else:
			self.H = len(np.unique(H))
		#if H > self.K:
		#	raise ValueError('H must be less than or equal to K')
		## Initialise second order cluster allocation counts
		self.vk = np.zeros(self.H)
		for h in range(self.H):
			self.vk[h] = np.sum(self.v == h)
		## Initialise the other parameters
		self.beta = beta
		self.csi = csi
		self.equal_var = True
		lambdank_v = np.zeros(self.H)
		sigmank_v = np.zeros((self.H,self.m-self.d))
		for h in range(self.H):
			lambdank_v[h] = self.lambda0 + np.sum(self.nk[self.v == h])
			sigmank_v[h] = self.prior_sigma[self.d:] + np.sum(self.sigmank[self.v == h] - self.prior_sigma[self.d:], axis=0)
		self.lambdank = lambdank_v
		self.sigmank = sigmank_v

	########################################################
	### a. Resample the allocations using Gibbs sampling ###
	########################################################
	def gibbs_communities(self,l=50):
		## Change the value of l when too large
		if l > self.n:
			l = self.n
		## Update the latent allocations in randomised order
		## Loop over the indices
		for j in np.random.permutation(range(self.n))[:l]:
			zold = self.z[j] 
			## Update parameters of the distribution
			position = self.X[j,:self.d]
			position_right = self.X[j,self.d:]
			out_position = np.outer(position,position)
			self.sum_x[zold] -= position
			self.squared_sum_x[zold] -= out_position
			self.nk[zold] -= 1.0
			self.nunk[zold] -= 1.0
			self.kappank[zold] -= 1.0
			self.lambdank[self.v[zold] if self.equal_var else zold] -= 1.0
			self.sigmank[self.v[zold] if self.equal_var else zold] -= (position_right ** 2)
			self.mean_k[zold] = (self.prior_sum[:self.d] + self.sum_x[zold]) / self.kappank[zold]
			## Update Delta (store the old values in case the allocation does not change)
			Delta_old = self.Delta_k[zold]
			Delta_inv_old = self.Delta_k_inv[zold]
			Delta_det_old = self.Delta_k_det[zold]
			self.Delta_k[zold] = self.Delta0[:self.d,:self.d] + self.squared_sum_x[zold] + self.prior_outer[:self.d,:self.d] - self.kappank[zold] * \
				np.outer(self.mean_k[zold],self.mean_k[zold])
			self.Delta_k_inv[zold] = self.kappank[zold] / (self.kappank[zold] + 1.0) * inv(self.Delta_k[zold])
			sign_det, self.Delta_k_det[zold] = slogdet(self.Delta_k[zold])
			## Raise error if the matrix is not positive definite
			if sign_det <= 0.0: 
				raise ValueError("Covariance matrix is negative definite.")
			## Calculate the probability of allocation for the left hand side of the matrix
			community_probs_left = np.array([mvt.dmvt_efficient(x=position,mu=self.mean_k[i],Sigma_inv=self.Delta_k_inv[i], \
				Sigma_logdet=self.d * log((self.kappank[i] + 1.0) / (self.kappank[i] * self.nunk[i])) + \
				self.Delta_k_det[i],nu=self.nunk[i]) for i in range(self.K)]).reshape(self.K,)
			## Raise error if nan probabilities are computed
			if np.isnan(community_probs_left).any():
				raise ValueError("Error in the allocation probabilities. Check invertibility of the covariance matrices.")
			## Calculate the probability of allocation for the left hand side of the matrix
			if self.d != self.m:
				if self.equal_var:
					community_probs_right = np.array([np.sum(t.logpdf(position_right, df=self.lambdank[self.v[k]], loc=0, \
						scale=sqrt(self.sigmank[self.v[k]] / self.lambdank[self.v[k]]))) for k in range(self.K)])
				else:
					community_probs_right = np.array([np.sum(t.logpdf(position_right, df=self.lambdank[k], loc=0, \
						scale=sqrt(self.sigmank[k] / self.lambdank[k]))) for k in range(self.K)])
			else:
				community_probs_right = np.zeros(self.K) 
			if (self.lambdank <= 0.0).any():
				raise ValueError('Problem with lambda')
			if (self.sigmank <= 0.0).any():
				raise ValueError('Problem with sigma')
			if np.isnan(community_probs_right).any():
				raise ValueError("Error in the allocation probabilities. Check variances of right hand side of the matrix.")
			## Calculate last component of the probability of allocation
			community_probs_allo = log(self.nk + float(self.alpha)/self.K)
			## Compute the allocation probabilities
			community_probs = exp(community_probs_left + community_probs_right + community_probs_allo)
			community_probs /= sum(community_probs)
			## Sample the new community allocation
			znew = int(np.random.choice(self.K,p=community_probs))
			self.z[j] = znew
			## Update the Student's t parameters accordingly
			self.sum_x[znew] += position
			self.squared_sum_x[znew] += out_position
			self.lambdank[self.v[znew] if self.equal_var else znew] += 1.0
			self.sigmank[self.v[znew] if self.equal_var else znew] += (position_right ** 2)
			self.nk[znew] += 1.0
			self.nunk[znew] += 1.0
			self.kappank[znew] += 1.0
			self.mean_k[znew] = (self.prior_sum[:self.d] + self.sum_x[znew]) / self.kappank[znew]
			## If znew == zold, do not recompute inverses and determinants but just copy the old values
			if znew != zold:
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
		if (self.sigmank < 0).any() or (self.lambdank < 0).any():
			return ValueError('Error with sigma and/or lambda')
		vv = Counter(self.v)
		vvv = np.array([vv[key] for key in range(self.H)])
		if (vvv != self.vk).any():
			raise ValueError('Error with vs')
		return None ## the global variables are updated within the function, no need to return anything

	###################################################
	### b. Propose a change in the latent dimension ###
	###################################################
	def dimension_change(self,verbose=False):
		## Propose a new value of d
		if self.d == 1:
			d_prop = 2
		elif self.d == self.m:
			d_prop = self.m-1
		else:
			d_prop = np.random.choice([self.d-1, self.d+1])
		## Calculate likelihood for the current value of d 
		squared_sum_x_prop = {}
		Delta_k_prop = {}
		Delta_k_det_prop = np.zeros(self.K)
		## Different proposed quantities according to the sampled value of d
		if d_prop > self.d:
			sum_x_prop = np.hstack((self.sum_x, np.array([np.sum(self.X[self.z == i,d_prop-1]) for i in range(self.K)], ndmin=2).T))
			mean_k_prop = np.divide((self.prior_sum[:d_prop] + sum_x_prop).T, self.kappank).T
			for i in range(self.K):
				squared_sum_x_prop[i] = self.full_outer_x[self.z == i,:d_prop,:d_prop].sum(axis=0)
				Delta_k_prop[i] = self.Delta0[:d_prop,:d_prop] + squared_sum_x_prop[i] + self.prior_outer[:d_prop,:d_prop] - \
					self.kappank[i] * np.outer(mean_k_prop[i],mean_k_prop[i])
				sign_det, Delta_k_det_prop[i] = slogdet(Delta_k_prop[i])
				if sign_det <= 0.0:
					raise ValueError("Covariance matrix for d_prop is not invertible. Check conditions.")
			if self.equal_var:
				sigmank_prop = np.zeros((self.H,self.m - d_prop))
			else:
				sigmank_prop = np.zeros((self.K,self.m - d_prop))
			for k in range(sigmank_prop.shape[0]):
				sigmank_prop[k] = self.sigmank[k,1:]
		else:
			sum_x_prop = self.sum_x[:,:-1]
			mean_k_prop = self.mean_k[:,:-1]
			for i in range(self.K):
				squared_sum_x_prop[i] = self.squared_sum_x[i][:-1,:-1]
				Delta_k_prop[i] = self.Delta0[:d_prop,:d_prop] + squared_sum_x_prop[i] + self.prior_outer[:d_prop,:d_prop] - \
					self.kappank[i] * np.outer(mean_k_prop[i],mean_k_prop[i])
				Delta_k_det_prop[i] = slogdet(Delta_k_prop[i])[1]
			## Pay attention to the index to add
			if self.equal_var:
				sigmank_prop = np.zeros((self.H,self.m - d_prop))
				for h in range(self.H):
					sigmank_prop[h] = np.insert(self.sigmank[h], 0, self.prior_sigma[d_prop] + np.sum(self.X[self.v[self.z] == h,d_prop] ** 2))
			else:
				sigmank_prop = np.zeros((self.K,self.m - d_prop))
				for k in range(self.K):
					sigmank_prop[k] = np.insert(self.sigmank[k], 0, self.prior_sigma[d_prop] + np.sum(self.X[self.z == k,d_prop] ** 2))
		# Calculate old likelihood
		old_lik = 0
		# Add the community specific components to the log-likelihood
		if self.d != 0:
			for k in range(self.K):
				old_lik += .5 * self.d * (log(self.kappa0) - log(self.kappank[k])) + .5 * (self.nu0 + self.d - 1) * self.Delta0_det[self.d] - \
					.5 * (self.nunk[k] + self.d - 1) * self.Delta_k_det[k]
			for k in range(self.sigmank.shape[0]):	
				old_lik += (self.m - self.d) * (gammaln(.5 * self.lambdank[k]) - gammaln(.5 * self.lambda0))
		# Calculate new likelihood
		new_lik = 0
		# Add the community specific components to the log-likelihood
		if d_prop != 0: 
			for k in range(self.K):
				new_lik += .5 * d_prop * (log(self.kappa0) - log(self.kappank[k])) + .5 * (self.nu0 + d_prop - 1) * self.Delta0_det[d_prop] - \
					.5 * (self.nunk[k] + d_prop - 1) * Delta_k_det_prop[k]
			for k in range(self.sigmank.shape[0]):
				new_lik += (self.m - d_prop) * (gammaln(.5 * self.lambdank[k]) - gammaln(.5 * self.lambda0))
		## Add the remaining component of the likelihood ratio
		if d_prop > self.d:
			new_lik += np.sum(gammaln(.5 * (self.nunk + d_prop - 1)) - gammaln(.5 * (self.nu0 + d_prop - 1)))
			old_lik += np.sum(.5 * self.lambda0 * log(self.prior_sigma[self.d]) - .5 * self.lambdank * log(self.sigmank[:,0]))
		else:
			old_lik += np.sum(gammaln(.5 * (self.nunk + self.d - 1)) - gammaln(.5 * (self.nu0 + self.d - 1)))
			new_lik += np.sum(.5 * self.lambda0 * log(self.prior_sigma[d_prop]) - .5 * self.lambdank * log(sigmank_prop[:,0]))
		## Acceptance ratio
		accept_ratio = new_lik - old_lik + (d_prop - self.d) * log(1 - self.delta)
		accept = ( -np.random.exponential(1) < accept_ratio )
		## If the proposal is accepted, update the parameters
		if accept:
			self.d = d_prop
			self.sum_x = sum_x_prop
			self.mean_k = mean_k_prop
			self.squared_sum_x = squared_sum_x_prop
			self.sigmank = sigmank_prop
			self.Delta_k = Delta_k_prop
			for k in range(self.K):
				self.Delta_k_inv[k] = self.kappank[k] / (self.kappank[k] + 1.0) * inv(self.Delta_k[k])
			self.Delta_k_det = Delta_k_det_prop
		if verbose:
			print 'Proposal: ' + str(d_prop) + '\t' + 'Accepted: ' + str(accept)
		if (self.sigmank < 0).any() or (self.lambdank < 0).any():
			return ValueError('Error with sigma and/or lambda')
		vv = Counter(self.v)
		vvv = np.array([vv[key] for key in range(self.H)])
		if (vvv != self.vk).any():
			raise ValueError('Error with vs')
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
		nunk_rest = self.nu0 + np.ones(2)
		kappa_rest = self.kappa0 + np.ones(2)
		sum_rest = self.X[[i,j],:self.d]
		if not self.equal_var:
			lambda_rest = self.lambda0 + np.ones(2)
			sigma_rest = self.prior_sigma[self.d:] + (self.X[[i,j],self.d:] ** 2)
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
			position_right = self.X[h,self.d:]
			out_position = np.outer(position,position)
			log_pred_prob = np.array([(mvt.dmvt_efficient(x=position,mu=mean_rest[q],Sigma_inv=Delta_restricted_inv[q], \
				Sigma_logdet=self.d*log((kappa_rest[q] + 1.0) / (kappa_rest[q] * nunk_rest[q])) + Delta_restricted_det[q], \
				nu=nunk_rest[q])) for q in [0,1]]).reshape(2,) + log(nunk_rest[q] + self.alpha/2.0)
			## Calculate the probability of allocation for the right hand side of the matrix (only when equal_var is false)
			## if not self.equal_var: 
				## log_pred_prob += np.array([np.sum(t.logpdf(position_right,df=lambda_rest[q],loc=0,scale=sqrt(sigma_rest[q]/lambda_rest[q]))) for q in [0,1]])
			## Compute the allocation probabilities
			pred_prob = exp(log_pred_prob)
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
				if pred_prob[znew] == 0.0:
					warnings.warn('Imaginary split yields impossible outcome: merge proposal automatically rejected')
			## Calculate second order cluster allocation (restricted posterior predictive)
			## Update parameters 
			nk_rest[znew] += 1.0
			nunk_rest[znew] += 1.0
			kappa_rest[znew] += 1.0
			sum_rest[znew] += position
			if not self.equal_var:
				lambda_rest[znew] += 1.0
				sigma_rest[znew] += (position_right ** 2)
			mean_rest[znew] = (self.prior_sum[:self.d] + sum_rest[znew]) / kappa_rest[znew]
			squared_sum_restricted[znew] += out_position
			Delta_restricted[znew] = self.Delta0[:self.d,:self.d] + squared_sum_restricted[znew] + self.prior_outer[:self.d,:self.d] - \
				kappa_rest[znew] * np.outer(mean_rest[znew],mean_rest[znew])
			Delta_restricted_inv[znew] = kappa_rest[znew] / (kappa_rest[znew] + 1.0) * inv(Delta_restricted[znew])
			Delta_restricted_det[znew] = slogdet(Delta_restricted[znew])[1]
		## Calculate the new second order allocation if equal_var is true
		if self.equal_var:
			if split:
				## Calculate the new second order allocation of the first cluster after splitting
				ind_left = np.where(z_prop == zsplit)[0]
				lambda_left = np.copy(self.lambdank)
				lambda_left[self.v[zsplit]] -= nk_rest[0]
				sigma_left = np.copy(self.sigmank)
				sigma_left[self.v[zsplit]] -= np.sum(self.X[ind_left,self.d:] ** 2,axis=0)
				prob_v_left = np.zeros(self.H)
				for node in np.random.permutation(ind_left):
					pos = self.X[node,self.d:]
					prob_v_left += np.array([np.sum(t.logpdf(pos,df=lambda_left[h],loc=0,scale=sqrt(sigma_left[h]/lambda_left[h]))) for h in range(self.H)])
					lambda_left += 1.0
					sigma_left += (pos ** 2)
				## Calculate the second order allocation of the second cluster after splitting
				ind_right = np.where(z_prop == self.K)[0]
				lambda_right = np.copy(self.lambdank)
				lambda_right[self.v[zsplit]] -= nk_rest[1]
				sigma_right = np.copy(self.sigmank)
				sigma_right[self.v[zsplit]] -= np.sum(self.X[ind_right,self.d:] ** 2,axis=0)
				prob_v_right = np.zeros(self.H)
				for node in np.random.permutation(ind_right):
					pos = self.X[node,self.d:]
					prob_v_right += np.array([np.sum(t.logpdf(pos,df=lambda_right[h],loc=0,scale=sqrt(sigma_right[h]/lambda_right[h]))) for h in range(self.H)])
					lambda_right += 1.0
					sigma_right += (pos ** 2)
			else:
				## Calculate the new second order allocation of the first cluster after the (imaginary) split
				lambda_left = np.copy(self.lambdank)
				lambda_left[self.v[zmerge]] -= self.nk[zmerge]
				sigma_left = np.copy(self.sigmank)
				sigma_left[self.v[zmerge]] -= np.sum(self.X[self.z == zmerge,self.d:] ** 2,axis=0)
				prob_v_left = np.zeros(self.H)
				for node in np.random.permutation(np.where(self.z == zmerge)[0]):
					pos = self.X[node,self.d:]
					prob_v_left += np.array([np.sum(t.logpdf(pos,df=lambda_left[h],loc=0,scale=sqrt(sigma_left[h]/lambda_left[h]))) for h in range(self.H)])
					lambda_left += 1.0
					sigma_left += (pos ** 2)
				## Calculate the new second order allocation of the first cluster after the (imaginary) split
				lambda_right = np.copy(self.lambdank)
				lambda_right[self.v[zlost]] -= self.nk[zlost]
				sigma_right = np.copy(self.sigmank)
				sigma_right[self.v[zlost]] -= np.sum(self.X[self.z == zlost,self.d:] ** 2,axis=0)
				prob_v_right = np.zeros(self.H)
				for node in np.random.permutation(np.where(self.z == zlost)[0]):
					pos = self.X[node,self.d:]
					prob_v_right += np.array([np.sum(t.logpdf(pos,df=lambda_right[h],loc=0,scale=sqrt(sigma_right[h]/lambda_right[h]))) for h in range(self.H)])
					lambda_right += 1.0
					sigma_right += (pos ** 2)
			## Resample the second order cluster allocation
			prob_v_left = exp(prob_v_left-max(prob_v_left)) / np.sum(exp(prob_v_left-np.max(prob_v_left)))
			left_sord = np.random.choice(range(self.H),p=prob_v_left)
			prob_v_right = exp(prob_v_right-max(prob_v_right)) / np.sum(exp(prob_v_right-max(prob_v_right)))
			right_sord = np.random.choice(range(self.H),p=prob_v_right)
			## Calculate the cumulative probability of allocation to the specific pair (left_sord and right_sord)
			prob_second_order = log(prob_v_left[left_sord]) + log(prob_v_right[right_sord])
			## Compute the proposed values of lambdank and sigmank
			lambdank_prop = np.copy(self.lambdank)
			sigmank_prop = np.copy(self.sigmank)
			v_prop = np.copy(self.v)
			vk_prop = np.copy(self.vk)
			if split:
				## Propose new values for the right hand side of the matrix
				v_prop[zsplit] = left_sord
				v_prop = np.append(v_prop,right_sord)
				## Second order cluster counts
				vk_prop[self.v[zsplit]] -= 1.0
				vk_prop[left_sord] += 1.0
				vk_prop[right_sord] += 1.0
				## Update lambda and sigma
				lambdank_prop[self.v[zsplit]] -= self.nk[zsplit]
				lambdank_prop[left_sord] += nk_rest[0]
				lambdank_prop[right_sord] += nk_rest[1]
				sigmank_prop[self.v[zsplit]] -= np.sum(self.X[self.z == zsplit,self.d:] ** 2, axis=0)
				sigmank_prop[left_sord] += np.sum(self.X[ind_left, self.d:] ** 2, axis=0)
				sigmank_prop[right_sord] += np.sum(self.X[ind_right, self.d:] **2, axis=0)
			else:
				v_prop = np.delete(v_prop,zlost)
				samp_so = np.random.choice([0,1])
				second_order = [self.v[zmerge],self.v[zlost]][samp_so]
				v_prop[zmerge] = second_order
				vk_prop[[self.v[zmerge],self.v[zlost]][1-samp_so]] -= 1.0
				lambdank_prop[self.v[zmerge]] -= self.nk[zmerge]
				lambdank_prop[self.v[zlost]] -= self.nk[zlost]
				lambdank_prop[second_order] += self.nk[zmerge] + self.nk[zlost]
				Xzm = np.sum(self.X[self.z == zmerge, self.d:] ** 2, axis=0)
				Xzl = np.sum(self.X[self.z == zlost, self.d:] ** 2, axis=0)
				sigmank_prop[self.v[zmerge]] -= Xzm
				sigmank_prop[self.v[zlost]] -= Xzl
				sigmank_prop[second_order] += Xzm + Xzl
		## Calculate the acceptance probability
		if split:
			## Calculate the acceptance ratio
			## Left hand side of matrix
			accept_ratio = .5 * self.d * (log(self.kappa0) + log(self.kappank[zsplit]) - np.sum(log(kappa_rest))) 
			accept_ratio += .5 * (self.nu0 + self.d - 1) * self.Delta0_det[self.d] + .5 * (self.nunk[zsplit] + self.d - 1) * self.Delta_k_det[zsplit] -\
				.5 * np.sum((nunk_rest + self.d - 1) * Delta_restricted_det)
			accept_ratio += np.sum(gammaln(.5 * (np.subtract.outer(nunk_rest + self.d,np.arange(self.d) + 1))))
			accept_ratio -= np.sum(gammaln(.5 * (self.nu0 + self.d - np.arange(self.d) + 1))) + np.sum(gammaln(.5 * (self.nunk[zsplit] + self.d - np.arange(self.d) + 1))) 
			## Right hand side of matrix
			if self.equal_var:
				accept_ratio += np.sum([(self.m - self.d) * (gammaln(.5 * lambdank_prop[h]) - gammaln(.5 * self.lambda0)) + \
					np.sum(.5*self.lambda0*log(self.prior_sigma[self.d:]) - .5*lambdank_prop[h]*log(sigmank_prop[h])) for h in range(self.H)])
				accept_ratio -= np.sum([(self.m - self.d) * (gammaln(.5 * self.lambdank[h]) - gammaln(.5 * self.lambda0)) + \
					np.sum(.5*self.lambda0*log(self.prior_sigma[self.d:]) - .5*self.lambdank[h]*log(self.sigmank[h])) for h in range(self.H)])
				accept_ratio += np.sum(gammaln(vk_prop + self.beta / self.H)) - np.sum(gammaln(self.vk + self.beta / self.H)) + gammaln(self.K + self.beta) - \
									gammaln(self.K + 1 + self.beta)
				accept_ratio -= prob_second_order
			else:
				accept_ratio += (self.m - self.d) * (np.sum(gammaln(.5 * lambda_rest)) - gammaln(.5 * self.lambdank[zsplit]) - gammaln(.5 * self.lambda0))
				accept_ratio += np.sum(.5 * self.lambda0 * log(self.prior_sigma[self.d:]) + .5 * self.lambdank[zsplit] * log(self.sigmank[zsplit]))
				accept_ratio -= .5 * np.sum(np.multiply(lambda_rest,log(sigma_rest).T).T)
			## Cluster allocations
			accept_ratio += self.K * gammaln(float(self.alpha) / self.K) - (self.K+1) * gammaln(self.alpha / (self.K+1)) - np.sum(gammaln(self.nk + float(self.alpha)/self.K))
			accept_ratio += np.sum(gammaln(np.delete(self.nk,zsplit) + self.alpha/(self.K+1.0))) + np.sum(gammaln(nk_rest + self.alpha / (self.K+1.0)))
			## Prior on K and q function
			accept_ratio += log(1.0 - self.omega) - prop_ratio
		else:
			## Merge the two clusters and calculate the acceptance ratio
			nk_sum = np.sum(self.nk[[zmerge,zlost]])
			nunk_sum = self.nu0 + nk_sum
			kappank_sum = self.kappa0 + nk_sum
			lambdank_sum = self.lambda0 + nk_sum
			sum_x_sum = self.sum_x[zmerge] + self.sum_x[zlost]
			## sigma_sum = self.sigmank[zmerge] + self.sigmank[zlost] - self.prior_sigma[self.d:]
			mean_k_sum = (self.prior_sum[:self.d] + sum_x_sum) / kappank_sum
			squared_sum_x_sum = self.squared_sum_x[zlost] + self.squared_sum_x[zlost]
			Delta_det_merged = slogdet(self.Delta0[:self.d,:self.d] + squared_sum_x_sum + self.prior_outer[:self.d,:self.d] - kappank_sum * np.outer(mean_k_sum,mean_k_sum))[1]
			## Caculate acceptance ratio
			## Left hand side of matrix
			accept_ratio = -.5 * self.d * (log(self.kappa0) + np.sum(log(self.kappank[[zmerge,zlost]])) - log(self.kappa0 + np.sum(self.nk[[zmerge,zlost]])))
			accept_ratio -= .5 * (self.nu0+self.d-1)*self.Delta0_det[self.d] + .5 * (self.nu0+np.sum(self.nk[[zmerge,zlost]]) + self.d - 1) * Delta_det_merged 
			accept_ratio += .5 * np.sum((self.nunk[[zmerge,zlost]] + self.d - 1) * self.Delta_k_det[[zmerge,zlost]])
			accept_ratio += np.sum(gammaln(.5 * (self.nu0 + self.d - np.arange(self.d) + 1))) + np.sum(gammaln(.5 * (self.nu0 + np.sum(self.nk[[zmerge,zlost]]) + \
				self.d - np.arange(self.d)+1)))
			accept_ratio -= np.sum(gammaln(.5 * (np.subtract.outer(self.nunk[[zmerge,zlost]] + self.d,np.arange(self.d) + 1))))
			## Right hand side of matrix
			if self.equal_var:
				accept_ratio += np.sum([(self.m - self.d) * (gammaln(.5 * lambdank_prop[h]) - gammaln(.5 * self.lambda0)) + \
					np.sum(.5*self.lambda0*log(self.prior_sigma[self.d:]) - .5*lambdank_prop[h]*log(sigmank_prop[h])) for h in range(self.H)])
				accept_ratio -= np.sum([(self.m - self.d) * (gammaln(.5 * self.lambdank[h]) - gammaln(.5 * self.lambda0)) + \
					np.sum(.5*self.lambda0*log(self.prior_sigma[self.d:]) - .5*self.lambdank[h]*log(self.sigmank[h])) for h in range(self.H)])
				accept_ratio += np.sum(gammaln(vk_prop + self.beta / self.H)) - np.sum(gammaln(self.vk + self.beta / self.H)) + gammaln(self.K + self.beta) - \
									gammaln(self.K - 1 + self.beta)
				accept_ratio += prob_second_order - (self.v[zmerge] != self.v[zlost])*log(.5)
			else:
				accept_ratio -= (self.m - self.d) * (np.sum(gammaln(.5 * self.lambdank[[zmerge,zlost]])) - gammaln(.5 * self.lambda0) - \
					gammaln(.5 * (self.lambda0 + np.sum(self.nk[[zmerge,zlost]]))))
				accept_ratio -= np.sum(.5 * self.lambda0 * log(self.prior_sigma[self.d:])) - \
					.5 * np.sum(np.multiply(self.lambdank[[zmerge,zlost]],log(self.sigmank[[zmerge,zlost]]).T).T)
				accept_ratio -= np.sum(.5 * (self.lambda0 + np.sum(self.nk[[zmerge,zlost]])) * log(np.sum(self.sigmank[[zmerge,zlost]],axis=0) - self.prior_sigma[self.d:]))
			## Cluster allocations
			accept_ratio += self.K * gammaln(float(self.alpha) / self.K) - (self.K-1) * gammaln(self.alpha / (self.K-1.0)) - \
				np.sum(gammaln(self.nk + float(self.alpha) / self.K))
			accept_ratio += np.sum(gammaln(np.delete(self.nk,[zmerge,zlost]) + float(self.alpha)/(self.K-1.0))) + gammaln(np.sum(nk_rest) + float(self.alpha)/(self.K-1.0))
			## Prior on K and q function
			accept_ratio -= log(1.0 - self.omega) - prop_ratio
			#if ((np.sum(self.sigmank[[zmerge,zlost]],axis=0) - self.prior_sigma[self.d:]) <= 0.0).any():
			#	raise ValueError('ERROR')
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
				if self.equal_var:
					self.v = v_prop
					self.vk = vk_prop
					self.lambdank = lambdank_prop
					self.sigmank = sigmank_prop
				else:
					self.lambdank = self.lambda0 + self.nk
					self.sigmank[zsplit] = sigma_rest[0]
					self.sigmank = np.vstack((self.sigmank,sigma_rest[1]))
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
				if self.equal_var:
					self.v = v_prop
					self.vk = vk_prop
					self.lambdank = lambdank_prop
					self.sigmank = sigmank_prop
				else:
					self.lambdank[zmerge] = self.lambda0 + self.nk[zmerge]
					self.sigmank[zmerge] += self.sigmank[zlost] - self.prior_sigma[self.d:]
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
				if not self.equal_var:
					self.lambdank = np.delete(self.lambdank,zlost)
					self.sigmank = np.delete(self.sigmank,zlost,axis=0)
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
			print 'Proposal: ' + ['MERGE','SPLIT'][split] + '\t' + 'Accepted: ' + str(accept)
		if (self.sigmank < 0).any() or (self.lambdank < 0).any():
			return ValueError('Error with sigma and/or lambda')
		vv = Counter(self.v)
		vvv = np.array([vv[key] for key in range(self.H)])
		if (vvv != self.vk).any():
			raise ValueError('Error with vs')
		return None

	########################################################
	### d. Propose to add (or delete) an empty component ###
	########################################################
	def propose_empty(self,verbose=False):
		## Propose to add or remove an empty cluster
		if self.K == 1:
			K_prop = 2
		elif self.K == self.n:
			K_prop = self.n - 1
		else:
			K_prop = np.random.choice([self.K-1,self.K+1])
		## Assign values to the variable remove
		if K_prop < self.K:
			remove = True
		else:
			remove = False
		## If there are no empty clusters and K_prop = K-1, reject the proposal
		if not (self.nk == 0).any() and K_prop < self.K:
			if verbose:
				print 'Proposal: ' + 'REMOVE' + '\t' + 'Accepted: ' + 'False'
			return None
		## Propose a new vector of cluster allocations
		if remove:
			## Delete empty cluster with largest index (or sample at random)
			ind_delete = np.random.choice(np.where(self.nk == 0)[0])##[-1]
			nk_prop = np.delete(self.nk,ind_delete)
			## Remove the second order cluster
			if self.equal_var:
				vk_prop = np.copy(self.vk)
				vk_prop[self.v[ind_delete]] -= 1
				v_prop = np.delete(self.v,ind_delete)
		else:
			nk_prop = np.append(self.nk,0)
			## Propose the new second order cluster
			if self.equal_var:
				v_prop = np.append(self.v,np.random.choice(self.H))
				vk_prop = np.copy(self.vk)
				vk_prop[v_prop[-1]] += 1.0 
		## Common term for the acceptance probability
		accept_ratio = self.K*gammaln(float(self.alpha) / self.K) - K_prop * gammaln(float(self.alpha) / K_prop) + \
								np.sum(gammaln(nk_prop + float(self.alpha) / K_prop)) - np.sum(gammaln(self.nk + float(self.alpha) / self.K)) + \
								(K_prop - self.K) * log(1 - self.omega) * log(.5) * int(self.K == 1) - log(.5) * int(self.K == self.n)
		## Add the equal variance component
		if self.equal_var:
			accept_ratio += gammaln(self.K + self.beta) - gammaln(K_prop + self.beta) - np.sum(gammaln(self.vk + float(self.beta) / self.H)) + \
								np.sum(gammaln(vk_prop + float(self.beta) / self.H))
			if remove:
				accept_ratio += log(1.0 / self.H) - log(1.0 / (np.sum(self.nk == 0)))
			else:
				accept_ratio += log(1.0 / (np.sum(self.nk == 0)+1)) - log(1.0 / self.H)
		## Accept or reject the proposal
		accept = (-np.random.exponential(1) < accept_ratio)
		## Scale all the values if an empty cluster is added
		if accept:
			if K_prop > self.K:
				self.nk = nk_prop
				self.kappank = np.append(self.kappank,self.kappa0)
				self.nunk = np.append(self.nunk,self.nu0)
				if self.equal_var:
					self.v = v_prop
					self.vk = vk_prop
				else:
					self.lambdank = np.append(self.lambdank,self.lambda0)
					self.sigmank = np.vstack((self.sigmank,self.prior_sigma[self.d:] * np.ones((1,self.m-self.d))))
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
				if self.equal_var:
					self.vk = vk_prop
					self.v = v_prop
				else:
					self.lambdank = np.delete(self.lambdank,ind_delete)
					self.sigmank = np.delete(self.sigmank,ind_delete,axis=0)
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
			print 'Proposal: ' + ['ADD','REMOVE'][remove] + '\t' + 'Accepted: ' + str(accept)
		if (self.sigmank < 0).any() or (self.lambdank < 0).any():
			return ValueError('Error with sigma and/or lambda')
		vv = Counter(self.v)
		vvv = np.array([vv[key] for key in range(self.H)])
		if (vvv != self.vk).any():
			raise ValueError('Error with vs')
		return None

	#####################################################################
	### Compute marginal likelihoods for all the possible values of d ###
	#####################################################################
	## Utility function: calculates the marginal likelihood for all the possible values of d given a set of allocations z
	def marginal_likelihoods_dimension(self):
		## Initialise sum, mean and sum of squares
		sums = np.zeros((self.K,self.m))
		squares = np.zeros((self.sigmank.shape[0],self.m))
		means = np.zeros((self.K,self.m))
		Deltank = {}
		Deltank_det = np.zeros((self.K,self.m+1))
		for k in range(self.K):
			x = self.X[self.z == k]
			sums[k] = x.sum(axis=0)
			means[k] = (sums[k] + self.prior_sum) / (self.nk[k] + self.kappa0)
			if not self.equal_var:
				squares[k] = self.prior_sigma + (x ** 2).sum(axis=0)
			Deltank[k] = self.Delta0 + np.dot(x.T,x) + self.prior_outer - (self.kappa0 + self.nk[k]) * np.outer(means[k],means[k])
			for d in range(self.m+1):
				Deltank_det[k,d] = slogdet(Deltank[k][:d,:d])[1] 
		if self.equal_var:
			for h in range(self.H):
				x = self.X[self.v[self.z] == h]
				squares[h] = self.prior_sigma + (x ** 2).sum(axis=0)
		## Calculate the determinants sequentially for the prior
		Delta0_det = np.zeros(self.m+1)
		for i in range(self.m+1):
			Delta0_det[i] = slogdet(self.Delta0[:i,:i])[1]
		## Calculate the marginal likelihoods sequentially
		self.mlik = np.zeros(self.m+1)
		for d in range(self.m+1):
		## Calculate the marginal likelihood for right hand side of the Gaussian
			if d != self.m:
				for k in range(self.sigmank.shape[0]):
						self.mlik[d] += np.sum(.5 * self.lambda0 * log(self.prior_sigma[d:]) - .5 * self.lambdank[k] * log(squares[k,d:]))
						self.mlik[d] += (self.m - d) * (gammaln(.5 * self.lambdank[k]) - gammaln(.5 * self.lambda0))
						if self.equal_var:
							self.mlik[d] += -.5 * self.vk[k] * (self.m - d) * log(np.pi)
						else:
							self.mlik[d] += -.5 * self.nk[k] * (self.m - d) * log(np.pi)
			else:
				self.mlik[d] = 0
			## Calculate the marginal likelihood for the community allocations
			if d != 0: 
				for k in range(self.K):
					self.mlik[d] += -.5 * self.nk[k] * d * log(np.pi) + .5 * d * (log(self.kappa0) - log(self.nk[k] + self.kappa0)) + \
						.5 * (self.nu0 + d - 1) * Delta0_det[d] - .5 * (self.nu0 + self.nk[k] + d - 1) * Deltank_det[k,d]
					self.mlik[d] += np.sum([(gammaln(.5 * (self.nu0 + self.nk[k] + d - i)) - gammaln(.5 * (self.nu0 + d - i))) for i in range(1,d+1)])

	######################################################################
	### 2a. Resample the second order allocations using Gibbs sampling ###
	######################################################################
	def gibbs_second_order(self):
		## Stop if equal_var is set to false
		if not self.equal_var:
			raise ValueError('equal_var is set to false')
		## Update the second order allocations in random order
		for j in np.random.permutation(range(self.K)):
			vold = self.v[j]
			## Update parameters of the distribution
			positions = self.X[self.z == j,self.d:]
			self.vk[vold] -= 1.0
			self.lambdank[vold] -= self.nk[j]
			self.sigmank[vold] -= np.sum(positions ** 2,axis=0)
			## Calculate the probabilities sequentially 
			lambdank_resamp = np.copy(self.lambdank)
			sigmank_resamp = np.copy(self.sigmank)
			prob_v = np.zeros(self.H)
			for i in np.random.permutation(range(positions.shape[0])):
				pos = positions[i]
				prob_v += np.array([np.sum(t.logpdf(pos,df=lambdank_resamp[h],loc=0,scale=sqrt(sigmank_resamp[h]/lambdank_resamp[h]))) for h in range(self.H)])
				lambdank_resamp += 1
				sigmank_resamp += (pos ** 2)
			## Calculate last component of the probability of allocation
			prob_v += log(self.vk + self.beta / self.H) - log(self.K - 1 + self.beta)
			prob_v = exp(prob_v-max(prob_v)) / np.sum(exp(prob_v-np.max(prob_v)))
			## Raise error if nan probabilities are computed or negative values appear in lambda or sigma
			if (lambdank_resamp <= 0.0).any():
				print lambdank_resamp
				raise ValueError('Problem with lambda')
			if (sigmank_resamp <= 0.0).any():
				raise ValueError('Problem with sigma')
			if np.isnan(prob_v).any():
				raise ValueError("Error in the allocation probabilities.")
			## Sample the new community allocation
			vnew = int(np.random.choice(range(self.H),p=prob_v))
			self.v[j] = vnew
			## Update the Student's t parameters accordingly
			self.lambdank[vnew] += self.nk[j]
			self.sigmank[vnew] += np.sum(positions ** 2, axis=0)
			self.vk[vnew] += 1.0
		if (self.sigmank < 0).any() or (self.lambdank < 0).any():
			return ValueError('Error with sigma and/or lambda')
		vv = Counter(self.v)
		vvv = np.array([vv[key] for key in range(self.H)])
		if (vvv != self.vk).any():
			raise ValueError('Error with vs')
		return None ## the global variables are updated within the function, no need to return anything

	###############################################################
	### 2b. Propose to split (or merge) two second order groups ###
	###############################################################
	def split_merge_second_order(self,verbose=False):
		## Stop if equal_var is set to false
		if not self.equal_var:
			raise ValueError('equal_var is set to false')
		## Randomly choose two indices
		i,j = np.random.choice(self.K,size=2,replace=False)
		## Propose a split or merge move according to the sampled values
		if self.v[i] == self.v[j]:
			split = True
			vsplit = self.v[i]
			## Obtain the indices that must be re-allocated
			S = np.delete(range(self.K),[i,j])[np.delete(self.v == vsplit,[i,j])]
			v_prop = np.copy(self.v)
			## Move j to a new cluster
			v_prop[j] = self.H
		else:
			split = False
			## Choose to merge to the cluster with minimum index (vmerge) and remove the cluster with maximum index (vlost)
			vmerge = min([self.v[i],self.v[j]])
			vj = self.v[j]
			vlost = max([self.v[i],self.v[j]])
			v_prop = np.copy(self.v)
			v_prop[self.v == vlost] = vmerge
			if vlost != (self.H-1):
				for k in range(vlost,self.H-1):
					v_prop[self.v == k+1] = k
			## Set of observations for the imaginary split (Dahl, 2003)
			S = np.delete(range(self.K),[i,j])[np.delete((self.v == vmerge) + (self.v == vlost),[i,j])]
		## Initialise the proposal ratio
		prop_ratio = 0
		## Construct vectors for sequential posterior predictives
		vk_rest = np.ones(2,int)
		lambda_rest = self.lambda0 + np.ones(2)
		sigma_rest = np.zeros((2,self.m-self.d))
		if split:
			sigma_rest[0] = self.prior_sigma[self.d:] + np.sum(self.X[v_prop[self.z] == vsplit,self.d:] ** 2, axis=0)
			sigma_rest[1] = self.prior_sigma[self.d:] + np.sum(self.X[v_prop[self.z] == self.H,self.d:] ** 2, axis=0)
		else:
			sigma_rest[0] = self.prior_sigma[self.d:] + np.sum(self.X[self.v[self.z] == vmerge,self.d:] ** 2, axis=0)
			sigma_rest[1] = self.prior_sigma[self.d:] + np.sum(self.X[self.v[self.z] == vlost,self.d:] ** 2, axis=0)
		## Randomly permute the indices in S and calculate the sequential allocations 
		for h in np.random.permutation(S):
			## Calculate the predictive probability
			positions = self.X[self.z == h,self.d:]
			## Calculate the new second order allocation of the first cluster after splitting
			lambda_left = np.copy(lambda_rest)
			sigma_left = np.copy(sigma_rest)
			prob_v_left = np.zeros(2)
			if positions.shape[0] != 0:
				for pos in np.random.permutation(positions):
					prob_v_left += np.array([np.sum(t.logpdf(pos,df=lambda_left[q],loc=0,scale=sqrt(sigma_left[q]/lambda_left[q]))) for q in range(2)])
					lambda_left += 1.0
					for q in range(2): sigma_left[q] += (pos ** 2)
			## Resample the second order cluster allocation
			pred_prob = exp(prob_v_left-max(prob_v_left)) / np.sum(exp(prob_v_left-np.max(prob_v_left)))
			## Calculate second order cluster allocation (restricted posterior predictive)
			if split:
				## Sample the new value
				vnew = np.random.choice(2,p=pred_prob)
				## Update proposal ratio
				prop_ratio += log(pred_prob[vnew])
				## Update proposed h
				v_prop[h] = [vsplit,self.H][vnew]
			else:
				## Determine the new value deterministically
				vnew = int(self.v[h] == vj)
				## Update proposal ratio in the imaginary split
				prop_ratio += log(pred_prob[vnew])
				if pred_prob[vnew] == 0.0:
					warnings.warn('Imaginary split yields impossible outcome: merge proposal automatically rejected')
			## Update parameters 
			vk_rest[vnew] += 1.0
			lambda_rest[vnew] += positions.shape[0]
			sigma_rest[vnew] += np.sum(positions ** 2, axis=0)
		## Proposed values of lambdank and sigmank
		H_prop = self.H + (1 if split else -1)
		vk_prop = np.zeros(H_prop)
		lambdank_prop = np.zeros(H_prop)
		sigmank_prop = np.zeros((H_prop,self.m-self.d))
		#sigmank_prop = np.copy(self.sigmank)
		for h in range(H_prop):
			## Second order cluster counts
			vk_prop[h] = np.sum(v_prop == h)
			lambdank_prop[h] = self.lambda0 + np.sum(self.nk[v_prop == h])
			sigmank_prop[h] = self.prior_sigma[self.d:] + np.sum(self.X[v_prop[self.z] == h, self.d:]**2, axis=0)
		#if split:
		#	sigmank_prop[vsplit] = sigma_rest[0]
		#	sigmank_prop = np.vstack((sigmank_prop,sigma_rest[1]))
		#else:
		#	sigmank_prop[vmerge] += sigmank_prop[vlost] - self.prior_sigma[self.d:]
		#	sigmank_prop = np.delete(sigmank_prop,vlost,axis=0)
		## Calculate the acceptance probability
		accept_ratio = 0.0
		accept_ratio += np.sum([(self.m - self.d) * (gammaln(.5 * lambdank_prop[h]) - gammaln(.5 * self.lambda0)) + \
							np.sum(.5*self.lambda0*log(self.prior_sigma[self.d:]) - .5*lambdank_prop[h]*log(sigmank_prop[h])) for h in range(H_prop)])
		accept_ratio -= np.sum([(self.m - self.d) * (gammaln(.5 * self.lambdank[h]) - gammaln(.5 * self.lambda0)) + \
							np.sum(.5*self.lambda0*log(self.prior_sigma[self.d:]) - .5*self.lambdank[h]*log(self.sigmank[h])) for h in range(self.H)])
		accept_ratio += np.sum(gammaln(vk_prop + self.beta / self.H)) - np.sum(gammaln(self.vk + self.beta / self.H)) + gammaln(self.K + self.beta) - \
							gammaln(self.K + 1 + self.beta)
		# Prior on H and q function
		accept_ratio += (1 if split else -1)*(log(1.0 - self.csi) - prop_ratio)
		## Accept or reject the proposal
		accept = (-np.random.exponential(1) < accept_ratio)
		if accept:
			## Update the stored values
			self.v = v_prop
			self.vk = vk_prop
			self.lambdank = lambdank_prop
			self.sigmank = sigmank_prop
			## Update H 
			self.H = H_prop
		if verbose:
			print 'Proposal: ' + ['MERGE 2nd order','SPLIT 2nd order'][split] + '\t' + 'Accepted: ' + str(accept)
		if (self.sigmank < 0).any() or (self.lambdank < 0).any():
			return ValueError('Error with sigma and/or lambda')
		vv = Counter(self.v)
		vvv = np.array([vv[key] for key in range(self.H)])
		if (vvv != self.vk).any():
			raise ValueError('Error with vs')
		return None

	######################################################################
	### 2c. Propose to add (or delete) an empty second order component ###
	######################################################################
	def propose_empty_second_order(self,verbose=False):
		## Stop if equal_var is set to false
		if not self.equal_var:
			raise ValueError('equal_var is set to false')
		## Propose to add or remove an empty cluster
		if self.H == 1:
			H_prop = 2
		elif self.H == self.n:
			H_prop = self.H - 1
		else:
			H_prop = np.random.choice([self.H-1,self.H+1])
		## Assign values to the variable remove
		if H_prop < self.H:
			remove = True
		else:
			remove = False
		## If there are no empty clusters and K_prop = K-1, reject the proposal
		if not (self.vk == 0).any() and H_prop < self.H:
			if verbose:
				print 'Proposal: ' + 'REMOVE 2nd order' + '\t' + 'Accepted: ' + 'False'
			return None
		## Propose a new vector of cluster allocations
		if remove:
			## Delete empty cluster with largest index (or sample at random)
			ind_delete = np.random.choice(np.where(self.vk == 0)[0]) ##[-1]
			vk_prop = np.delete(self.vk,ind_delete)
		else:
			## Add an empty second order cluster
			vk_prop = np.append(self.vk,0)
		## Common term for the acceptance probability
		accept_ratio = self.H*gammaln(float(self.beta) / self.H) - H_prop * gammaln(float(self.beta) / H_prop) + \
								np.sum(gammaln(vk_prop + float(self.beta) / H_prop)) - np.sum(gammaln(self.vk + float(self.alpha) / self.H)) + \
								(H_prop - self.H) * log(1 - self.csi) * log(.5) * int(self.H == 1) - log(.5) * int(self.H == self.n)
		## Accept or reject the proposal
		accept = (-np.random.exponential(1) < accept_ratio)
		## Scale all the values if an empty cluster is added
		if accept:
			if H_prop > self.H:
				## v does not change
				self.vk = vk_prop
				self.lambdank = np.append(self.lambdank,self.lambda0)
				self.sigmank = np.vstack((self.sigmank,self.prior_sigma[self.d:] * np.ones((1,self.m-self.d))))
			else:
				self.vk = vk_prop
				self.lambdank = np.delete(self.lambdank,ind_delete)
				self.sigmank = np.delete(self.sigmank,ind_delete,axis=0)
				if ind_delete != H_prop:
					for h in range(ind_delete,H_prop):
						self.v[self.v == h+1] = h
			self.H = H_prop
		if verbose:
			print 'Proposal: ' + ['ADD 2nd order','REMOVE 2nd order'][remove] + '\t' + 'Accepted: ' + str(accept)
		if (self.sigmank < 0).any() or (self.lambdank < 0).any():
			return ValueError('Error with sigma and/or lambda')
		vv = Counter(self.v)
		vvv = np.array([vv[key] for key in range(self.H)])
		if (vvv != self.vk).any():
			raise ValueError('Error with vs')
		return None

