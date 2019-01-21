#!/usr/bin/env python
import sys
import argparse
from scipy.stats import t
from scipy.special import gammaln
import numpy as np
from numpy import pi,log,exp,sqrt
from numpy.linalg import svd,eigh,slogdet,inv
import mvt
import warnings
from collections import Counter
from sklearn.cluster import KMeans

#######################################################################################################################
### A Bayesian model for estimation of the optimal latent dimension of network embeddings of stochastic blockmodels ###
#######################################################################################################################

class mcmc_sbm:
	
	## Initialise the class from the adjacency matrix A (or biadjacency matrix)
	def __init__(self,A,m=100,grdpg=True):
		## Rectangular or square adjacency matrix
		if A.shape[0] != A.shape[1]:
			self.bipartite = True
			self.n = {}
			self.n['s'] = A.shape[0]
			self.n['r'] = A.shape[1]
		else:
			self.bipartite = False
			self.directed = True
			self.n = A.shape[0]
		## Directed or undirected graph
		if not self.bipartite and (A == A.T).all():
			self.directed = False
		else:
			self.directed = True
		## Intialise m (note that cannot have different values of m, otherwise meaning of low rank approximation of A is lost)
		if self.bipartite:
			self.m = int(m) if int(m) < np.min(self.n.values()) else np.min([self.n.values()])
		else:
			self.m = int(m) if int(m) < self.n else self.n
		## Calculate the embedding
		if self.directed:
			## SVD decomposition of A
			u,s,v = svd(A)
			self.X = {}
			self.X['s'] = np.dot(u[:,:self.m],np.diag(np.sqrt(s[:self.m])))
			self.X['r'] = np.dot(v.T[:,:self.m],np.diag(np.sqrt(s[:self.m])))
		else:
			## Spectral decomposition of A
			w,v = eigh(A)
			## If GRDPG, use the top m eigenvalues in magnitude, otherwise use the standard RDPG embedding (top m eigenvalues)
			w_mag = (-np.abs(w)).argsort() if grdpg else (-w).argsort()
			self.X = np.dot(v[:,w_mag[:m]],np.diag(np.sqrt(abs(w[w_mag[:m]]))))
		## Calculate an array of outer products for the entire node set
		if self.directed:
			self.full_outer_x = {}
			for key in ['s','r']:
				self.full_outer_x[key] = np.zeros((self.n[key] if self.bipartite else self.n,self.m,self.m))
				for x in range(self.n[key] if self.bipartite else self.n):
					self.full_outer_x[key][x] = np.outer(self.X[key][x],self.X[key][x])
		else:
			self.full_outer_x = np.zeros((self.n,self.m,self.m))
			for x in range(self.n):
				self.full_outer_x[x] = np.outer(self.X[x],self.X[x])

	## Initialise the optimal dimension d
	def init_dim(self,d,delta=0.1):
		self.d = d 
		self.delta = delta
	
	## Initialise the cluster structure (assuming that allocations in z range from 0 to K-1)
	def init_cluster(self,z,K=0,alpha=1.0,omega=0.1):
		if self.bipartite:
			raise ValueError('The graph is bipartite. Use init_cocluster() for initialisation.')
		## Set coclust to False
		self.coclust = False
		## Initialise z 
		self.z = np.copy(z)
		## Initialise number of clusters
		if K != 0 and K >= (np.max(z)+1): ##len(np.unique(z)):
			self.K = K
		else:
			self.K = (np.max(z)+1) ##len(np.unique(z))
		## Initialise cluster counts and parameters of the prior distributions
		self.nk = np.array([np.sum(self.z == k) for k in range(self.K)])
		self.alpha = alpha
		self.omega = omega

	## Initialise the cluster structure for bipartite graphs and coclustering
	def init_cocluster(self,zs,zr,Ks=0,Kr=0,alphas=1.0,alphar=1.0,omega=0.1):
		if self.directed or self.bipartite:
			self.coclust = True
		else:
			raise ValueError('The graph is undirected. Use init_cluster() for initialisation.')
		## If the graph is directed (but not bipartite), set ns and nr (for simplicity)
		if self.directed and not self.bipartite:
			n = {}
			n['s'] = self.n
			n['r'] = self.n
			self.n = n
		if len(zs) != self.n['s'] or len(zr) != self.n['r']:
			raise ValueError('zs and zr must have lengths ns and nr.')
		## Initialise the cluster allocations
		self.z = {}
		self.z['s'] = np.copy(zs)
		self.z['r'] = np.copy(zr)
		## Initialise Ks and Kr
		self.K = {}
		if Ks != 0 and Ks >= (np.max(zs)+1): ## len(np.unique(zs)):
			self.K['s'] = Ks
		else:
			self.K['s'] = (np.max(zs)+1) ##len(np.unique(zs))
		if Kr != 0 and Kr >= (np.max(zr)+1): ##len(np.unique(zr)):
			self.K['r'] = Kr
		else:
			self.K['r'] = (np.max(zr)+1) ## len(np.unique(zr))
		## Initialise the cluster counts
		self.nk = {}
		for key in ['s','r']: 
			self.nk[key] = np.array([np.sum(self.z[key] == k) for k in range(self.K[key])])
		## Initalise the remaining parameters
		self.alpha = {}
		self.alpha['s'] = alphas
		self.alpha['r'] = alphar
		self.omega = omega
	
	## Add the prior parameters for the Gaussian components of the matrix
	def prior_gauss_left_undirected(self,mean0,Delta0,kappa0=1.0,nu0=1.0):
		if self.directed:
			raise ValueError('The graph is directed: use prior_gauss_left_directed().')
		if mean0.shape[0] != self.m or Delta0.shape[0] != Delta0.shape[0] or Delta0.shape[0] != self.m:
			raise ValueError('mean0 and Delta0 must have dimension m and mxm.')
		## Intialise the parameters of the NIW distribution
		self.mean0 = np.copy(mean0)
		self.kappa0 = kappa0
		self.nu0 = nu0
		self.Delta0 = np.copy(Delta0)
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
			sign_det, self.Delta_k_det[k] = slogdet(self.Delta_k[k])
			if sign_det <= 0.0: 
				raise ValueError("Covariance matrix is negative definite.")		
		## Calculate the determinants sequentially for the prior
		self.Delta0_det = np.zeros(self.m+1)
		for i in range(self.m+1):
			sign_det, self.Delta0_det[i] = slogdet(self.Delta0[:i,:i])
			if sign_det <= 0.0: 
				raise ValueError("Covariance matrix is negative definite.")		

	## Add the prior parameters for the Gaussian components of the matrix
	def prior_gauss_left_directed(self,mean0s,mean0r,Delta0s,Delta0r,kappa0s=1.0,kappa0r=1.0,nu0s=1.0,nu0r=1.0):
		if not self.directed:
			raise ValueError('The graph is undirected: use prior_gauss_left_undirected().')
		if mean0s.shape[0] != self.m or Delta0s.shape[0] != Delta0s.shape[0] or Delta0s.shape[0] != self.m:
			raise ValueError('mean0 and Delta0 must have dimension m and mxm.')
		if mean0r.shape[0] != self.m or Delta0r.shape[0] != Delta0r.shape[0] or Delta0r.shape[0] != self.m:
			raise ValueError('mean0 and Delta0 must have dimension m and mxm.')
		## Intialise the parameters of the NIW distribution
		self.mean0 = {}
		self.mean0['s'] = np.copy(mean0s)
		self.mean0['r'] = np.copy(mean0r)
		self.kappa0 = {}
		self.kappa0['s'] = kappa0s
		self.kappa0['r'] = kappa0r
		self.nu0 = {}
		self.nu0['s'] = nu0s
		self.nu0['r'] = nu0r
		self.Delta0 = {}
		self.Delta0['s'] = np.copy(Delta0s)
		self.Delta0['r'] = np.copy(Delta0r)
		## Compute kappank and nunk
		self.kappank = {}
		self.kappank['s'] = kappa0s + (self.nk['s'] if self.coclust else self.nk)
		self.kappank['r'] = kappa0r + (self.nk['r'] if self.coclust else self.nk)
		self.nunk = {}
		self.nunk['s'] = nu0s + (self.nk['s'] if self.coclust else self.nk)
		self.nunk['r'] = nu0r + (self.nk['r'] if self.coclust else self.nk)
		## Calculate the prior sum
		self.prior_sum = {}
		self.prior_sum['s'] = kappa0s * mean0s
		self.prior_sum['r'] = kappa0r * mean0r
		## Prior outer product, scaled by kappa0
		self.prior_outer = {}
		self.prior_outer['s'] = kappa0s * np.outer(mean0s,mean0s)
		self.prior_outer['r'] = kappa0r * np.outer(mean0r,mean0r)
		## Initialise the sums
		self.sum_x = {}; self.mean_k = {}; self.squared_sum_x = {}
		for key in ['s','r']:
			self.sum_x[key] = np.zeros((self.K if not self.coclust else self.K[key],self.d))
			self.mean_k[key] = np.zeros((self.K if not self.coclust else self.K[key],self.d))
			self.squared_sum_x[key] = {}
			for k in range(self.K if not self.coclust else self.K[key]):
				x = self.X[key][(self.z if not self.coclust else self.z[key]) == k,:self.d]
				self.sum_x[key][k] = x.sum(axis=0)
				self.mean_k[key][k] = (self.sum_x[key][k] + self.prior_sum[key][:self.d]) / self.kappank[key][k]
				self.squared_sum_x[key][k] = np.dot(x.T,x)
		## Initialise the Deltas 
		self.Delta_k = {}; self.Delta_k_inv = {}; self.Delta_k_det = {}
		for key in ['s','r']:
			self.Delta_k[key] = {}; self.Delta_k_inv[key] = {}
			self.Delta_k_det[key] = np.zeros(self.K if not self.coclust else self.K[key])
			for k in range(self.K if not self.coclust else self.K[key]):
				self.Delta_k[key][k] = self.Delta0[key][:self.d,:self.d] + self.squared_sum_x[key][k] + self.prior_outer[key][:self.d,:self.d] - \
					self.kappank[key][k] * np.outer(self.mean_k[key][k],self.mean_k[key][k])
				self.Delta_k_inv[key][k] = self.kappank[key][k] / (self.kappank[key][k] + 1.0) * inv(self.Delta_k[key][k])
				self.Delta_k_det[key][k] = slogdet(self.Delta_k[key][k])[1]
		## Calculate the determinants sequentially for the prior
		self.Delta0_det = {}
		for key in ['s','r']:
			self.Delta0_det[key] = np.zeros(self.m+1)
			for i in range(self.m+1):
				self.Delta0_det[key][i] = slogdet(self.Delta0[key][:i,:i])[1]

	def prior_gauss_right_undirected(self,sigma0,lambda0=1.0):
		if self.directed:
			raise ValueError('The graph is directed: use prior_gauss_right_directed().')
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

	def prior_gauss_right_directed(self,sigma0s,sigma0r,lambda0s=1.0,lambda0r=1.0):
		if not self.directed:
			raise ValueError('The graph is undirected: use prior_gauss_right_undirected().')
		## sigma0 is a n-dimensional vector of variances
		self.sigma0 = {}; self.sigma0['s'] = sigma0s; self.sigma0['r'] = sigma0r
		self.lambda0 = {}; self.lambda0['s'] = lambda0s; self.lambda0['r'] = lambda0r 
		## Calculate the updated values
		self.prior_sigma = {}
		for key in ['s','r']:
			self.prior_sigma[key] = self.lambda0[key] * self.sigma0[key]
		## Initialise lambdas
		self.lambdank = {}
		self.lambdank['s'] = lambda0s + (self.nk['s'] if self.coclust else self.nk)
		self.lambdank['r'] = lambda0r + (self.nk['r'] if self.coclust else self.nk)
		### Initialise sigmas
		self.sigmank = {}
		for key in ['s','r']:
			self.sigmank[key] = np.zeros((self.K[key] if self.coclust else self.K,self.m-self.d)) 
			for k in range(self.K if not self.coclust else self.K[key]):
				self.sigmank[key][k] = self.prior_sigma[key][self.d:] + \
					np.sum(self.X[key][(self.z[key] if self.coclust else self.z) == k,self.d:] ** 2,axis=0)
		## Initialise equal_var to false 
		if not hasattr(self,'equal_var'):
			self.equal_var = False

    ### Must be initialised AFTER prior_gauss_right (directed or undirected)
	def init_group_variance_clust(self,v,H=0,beta=1.0): ## add e.g. csi=0.1 for geometric prior on H
		if self.coclust:
			raise ValueError('For co-clustering, use init_group_variance_coclust().')
		self.v = np.copy(v)
		if len(v) != self.K:
			raise ValueError('v must have dimension K')
		if H != 0 and H >= len(np.unique(v)):
			self.H = H
		else:
			self.H = len(np.unique(v))
		if H > self.K:
			raise ValueError('H must be less than or equal to K')
		## Initialise second order cluster allocation counts
		self.vk = np.zeros(self.H)
		for h in range(self.H):
			self.vk[h] = np.sum(self.v == h)
		## Initialise the other parameters
		self.beta = beta
		## self.csi = csi
		self.equal_var = True
		lambdank_v = {} if self.directed else np.zeros(self.H)
		sigmank_v = {} if self.directed else np.zeros((self.H,self.m-self.d))
		if self.directed:
			for key in ['s','r']:
				lambdank_v[key] = np.zeros(self.H)
				sigmank_v[key] = np.zeros((self.H,self.m-self.d))
				for h in range(self.H):
					lambdank_v[key][h] = self.lambda0[key] + np.sum(self.nk[self.v == h])
					sigmank_v[key][h] = self.prior_sigma[key][self.d:] + np.sum(self.sigmank[key][self.v == h] - self.prior_sigma[key][self.d:], axis=0)
		else:
			for h in range(self.H):
				lambdank_v[h] = self.lambda0 + np.sum(self.nk[self.v == h])
				sigmank_v[h] = self.prior_sigma[self.d:] + np.sum(self.sigmank[self.v == h] - self.prior_sigma[self.d:], axis=0)
		self.lambdank = lambdank_v
		self.sigmank = sigmank_v

	def init_group_variance_coclust(self,vs,vr,Hs=0,Hr=0,betas=1.0,betar=1.0): ## add e.g. csi=0.1 for geometric prior on H
		if not self.coclust:
			raise ValueError('For standard clustering, use init_group_variance_clust().')
		self.v = {}; self.v['s'] = np.copy(vs); self.v['r'] = np.copy(vr)
		if len(vs) != self.K['s'] or len(vr) != self.K['r']:
			raise ValueError('v must have dimension K')
		self.H = {}
		if Hs != 0 and Hs >= len(np.unique(vs)):
			self.H['s'] = int(Hs)
		else:
			self.H['s'] = len(np.unique(vs))
		if Hr != 0 and Hr >= len(np.unique(vr)):
			self.H['r'] = int(Hr)
		else:
			self.H['r'] = len(np.unique(vr))
		if Hs > self.K['s'] or Hr > self.K['r']:
			raise ValueError('H must be less than or equal to K')
		## Initialise second order cluster allocation counts
		self.vk = {}
		for key in ['s','r']: 
			self.vk[key] = np.zeros(self.H[key])
			for h in range(self.H[key]):
				self.vk[key][h] = np.sum(self.v[key] == h)
		## Initialise the other parameters
		self.beta = {}; self.beta['s'] = betas; self.beta['r'] = betar
		## self.csi = csi; 
		self.equal_var = True
		lambdank_v = {}; sigmank_v = {} 
		for key in ['s','r']:
			lambdank_v[key] = np.zeros(self.H[key])
			sigmank_v[key] = np.zeros((self.H[key],self.m-self.d))
			for h in range(self.H[key]):
				lambdank_v[key][h] = self.lambda0[key] + np.sum(self.nk[key][self.v[key] == h])
				sigmank_v[key][h] = self.prior_sigma[key][self.d:] + np.sum(self.sigmank[key][self.v[key] == h] - self.prior_sigma[key][self.d:], axis=0)
		self.lambdank = lambdank_v
		self.sigmank = sigmank_v

	########################################################
	### a. Resample the allocations using Gibbs sampling ###
	########################################################
	def gibbs_communities(self,l=50):
		## For coclustering: Gibbs sample at random sources or receivers
		if self.coclust:
			sr = np.random.choice(['s','r'])
		## Change the value of l when too large
		if l > (self.n[sr] if self.coclust else self.n):
			l = self.n[sr] if self.coclust else self.n
		## Update the latent allocations in randomised order
		## Loop over the indices
		for j in np.random.permutation(range(self.n[sr] if self.coclust else self.n))[:l]:
			zold = self.z[sr][j] if self.coclust else self.z[j] 
			## Update parameters of the distribution
			if self.coclust:
				position = self.X[sr][j,:self.d]
				position_right = self.X[sr][j,self.d:]
				out_position = np.outer(position,position)
				self.sum_x[sr][zold] -= position
				self.squared_sum_x[sr][zold] -= out_position
				self.nk[sr][zold] -= 1.0
				self.nunk[sr][zold] -= 1.0
				self.kappank[sr][zold] -= 1.0
				self.lambdank[sr][self.v[sr][zold] if self.equal_var else zold] -= 1.0		
				self.sigmank[sr][self.v[sr][zold] if self.equal_var else zold] -= (position_right ** 2)
				self.mean_k[sr][zold] = (self.prior_sum[sr][:self.d] + self.sum_x[sr][zold]) / self.kappank[sr][zold]
				## Update Delta (store the old values in case the allocation does not change)
				Delta_old = self.Delta_k[sr][zold]
				Delta_inv_old = self.Delta_k_inv[sr][zold]
				Delta_det_old = self.Delta_k_det[sr][zold]
				self.Delta_k[sr][zold] = self.Delta0[sr][:self.d,:self.d] + self.squared_sum_x[sr][zold] + self.prior_outer[sr][:self.d,:self.d] - \
					self.kappank[sr][zold] * np.outer(self.mean_k[sr][zold],self.mean_k[sr][zold])
				self.Delta_k_inv[sr][zold] = self.kappank[sr][zold] / (self.kappank[sr][zold] + 1.0) * inv(self.Delta_k[sr][zold])
				sign_det, self.Delta_k_det[sr][zold] = slogdet(self.Delta_k[sr][zold])
				## Raise error if the matrix is not positive definite
				if sign_det <= 0.0: 
					raise ValueError("Covariance matrix is negative definite.")		
			else:
				if self.directed:
					position = {}; position_right = {}; out_position = {}
					for key in ['s','r']: 
						position[key] = self.X[key][j,:self.d]
						position_right[key] = self.X[key][j,self.d:]
						out_position[key] = np.outer(position[key],position[key])
						self.sum_x[key][zold] -= position[key]
						self.squared_sum_x[key][zold] -= out_position[key]
				else:
					position = self.X[j,:self.d]
					position_right = self.X[j,self.d:]
					out_position = np.outer(position,position)
					self.sum_x[zold] -= position
					self.squared_sum_x[zold] -= out_position
				self.nk[zold] -= 1.0
				if self.directed:
					for key in ['s','r']:	
						self.nunk[key][zold] -= 1.0
						self.kappank[key][zold] -= 1.0
						self.lambdank[key][self.v[zold] if self.equal_var else zold] -= 1.0
				else:
					self.nunk[zold] -= 1.0
					self.kappank[zold] -= 1.0
					self.lambdank[self.v[zold] if self.equal_var else zold] -= 1.0
				if self.directed:
					Delta_old = {}; Delta_inv_old = {}; Delta_det_old = {};
					for key in ['s','r']:
						self.sigmank[key][self.v[zold] if self.equal_var else zold] -= (position_right[key] ** 2)
						self.mean_k[key][zold] = (self.prior_sum[key][:self.d] + self.sum_x[key][zold]) / self.kappank[key][zold]
						## Update Delta (store the old values in case the allocation does not change)
						Delta_old[key] = self.Delta_k[key][zold]
						Delta_inv_old[key] = self.Delta_k_inv[key][zold]
						Delta_det_old[key] = self.Delta_k_det[key][zold]
						self.Delta_k[key][zold] = self.Delta0[key][:self.d,:self.d] + self.squared_sum_x[key][zold] + self.prior_outer[key][:self.d,:self.d] - \
							self.kappank[key][zold] * np.outer(self.mean_k[key][zold],self.mean_k[key][zold])
						self.Delta_k_inv[key][zold] = self.kappank[key][zold] / (self.kappank[key][zold] + 1.0) * inv(self.Delta_k[key][zold])
						sign_det, self.Delta_k_det[key][zold] = slogdet(self.Delta_k[key][zold])
						## Raise error if the matrix is not positive definite
						if sign_det <= 0.0: 
							raise ValueError("Covariance matrix is negative definite.")
				else:
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
			if self.coclust:
				community_probs_left = np.array([mvt.dmvt_efficient(x=position,mu=self.mean_k[sr][i],Sigma_inv=self.Delta_k_inv[sr][i], \
					Sigma_logdet=self.d * log((self.kappank[sr][i] + 1.0) / (self.kappank[sr][i] * self.nunk[sr][i])) + \
					self.Delta_k_det[sr][i],nu=self.nunk[sr][i]) for i in range(self.K[sr])]).reshape(self.K[sr],)
				## Raise error if nan probabilities are computed
				if np.isnan(community_probs_left).any():
					raise ValueError("Error in the allocation probabilities. Check invertibility of the covariance matrices.")
			else:
				if self.directed:
					community_probs_left = {}
					for key in ['s','r']:
						community_probs_left[key] = np.array([mvt.dmvt_efficient(x=position[key],mu=self.mean_k[key][i],Sigma_inv=self.Delta_k_inv[key][i], \
							Sigma_logdet=self.d * log((self.kappank[key][i] + 1.0) / (self.kappank[key][i] * self.nunk[key][i])) + \
							self.Delta_k_det[key][i],nu=self.nunk[key][i]) for i in range(self.K)]).reshape(self.K,)
						## Raise error if nan probabilities are computed
						if np.isnan(community_probs_left[key]).any():
							raise ValueError("Error in the allocation probabilities. Check invertibility of the covariance matrices.")
				else:
					community_probs_left = np.array([mvt.dmvt_efficient(x=position,mu=self.mean_k[i],Sigma_inv=self.Delta_k_inv[i], \
						Sigma_logdet=self.d * log((self.kappank[i] + 1.0) / (self.kappank[i] * self.nunk[i])) + \
						self.Delta_k_det[i],nu=self.nunk[i]) for i in range(self.K)]).reshape(self.K,)
					## Raise error if nan probabilities are computed
					if np.isnan(community_probs_left).any():
						raise ValueError("Error in the allocation probabilities. Check invertibility of the covariance matrices.")
			## Calculate the probability of allocation for the left hand side of the matrix
			if self.d != self.m:
				if self.equal_var:
					if self.coclust:
						community_probs_right = np.array([np.sum(t.logpdf(position_right, df=self.lambdank[sr][self.v[sr][k]], loc=0, \
								scale=sqrt(self.sigmank[sr][self.v[sr][k]] / self.lambdank[sr][self.v[sr][k]]))) for k in range(self.K[sr])])
					else:
						if self.directed:
							community_probs_right = {}
							for key in ['s','r']:
								community_probs_right[key] = np.array([np.sum(t.logpdf(position_right[key], df=self.lambdank[key][self.v[k]], loc=0, \
									scale=sqrt(self.sigmank[key][self.v[k]] / self.lambdank[key][self.v[k]]))) for k in range(self.K)])
						else:
							community_probs_right = np.array([np.sum(t.logpdf(position_right, df=self.lambdank[self.v[k]], loc=0, \
								scale=sqrt(self.sigmank[self.v[k]] / self.lambdank[self.v[k]]))) for k in range(self.K)])
				else:
					if self.coclust:
						community_probs_right = np.array([np.sum(t.logpdf(position_right, df=self.lambdank[sr][k], loc=0, \
							scale=sqrt(self.sigmank[sr][k] / self.lambdank[sr][k]))) for k in range(self.K[sr])])
						if np.isnan(community_probs_right).any():
							print community_probs_right
							raise ValueError("Error in the allocation probabilities. Check variances of right hand side of the matrix.")
					else:
						if self.directed:
							community_probs_right = {}
							for key in ['s','r']:
								community_probs_right[key] = np.array([np.sum(t.logpdf(position_right[key], df=self.lambdank[key][k], loc=0, \
									scale=sqrt(self.sigmank[key][k] / self.lambdank[key][k]))) for k in range(self.K)])
								if np.isnan(community_probs_right[key]).any():
									print community_probs_right
									raise ValueError("Error in the allocation probabilities. Check variances of right hand side of the matrix.")
						else:
							community_probs_right = np.array([np.sum(t.logpdf(position_right, df=self.lambdank[k], loc=0, \
								scale=sqrt(self.sigmank[k] / self.lambdank[k]))) for k in range(self.K)])
							if np.isnan(community_probs_right).any():
								print community_probs_right
								raise ValueError("Error in the allocation probabilities. Check variances of right hand side of the matrix.")
			else:
				if self.coclust:
					community_probs_right = np.zeros(self.K[sr]) 
				else:
					if self.directed:
						community_probs_right = {}
						for key in ['s','r']:
							community_probs_right[key] = np.zeros(self.K) 
					else:
						community_probs_right = np.zeros(self.K) 
			#if (self.lambdank <= 0.0).any():
			#	raise ValueError('Problem with lambda')
			#if not self.directed and (self.sigmank <= 0.0).any():
			#	raise ValueError('Problem with sigma')
			## Calculate last component of the probability of allocation
			if self.coclust:
				community_probs_allo = log(self.nk[sr] + float(self.alpha[sr])/ self.K[sr])
			else:
				community_probs_allo = log(self.nk + float(self.alpha)/ self.K)
			## Compute the allocation probabilities
			if self.directed and not self.coclust:
				community_probs = exp(community_probs_left['s'] + community_probs_right['s'] + \
					community_probs_left['r'] + community_probs_right['r'] + community_probs_allo)
				community_probs /= sum(community_probs)
			else:
				community_probs = exp(community_probs_left + community_probs_right + community_probs_allo)
				community_probs /= sum(community_probs)
			## Sample the new community allocation
			znew = int(np.random.choice(self.K[sr] if self.coclust else self.K,p=community_probs))
			if self.coclust:
				self.z[sr][j] = znew
				## Update the Student's t parameters accordingly
				self.nk[sr][znew] += 1.0
				self.nunk[sr][znew] += 1.0
				self.kappank[sr][znew] += 1.0
				self.lambdank[sr][self.v[sr][znew] if self.equal_var else znew] += 1.0
				self.sum_x[sr][znew] += position
				self.squared_sum_x[sr][znew] += out_position
				self.sigmank[sr][self.v[sr][znew] if self.equal_var else znew] += (position_right ** 2)
				self.mean_k[sr][znew] = (self.prior_sum[sr][:self.d] + self.sum_x[sr][znew]) / self.kappank[sr][znew]
				## If znew == zold, do not recompute inverses and determinants but just copy the old values
				if znew != zold:
					self.Delta_k[sr][znew] = self.Delta0[sr][:self.d,:self.d] + self.squared_sum_x[sr][znew] + self.prior_outer[sr][:self.d,:self.d] - \
						self.kappank[sr][znew] * np.outer(self.mean_k[sr][znew],self.mean_k[sr][znew])
					self.Delta_k_inv[sr][znew] = self.kappank[sr][znew] / (self.kappank[sr][znew] + 1.0) * inv(self.Delta_k[sr][znew])
					sign_det, self.Delta_k_det[sr][znew] = slogdet(self.Delta_k[sr][znew])
					## Raise error if the matrix is not positive definite
					if sign_det <= 0.0: 
						raise ValueError("Covariance matrix is negative definite.")
				else:
					self.Delta_k[sr][znew] = Delta_old
					self.Delta_k_inv[sr][znew] = Delta_inv_old
					self.Delta_k_det[sr][znew] = Delta_det_old
			else:
				self.z[j] = znew
				## Update the Student's t parameters accordingly
				self.nk[znew] += 1.0
				if self.directed:
					for key in ['s','r']:
						self.nunk[key][znew] += 1.0
						self.kappank[key][znew] += 1.0
						self.lambdank[key][self.v[znew] if self.equal_var else znew] += 1.0
						self.sum_x[key][znew] += position[key]
						self.squared_sum_x[key][znew] += out_position[key]
						self.sigmank[key][self.v[znew] if self.equal_var else znew] += (position_right[key] ** 2)
						self.mean_k[key][znew] = (self.prior_sum[key][:self.d] + self.sum_x[key][znew]) / self.kappank[key][znew]
				else:
					self.nunk[znew] += 1.0
					self.kappank[znew] += 1.0
					self.lambdank[self.v[znew] if self.equal_var else znew] += 1.0
					self.sum_x[znew] += position
					self.squared_sum_x[znew] += out_position
					self.sigmank[self.v[znew] if self.equal_var else znew] += (position_right ** 2)
					self.mean_k[znew] = (self.prior_sum[:self.d] + self.sum_x[znew]) / self.kappank[znew]
				## If znew == zold, do not recompute inverses and determinants but just copy the old values
				if znew != zold:
					if self.directed:
						for key in ['s','r']:
							self.Delta_k[key][znew] = self.Delta0[key][:self.d,:self.d] + self.squared_sum_x[key][znew] + \
								self.prior_outer[key][:self.d,:self.d] - self.kappank[key][znew] * np.outer(self.mean_k[key][znew],self.mean_k[key][znew])
							self.Delta_k_inv[key][znew] = self.kappank[key][znew] / (self.kappank[key][znew] + 1.0) * inv(self.Delta_k[key][znew])
							sign_det, self.Delta_k_det[key][znew] = slogdet(self.Delta_k[key][znew])
							## Raise error if the matrix is not positive definite
							if sign_det <= 0.0: 
								raise ValueError("Covariance matrix is negative definite.")
					else:
						self.Delta_k[znew] = self.Delta0[:self.d,:self.d] + self.squared_sum_x[znew] + self.prior_outer[:self.d,:self.d] - self.kappank[znew] * \
							np.outer(self.mean_k[znew],self.mean_k[znew])
						self.Delta_k_inv[znew] = self.kappank[znew] / (self.kappank[znew] + 1.0) * inv(self.Delta_k[znew])
						sign_det, self.Delta_k_det[znew] = slogdet(self.Delta_k[znew])
						## Raise error if the matrix is not positive definite
						if sign_det <= 0.0: 
							raise ValueError("Covariance matrix is negative definite.")
				else:
					if self.directed:
						for key in ['s','r']:
							self.Delta_k[key][znew] = Delta_old[key]
							self.Delta_k_inv[key][znew] = Delta_inv_old[key]
							self.Delta_k_det[key][znew] = Delta_det_old[key]
					else:
						self.Delta_k[znew] = Delta_old
						self.Delta_k_inv[znew] = Delta_inv_old
						self.Delta_k_det[znew] = Delta_det_old
		#if (self.sigmank < 0).any() or (self.lambdank < 0).any():
		#	raise ValueError('Error with sigma and/or lambda')
		#vv = Counter(self.v)
		#vvv = np.array([vv[key] for key in range(self.H)])
		#if (vvv != self.vk).any():
		#	raise ValueError('Error with vs')
		return None ## the global variables are updated within the function, no need to return anything

	###################################################
	### b. Propose a change in the latent dimension ###
	###################################################
	def dimension_change(self,prop_step=1,delta_prop=0.2,verbose=False):
		## Propose a new value of d
		if prop_step == 1:
			if self.d == 1:
				d_prop = 2
			elif self.d == self.m:
				d_prop = self.m-1
			else:
				d_prop = np.random.choice([self.d-1, self.d+1])
		else:
			## Calculate the proposal 
			if delta_prop != 0.0:
				prop_probs_left = (1-delta_prop) ** (np.arange(len(range(np.max([1,self.d-prop_step]),self.d))))[::-1] * delta_prop
				prop_probs_right = (1-delta_prop) ** (np.arange(len(range(self.d+1,int(np.min([self.m+1,self.d+prop_step+1])))))) * delta_prop
				if len(prop_probs_right) == 0:
					probs = prop_probs_left / np.sum(prop_probs_left)
				elif len(prop_probs_left) == 0:
					probs = .5 * prop_probs_right / np.sum(prop_probs_right)
				else:
					probs = np.append(.5 * prop_probs_left / np.sum(prop_probs_left), .5 * prop_probs_right / np.sum(prop_probs_right))
				props = np.append(range(np.max([1,self.d-prop_step]),self.d),range(self.d+1,int(np.min([self.m+1,self.d+prop_step+1]))))
			else:
				props = np.append(range(np.max([1,self.d-prop_step]),self.d),range(self.d+1,int(np.min([self.m+1,self.d+prop_step+1]))))
				probs = 1.0/len(props) * np.ones(len(props))
			d_prop = int(np.random.choice(props,p=probs))
			q_prop_old = probs[np.where(props == d_prop)[0]][0]
			## Calculate the inverse proposal
			if delta_prop != 0.0:
				prop_probs_left = (1-delta_prop) ** (np.arange(len(range(np.max([1,d_prop-prop_step]),d_prop))))[::-1] * delta_prop
				prop_probs_right = (1-delta_prop) ** (np.arange(len(range(d_prop+1,int(np.min([self.m+1,d_prop+prop_step+1])))))) * delta_prop
				if len(prop_probs_right) == 0:
					probs = prop_probs_left / np.sum(prop_probs_left)
				elif len(prop_probs_left) == 0:
					probs = .5 * prop_probs_right / np.sum(prop_probs_right)
				else:
					probs = np.append(.5 * prop_probs_left / np.sum(prop_probs_left), .5 * prop_probs_right / np.sum(prop_probs_right))
				props = np.append(range(np.max([1,d_prop-prop_step]),d_prop),range(d_prop+1,int(np.min([self.m+1,d_prop+prop_step+1]))))
			else:
				props = np.append(range(np.max([1,d_prop-prop_step]),d_prop),range(d_prop+1,int(np.min([self.m+1,d_prop+prop_step+1]))))
				probs = 1.0/len(props) * np.ones(len(props))
			q_prop_new = probs[np.where(props == self.d)[0]][0]
		## Calculate likelihood for the current value of d 
		squared_sum_x_prop = {}; Delta_k_prop = {}
		Delta_k_det_prop = {} if self.directed else np.zeros(self.K)
		if self.directed:
			sum_x_prop = {}
			mean_k_prop = {}
			for key in ['s','r']:
				squared_sum_x_prop[key] = {}
				Delta_k_prop[key] = {}
				Delta_k_det_prop[key] = np.zeros(self.K[key] if self.coclust else self.K)
			sigmank_prop = {} 
		## Different proposed quantities according to the sampled value of d
		if d_prop > self.d:
			if self.directed:
				for key in ['s','r']:
					sum_x_prop[key] = np.hstack((self.sum_x[key], 
						np.array([np.sum(self.X[key][(self.z[key] if self.coclust else self.z) == i][:,range(self.d,d_prop)],axis=0) \
						for i in range(self.K[key] if self.coclust else self.K)], ndmin=2)))
					mean_k_prop[key] = np.divide((self.prior_sum[key][:d_prop] + sum_x_prop[key]).T, self.kappank[key]).T
					for i in range(self.K[key] if self.coclust else self.K):
						squared_sum_x_prop[key][i] = self.full_outer_x[key][(self.z[key] if self.coclust else self.z) == i,:d_prop,:d_prop].sum(axis=0)
						Delta_k_prop[key][i] = self.Delta0[key][:d_prop,:d_prop] + squared_sum_x_prop[key][i] + self.prior_outer[key][:d_prop,:d_prop] - \
							self.kappank[key][i] * np.outer(mean_k_prop[key][i],mean_k_prop[key][i])
						sign_det, Delta_k_det_prop[key][i] = slogdet(Delta_k_prop[key][i])
						if sign_det <= 0.0:
							raise ValueError("Covariance matrix for d_prop is not invertible. Check conditions.")
			else:
				sum_x_prop = np.hstack((self.sum_x, np.array([np.sum(self.X[self.z == i,range(self.d,d_prop)]) for i in range(self.K)], ndmin=2).T))
				mean_k_prop = np.divide((self.prior_sum[:d_prop] + sum_x_prop).T, self.kappank).T
				for i in range(self.K):
					squared_sum_x_prop[i] = self.full_outer_x[self.z == i,:d_prop,:d_prop].sum(axis=0)
					Delta_k_prop[i] = self.Delta0[:d_prop,:d_prop] + squared_sum_x_prop[i] + self.prior_outer[:d_prop,:d_prop] - \
						self.kappank[i] * np.outer(mean_k_prop[i],mean_k_prop[i])
					sign_det, Delta_k_det_prop[i] = slogdet(Delta_k_prop[i])
					if sign_det <= 0.0:
						raise ValueError("Covariance matrix for d_prop is not invertible. Check conditions.")
			## Initialise sigmank_prop
			if self.coclust:
				for key in ['s','r']:
					sigmank_prop[key] = np.zeros((self.H[key] if self.equal_var else self.K[key],self.m - d_prop))
					for k in range(sigmank_prop[key].shape[0]):
						sigmank_prop[key][k] = self.sigmank[key][k,(d_prop-self.d):]
			else:
				if self.directed:
					for key in ['s','r']:
						sigmank_prop[key] = np.zeros((self.H if self.equal_var else self.K,self.m - d_prop))
						for k in range(sigmank_prop[key].shape[0]):
							sigmank_prop[key][k] = self.sigmank[key][k,(d_prop-self.d):]
				else:
					sigmank_prop = np.zeros((self.H if self.equal_var else self.K,self.m - d_prop))
					for k in range(sigmank_prop.shape[0]):
						sigmank_prop[k] = self.sigmank[k,(d_prop-self.d):]
		else:
			if self.directed:
				for key in ['s','r']:
					sum_x_prop[key] = self.sum_x[key][:,:(d_prop-self.d)]
					mean_k_prop[key] = self.mean_k[key][:,:(d_prop-self.d)]
					for i in range(self.K[key] if self.coclust else self.K):
						squared_sum_x_prop[key][i] = self.squared_sum_x[key][i][:(d_prop-self.d),:(d_prop-self.d)]
						Delta_k_prop[key][i] = self.Delta0[key][:d_prop,:d_prop] + squared_sum_x_prop[key][i] + self.prior_outer[key][:d_prop,:d_prop] - \
							self.kappank[key][i] * np.outer(mean_k_prop[key][i],mean_k_prop[key][i])
						Delta_k_det_prop[key][i] = slogdet(Delta_k_prop[key][i])[1]
			else:
				sum_x_prop = self.sum_x[:,:(d_prop-self.d)]
				mean_k_prop = self.mean_k[:,:(d_prop-self.d)]
				for i in range(self.K):
					squared_sum_x_prop[i] = self.squared_sum_x[i][:(d_prop-self.d),:(d_prop-self.d)]
					Delta_k_prop[i] = self.Delta0[:d_prop,:d_prop] + squared_sum_x_prop[i] + self.prior_outer[:d_prop,:d_prop] - \
						self.kappank[i] * np.outer(mean_k_prop[i],mean_k_prop[i])
					Delta_k_det_prop[i] = slogdet(Delta_k_prop[i])[1]
			## Pay attention to the index to add
			if self.equal_var:
				if self.directed:
					for key in ['s','r']:
						sigmank_prop[key] = np.zeros((self.H[key] if self.coclust else self.H,self.m - d_prop))
						for h in range(self.H[key] if self.coclust else self.H):
							if prop_step == 1 or (self.d-d_prop) == 1:
								sigmank_prop[key][h] = np.insert(self.sigmank[key][h], 0, self.prior_sigma[key][d_prop] + \
									np.sum(self.X[key][(self.v[key][self.z[key]] if self.coclust else self.v[self.z]) == h,d_prop] ** 2))
							else:
								sigmank_prop[key][h] = np.append(self.prior_sigma[key][np.arange(d_prop,self.d)] + \
									np.sum(self.X[key][(self.v[key][self.z[key]] if self.coclust else self.v[self.z]) == h][:,np.arange(d_prop,self.d)] ** 2, 
									axis=0), self.sigmank[key][h])												
				else:
					sigmank_prop = np.zeros((self.H,self.m - d_prop))
					for h in range(self.H):
						if prop_step == 1 or (self.d-d_prop) == 1:
							sigmank_prop[h] = np.insert(self.sigmank[h], 0, self.prior_sigma[d_prop] + np.sum(self.X[self.v[self.z] == h,d_prop] ** 2))
						else:
							sigmank_prop[h] = np.append(self.prior_sigma[np.arange(d_prop,self.d)] + \
								np.sum(self.X[self.v[self.z] == h][:,np.arange(d_prop,self.d)] ** 2, axis=0), self.sigmank[h])
			else:
				if self.directed:
					for key in ['s','r']:
						sigmank_prop[key] = np.zeros((self.K[key] if self.coclust else self.K,self.m - d_prop))
						for k in range(self.K[key] if self.coclust else self.K):
							if prop_step == 1 or (self.d-d_prop) == 1:
								sigmank_prop[key][k] = np.insert(self.sigmank[key][k], 0, self.prior_sigma[key][d_prop] + \
									np.sum(self.X[key][(self.z[key] if self.coclust else self.z) == k,d_prop] ** 2))
							else:
								sigmank_prop[key][k] = np.append(self.prior_sigma[key][np.arange(d_prop,self.d)] + \
									np.sum(self.X[key][(self.z[key] if self.coclust else self.z) == k][:,np.arange(d_prop,self.d)] ** 2, axis=0),
									self.sigmank[key][k])
				else:
					sigmank_prop = np.zeros((self.K,self.m - d_prop))
					for k in range(self.K):
						if prop_step == 1 or (self.d-d_prop) == 1:
							sigmank_prop[k] = np.insert(self.sigmank[k], 0, self.prior_sigma[d_prop] + np.sum(self.X[self.z == k,d_prop] ** 2))
						else:
							sigmank_prop[k] = np.append(self.prior_sigma[np.arange(d_prop,self.d)] + \
								np.sum(self.X[self.z == k][:,np.arange(d_prop,self.d)] ** 2, axis=0),self.sigmank[k])
		# Calculate old likelihood
		old_lik = 0
		# Add the community specific components to the log-likelihood
		if self.d != 0:
			if self.directed:
				for key in ['s','r']:
					for k in range(self.K[key] if self.coclust else self.K):
						old_lik += .5 * self.d * (log(self.kappa0[key]) - log(self.kappank[key][k])) + \
							.5 * (self.nu0[key] + self.d - 1) * self.Delta0_det[key][self.d] - \
							.5 * (self.nunk[key][k] + self.d - 1) * self.Delta_k_det[key][k]
					for k in range(self.sigmank[key].shape[0]):	
						old_lik += (self.m - self.d) * (gammaln(.5 * self.lambdank[key][k]) - gammaln(.5 * self.lambda0[key]))
			else:
				for k in range(self.K):
					old_lik += .5 * self.d * (log(self.kappa0) - log(self.kappank[k])) + .5 * (self.nu0 + self.d - 1) * self.Delta0_det[self.d] - \
						.5 * (self.nunk[k] + self.d - 1) * self.Delta_k_det[k]
				for k in range(self.sigmank.shape[0]):	
					old_lik += (self.m - self.d) * (gammaln(.5 * self.lambdank[k]) - gammaln(.5 * self.lambda0))
		# Calculate new likelihood
		new_lik = 0
		# Add the community specific components to the log-likelihood
		if d_prop != 0: 
			if self.directed:
				for key in ['s','r']:
					for k in range(self.K[key] if self.coclust else self.K):
						new_lik += .5 * d_prop * (log(self.kappa0[key]) - log(self.kappank[key][k])) + \
							.5 * (self.nu0[key] + d_prop - 1) * self.Delta0_det[key][d_prop] - .5 * (self.nunk[key][k] + d_prop - 1) * Delta_k_det_prop[key][k]
					for k in range(self.sigmank[key].shape[0]):
						new_lik += (self.m - d_prop) * (gammaln(.5 * self.lambdank[key][k]) - gammaln(.5 * self.lambda0[key]))			
			else:
				for k in range(self.K):
					new_lik += .5 * d_prop * (log(self.kappa0) - log(self.kappank[k])) + .5 * (self.nu0 + d_prop - 1) * self.Delta0_det[d_prop] - \
						.5 * (self.nunk[k] + d_prop - 1) * Delta_k_det_prop[k]
				for k in range(self.sigmank.shape[0]):
					new_lik += (self.m - d_prop) * (gammaln(.5 * self.lambdank[k]) - gammaln(.5 * self.lambda0))
		## Add the remaining component of the likelihood ratio
		if d_prop > self.d:
			if self.directed:
				for key in ['s','r']:
					if prop_step == 1:
						new_lik += np.sum(gammaln(.5 * (self.nunk[key] + d_prop - 1)) - gammaln(.5 * (self.nu0[key] + d_prop - 1)))
						old_lik += np.sum(.5 * self.lambda0[key] * log(self.prior_sigma[key][self.d]) - .5 * self.lambdank[key] * log(self.sigmank[key][:,0]))
					else:
						for k in range(self.K[key] if self.coclust else self.K):
							new_lik += np.sum(gammaln(.5 * (self.nunk[key][k] + d_prop - 1 - np.arange(d_prop-self.d))) - \
								gammaln(.5 * (self.nu0[key] + d_prop - 1 - np.arange(d_prop-self.d))))
						for k in range(self.sigmank[key].shape[0]):
							old_lik += np.sum(.5 * self.lambda0[key] * log(self.prior_sigma[key][np.arange(self.d,d_prop)]) - \
								.5 * self.lambdank[key][k] * log(self.sigmank[key][k,np.arange(d_prop-self.d)]))
			else:
				if prop_step == 1:
					new_lik += np.sum(gammaln(.5 * (self.nunk + d_prop - 1)) - gammaln(.5 * (self.nu0 + d_prop - 1)))
					old_lik += np.sum(.5 * self.lambda0 * log(self.prior_sigma[self.d]) - .5 * self.lambdank * log(self.sigmank[:,0]))
				else:
					for k in range(self.K):
						new_lik += np.sum(gammaln(.5 * (self.nunk[k] + d_prop - 1 - np.arange(d_prop-self.d))) - \
							gammaln(.5 * (self.nu0 + d_prop - 1 - np.arange(d_prop-self.d))))
					for k in range(self.sigmank.shape[0]):
						old_lik += np.sum(.5 * self.lambda0 * log(self.prior_sigma[np.arange(self.d,d_prop)]) - \
							.5 * self.lambdank[k] * log(self.sigmank[k,np.arange(d_prop-self.d)]))					
		else:
			if self.directed:
				for key in ['s','r']:
					if prop_step == 1:
						old_lik += np.sum(gammaln(.5 * (self.nunk[key] + self.d - 1)) - gammaln(.5 * (self.nu0[key] + self.d - 1)))
						new_lik += np.sum(.5 * self.lambda0[key] * log(self.prior_sigma[key][d_prop]) - .5 * self.lambdank[key] * log(sigmank_prop[key][:,0]))
					else:
						for k in range(self.K[key] if self.coclust else self.K):
							old_lik += np.sum(gammaln(.5 * (self.nunk[key][k] + self.d - 1 - np.arange(self.d-d_prop))) - \
								gammaln(.5 * (self.nu0[key] + self.d - 1 - np.arange(self.d-d_prop))))
						for k in range(self.sigmank[key].shape[0]):
							new_lik += np.sum(.5 * self.lambda0[key] * log(self.prior_sigma[key][np.arange(d_prop,self.d)]) - \
								.5 * self.lambdank[key][k] * log(sigmank_prop[key][k,np.arange(self.d-d_prop)]))
			else:
				if prop_step == 1:		
					old_lik += np.sum(gammaln(.5 * (self.nunk + self.d - 1)) - gammaln(.5 * (self.nu0 + self.d - 1)))
					new_lik += np.sum(.5 * self.lambda0 * log(self.prior_sigma[d_prop]) - .5 * self.lambdank * log(sigmank_prop[:,0]))
				else:
					for k in range(self.K[key] if self.coclust else self.K):
						old_lik += np.sum(gammaln(.5 * (self.nunk[k] + self.d - 1 - np.arange(self.d-d_prop))) - \
							gammaln(.5 * (self.nu0 + self.d - 1 - np.arange(self.d-d_prop))))
					for k in range(self.sigmank.shape[0]):
						new_lik += np.sum(.5 * self.lambda0 * log(self.prior_sigma[np.arange(d_prop,self.d)]) - \
							.5 * self.lambdank[k] * log(sigmank_prop[k,np.arange(self.d-d_prop)]))
		## Acceptance ratio
		accept_ratio = new_lik - old_lik + (d_prop - self.d) * log(1 - self.delta)
		if prop_step == 1:
			if self.d == 1 or d_prop == 1:
				accept_ratio += (self.d - d_prop) * log(2)
			if self.d == self.m or d_prop == self.m:
				accept_ratio -= (self.d - d_prop) * log(2)
		else:
			accept_ratio += log(q_prop_new) - log(q_prop_old)
		accept = ( -np.random.exponential(1) < accept_ratio )
		print str(exp(accept_ratio))
		## If the proposal is accepted, update the parameters
		if accept:
			self.d = d_prop
			self.sum_x = sum_x_prop
			self.mean_k = mean_k_prop
			self.squared_sum_x = squared_sum_x_prop
			self.sigmank = sigmank_prop
			self.Delta_k = Delta_k_prop
			if self.directed:
				for key in ['s','r']:
					for k in range(self.K[key] if self.coclust else self.K):
						self.Delta_k_inv[key][k] = self.kappank[key][k] / (self.kappank[key][k] + 1.0) * inv(self.Delta_k[key][k])					
			else:
				for k in range(self.K):
					self.Delta_k_inv[k] = self.kappank[k] / (self.kappank[k] + 1.0) * inv(self.Delta_k[k])
			self.Delta_k_det = Delta_k_det_prop
		if verbose:
			print 'Proposal: ' + str(d_prop) + '\t' + 'Accepted: ' + str(accept)
		#if (self.sigmank < 0).any() or (self.lambdank < 0).any():
		#	raise ValueError('Error with sigma and/or lambda')
		#vv = Counter(self.v)
		#vvv = np.array([vv[key] for key in range(self.H)])
		#if (vvv != self.vk).any():
		#	raise ValueError('Error with vs')
		return None

	#################################################
	### c. Propose to split (or merge) two groups ###
	#################################################
	### Split-merge move
	def split_merge(self,verbose=False):
		## For coclustering: Gibbs sample at random sources or receivers
		if self.coclust:
			sr = np.random.choice(['s','r'])
		## Randomly choose two indices
		i,j = np.random.choice(self.n[sr] if self.coclust else self.n,size=2,replace=False)
		## Propose a split or merge move according to the sampled values
		if (self.z[sr][i] == self.z[sr][j]) if self.coclust else (self.z[i] == self.z[j]):
			split = True
			if self.coclust:
				zsplit = self.z[sr][i]
				## Obtain the indices that must be re-allocated
				S = np.delete(range(self.n[sr]),[i,j])[np.delete(self.z[sr] == zsplit,[i,j])]
				z_prop = np.copy(self.z[sr])
				## Move j to a new cluster
				z_prop[j] = self.K[sr]
			else:
				zsplit = self.z[i]
				## Obtain the indices that must be re-allocated
				S = np.delete(range(self.n),[i,j])[np.delete(self.z == zsplit,[i,j])]
				z_prop = np.copy(self.z)
				## Move j to a new cluster
				z_prop[j] = self.K
		else:
			split = False
			## Choose to merge to the cluster with minimum index (zmerge) and remove the cluster with maximum index (zlost)
			if self.coclust:
				zmerge = min([self.z[sr][i],self.z[sr][j]])
				zj = self.z[sr][j]
				zlost = max([self.z[sr][i],self.z[sr][j]])
				z_prop = np.copy(self.z[sr])
				z_prop[self.z[sr] == zlost] = zmerge
				if zlost != self.K[sr]-1:
					for k in range(zlost,self.K[sr]-1):
						z_prop[self.z[sr] == k+1] = k
				## Set of observations for the imaginary split (Dahl, 2003)
				S = np.delete(range(self.n[sr]),[i,j])[np.delete((self.z[sr] == zmerge) + (self.z[sr] == zlost),[i,j])]
			else:
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
		if not self.directed or self.coclust:
			nk_rest = np.ones(2,int)
			if self.coclust:
				nunk_rest = self.nu0[sr] + np.ones(2)
				kappa_rest = self.kappa0[sr] + np.ones(2)
				sum_rest = self.X[sr][[i,j],:self.d]
				if not self.equal_var:
					lambda_rest = self.lambda0[sr] + np.ones(2)
					sigma_rest = self.prior_sigma[sr][self.d:] + (self.X[sr][[i,j],self.d:] ** 2)
				mean_rest = (self.prior_sum[sr][:self.d] + sum_rest) / (self.kappa0[sr] + 1.0)
				squared_sum_restricted = self.full_outer_x[sr][[i,j],:self.d,:self.d]
				Delta_restricted = {}; Delta_restricted_inv = {}; Delta_restricted_det = np.zeros(2)
				for q in [0,1]:
					Delta_restricted[q] = self.Delta0[sr][:self.d,:self.d] + squared_sum_restricted[q] + self.prior_outer[sr][:self.d,:self.d] - \
						kappa_rest[q] * np.outer(mean_rest[q],mean_rest[q])
					Delta_restricted_inv[q] = kappa_rest[q] / (kappa_rest[q] + 1) * inv(Delta_restricted[q])
					Delta_restricted_det[q] = slogdet(Delta_restricted[q])[1]
			else:
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
		else:
			nk_rest = np.ones(2,int)
			nunk_rest = {}; kappa_rest = {}; 
			sum_rest = {}; mean_rest = {}; squared_sum_restricted = {}
			lambda_rest = {}; sigma_rest = {}
			Delta_restricted = {}; Delta_restricted_inv = {}; Delta_restricted_det = {}
			for key in ['s','r']:
				nunk_rest[key] = self.nu0[key] + np.ones(2)
				kappa_rest[key] = self.kappa0[key] + np.ones(2)
				sum_rest[key] = self.X[key][[i,j],:self.d]
				if not self.equal_var:
					lambda_rest[key] = self.lambda0[key] + np.ones(2)
					sigma_rest[key] = self.prior_sigma[key][self.d:] + (self.X[key][[i,j],self.d:] ** 2)
				mean_rest[key] = (self.prior_sum[key][:self.d] + sum_rest[key]) / (self.kappa0[key] + 1.0)
				squared_sum_restricted[key] = self.full_outer_x[key][[i,j],:self.d,:self.d]
				Delta_restricted[key] = {}; Delta_restricted_inv[key] = {}; Delta_restricted_det[key] = np.zeros(2)
				for q in [0,1]:
					Delta_restricted[key][q] = self.Delta0[key][:self.d,:self.d] + squared_sum_restricted[key][q] + self.prior_outer[key][:self.d,:self.d] - \
						kappa_rest[key][q] * np.outer(mean_rest[key][q],mean_rest[key][q])
					Delta_restricted_inv[key][q] = kappa_rest[key][q] / (kappa_rest[key][q] + 1.0) * inv(Delta_restricted[key][q])
					Delta_restricted_det[key][q] = slogdet(Delta_restricted[key][q])[1]
		## Randomly permute the indices in S and calculate the sequential allocations 
		for h in np.random.permutation(S):
			## Calculate the predictive probability
			if not self.directed or self.coclust:
				if self.coclust:
					position = self.X[sr][h,:self.d]
					position_right = self.X[sr][h,self.d:]
				else:
					position = self.X[h,:self.d]
					position_right = self.X[h,self.d:]
				out_position = np.outer(position,position)
				
				log_pred_prob = np.array([(mvt.dmvt_efficient(x=position,mu=mean_rest[q],Sigma_inv=Delta_restricted_inv[q], \
						Sigma_logdet=self.d*log((kappa_rest[q] + 1.0) / (kappa_rest[q] * nunk_rest[q])) + Delta_restricted_det[q], \
						nu=nunk_rest[q])) for q in [0,1]]).reshape(2,) + log(nunk_rest[q] + (self.alpha[sr] if self.coclust else self.alpha)/2.0)					
			else:
				position = {}; position_right = {}; out_position = {}; log_pred_prob_sr = {}
				for key in ['s','r']:
					position[key] = self.X[key][h,:self.d]
					position_right[key] = self.X[key][h,self.d:]
					out_position[key] = np.outer(position[key],position[key])
					log_pred_prob_sr[key] = np.array([(mvt.dmvt_efficient(x=position[key],mu=mean_rest[key][q],Sigma_inv=Delta_restricted_inv[key][q], \
						Sigma_logdet=self.d*log((kappa_rest[key][q] + 1.0) / (kappa_rest[key][q] * nunk_rest[key][q])) + Delta_restricted_det[key][q], \
						nu=nunk_rest[key][q])) for q in [0,1]]).reshape(2,) + log(nunk_rest[key][q] + self.alpha/2.0)
				log_pred_prob = log_pred_prob_sr['s'] + log_pred_prob_sr['r']
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
				z_prop[h] = [zsplit,self.K[sr] if self.coclust else self.K][znew]
			else:
				## Determine the new value deterministically
				znew = int((self.z[sr][h] if self.coclust else self.z[h]) == zj)
				## Update proposal ratio in the imaginary split
				prop_ratio += log(pred_prob[znew])
				if pred_prob[znew] == 0.0:
					warnings.warn('Imaginary split yields impossible outcome: merge proposal automatically rejected')
			## Calculate second order cluster allocation (restricted posterior predictive)
			## Update parameters 
			nk_rest[znew] += 1.0
			if not self.directed or self.coclust:
				nunk_rest[znew] += 1.0
				kappa_rest[znew] += 1.0
				sum_rest[znew] += position
				if not self.equal_var:
					lambda_rest[znew] += 1.0
					sigma_rest[znew] += (position_right ** 2)
				mean_rest[znew] = ((self.prior_sum[sr][:self.d] if self.coclust else self.prior_sum[:self.d]) + sum_rest[znew]) / kappa_rest[znew]
				squared_sum_restricted[znew] += out_position
				Delta_restricted[znew] = (self.Delta0[sr][:self.d,:self.d] if self.coclust else self.Delta0[:self.d,:self.d]) + squared_sum_restricted[znew] + \
					(self.prior_outer[sr][:self.d,:self.d] if self.coclust else self.prior_outer[:self.d,:self.d]) - \
					kappa_rest[znew] * np.outer(mean_rest[znew],mean_rest[znew])
				Delta_restricted_inv[znew] = kappa_rest[znew] / (kappa_rest[znew] + 1.0) * inv(Delta_restricted[znew])
				Delta_restricted_det[znew] = slogdet(Delta_restricted[znew])[1]
			else:
				for key in ['s','r']:
					nunk_rest[key][znew] += 1.0
					kappa_rest[key][znew] += 1.0
					sum_rest[key][znew] += position[key]
					if not self.equal_var:
						lambda_rest[key][znew] += 1.0
						sigma_rest[key][znew] += (position_right[key] ** 2)
					mean_rest[key][znew] = (self.prior_sum[key][:self.d] + sum_rest[key][znew]) / kappa_rest[key][znew]
					squared_sum_restricted[key][znew] += out_position[key]
					Delta_restricted[key][znew] = self.Delta0[key][:self.d,:self.d] + squared_sum_restricted[key][znew] + self.prior_outer[key][:self.d,:self.d] - \
						kappa_rest[key][znew] * np.outer(mean_rest[key][znew],mean_rest[key][znew])
					Delta_restricted_inv[key][znew] = kappa_rest[key][znew] / (kappa_rest[key][znew] + 1.0) * inv(Delta_restricted[key][znew])
					Delta_restricted_det[key][znew] = slogdet(Delta_restricted[key][znew])[1]
		## Calculate the new second order allocation if equal_var is true
		if self.equal_var:
			if split:
				## Calculate the new second order allocation of the first cluster after splitting
				ind_left = np.where(z_prop == zsplit)[0]
				ind_right = np.where(z_prop == (self.K[sr] if self.coclust else self.K))[0]
				if not self.directed or self.coclust:
					if self.coclust:
						lambda_left = np.copy(self.lambdank[sr])
						lambda_left[self.v[sr][zsplit]] -= nk_rest[0]
						sigma_left = np.copy(self.sigmank[sr])
						sigma_left[self.v[sr][zsplit]] -= np.sum(self.X[sr][ind_left,self.d:] ** 2,axis=0)
						prob_v_left = np.zeros(self.H[sr])
						for node in np.random.permutation(ind_left):
							pos = self.X[sr][node,self.d:]
							prob_v_left += np.array([np.sum(t.logpdf(pos,df=lambda_left[h],loc=0, \
								scale=sqrt(sigma_left[h]/lambda_left[h]))) for h in range(self.H[sr])])
							lambda_left += 1.0
							sigma_left += (pos ** 2)
						## Calculate the second order allocation of the second cluster after splitting
						lambda_right = np.copy(self.lambdank[sr])
						lambda_right[self.v[sr][zsplit]] -= nk_rest[1]
						sigma_right = np.copy(self.sigmank[sr])
						sigma_right[self.v[sr][zsplit]] -= np.sum(self.X[sr][ind_right,self.d:] ** 2,axis=0)
						prob_v_right = np.zeros(self.H[sr])
						for node in np.random.permutation(ind_right):
							pos = self.X[sr][node,self.d:]
							prob_v_right += np.array([np.sum(t.logpdf(pos,df=lambda_right[h],loc=0, \
								scale=sqrt(sigma_right[h]/lambda_right[h]))) for h in range(self.H[sr])])
							lambda_right += 1.0
							sigma_right += (pos ** 2)
					else:
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
					## Directed graph - standard clustering
					lambda_left = {}; sigma_left = {}
					for key in ['s','r']:
						lambda_left[key] = np.copy(self.lambdank[key])
						lambda_left[key][self.v[zsplit]] -= nk_rest[0]
						sigma_left[key] = np.copy(self.sigmank[key])
						sigma_left[key][self.v[zsplit]] -= np.sum(self.X[key][ind_left,self.d:] ** 2,axis=0)
					prob_v_left = np.zeros(self.H)
					for node in np.random.permutation(ind_left):
						for key in ['s','r']:
							pos = self.X[key][node,self.d:]
							prob_v_left += np.array([np.sum(t.logpdf(pos,df=lambda_left[key][h],loc=0,
								scale=sqrt(sigma_left[key][h]/lambda_left[key][h]))) for h in range(self.H)])
							lambda_left[key] += 1.0
							sigma_left[key] += (pos ** 2)
					## Calculate the second order allocation of the second cluster after splitting
					lambda_right = {}; sigma_right = {}
					for key in ['s','r']:
						lambda_right[key] = np.copy(self.lambdank[key])
						lambda_right[key][self.v[zsplit]] -= nk_rest[1]
						sigma_right[key] = np.copy(self.sigmank[key])
						sigma_right[key][self.v[zsplit]] -= np.sum(self.X[key][ind_right,self.d:] ** 2,axis=0)
					prob_v_right = np.zeros(self.H)
					for node in np.random.permutation(ind_right):
						for key in ['s','r']:
							pos = self.X[key][node,self.d:]
							prob_v_right += np.array([np.sum(t.logpdf(pos,df=lambda_right[key][h],loc=0,
								scale=sqrt(sigma_right[key][h]/lambda_right[key][h]))) for h in range(self.H)])
							lambda_right[key] += 1.0
							sigma_right[key] += (pos ** 2)
			else:
				if not self.directed or self.coclust:
					if self.coclust:
						## Calculate the new second order allocation of the first cluster after the (imaginary) split
						lambda_left = np.copy(self.lambdank[sr])
						lambda_left[self.v[sr][zmerge]] -= self.nk[sr][zmerge]
						sigma_left = np.copy(self.sigmank[sr])
						sigma_left[self.v[sr][zmerge]] -= np.sum(self.X[sr][self.z[sr] == zmerge,self.d:] ** 2,axis=0)
						prob_v_left = np.zeros(self.H[sr])
						for node in np.random.permutation(np.where(self.z[sr] == zmerge)[0]):
							pos = self.X[sr][node,self.d:]
							prob_v_left += np.array([np.sum(t.logpdf(pos,df=lambda_left[h],loc=0,scale=sqrt(sigma_left[h]/lambda_left[h]))) for h in range(self.H[sr])])
							lambda_left += 1.0
							sigma_left += (pos ** 2)
						## Calculate the new second order allocation of the first cluster after the (imaginary) split
						lambda_right = np.copy(self.lambdank[sr])
						lambda_right[self.v[sr][zlost]] -= self.nk[sr][zlost]
						sigma_right = np.copy(self.sigmank[sr])
						sigma_right[self.v[sr][zlost]] -= np.sum(self.X[sr][self.z[sr] == zlost,self.d:] ** 2,axis=0)
						prob_v_right = np.zeros(self.H[sr])
						for node in np.random.permutation(np.where(self.z[sr] == zlost)[0]):
							pos = self.X[sr][node,self.d:]
							prob_v_right += np.array([np.sum(t.logpdf(pos,df=lambda_right[h],loc=0,scale=sqrt(sigma_right[h]/lambda_right[h]))) for h in range(self.H[sr])])
							lambda_right += 1.0
							sigma_right += (pos ** 2)
					else:
						## Directed graph - standard clustering
						lambda_left = {}; sigma_left = {}		
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
				else:
					## Directed graph - standard clustering
					lambda_left = {}; sigma_left = {}
					## Calculate the new second order allocation of the first cluster after the (imaginary) split
					for key in ['s','r']:
						lambda_left[key] = np.copy(self.lambdank[key])
						lambda_left[key][self.v[zmerge]] -= self.nk[zmerge]
						sigma_left[key] = np.copy(self.sigmank[key])
						sigma_left[key][self.v[zmerge]] -= np.sum(self.X[key][self.z == zmerge,self.d:] ** 2,axis=0)
					prob_v_left = np.zeros(self.H)
					for node in np.random.permutation(np.where(self.z == zmerge)[0]):
						for key in ['s','r']:
							pos = self.X[key][node,self.d:]
							prob_v_left += np.array([np.sum(t.logpdf(pos,df=lambda_left[key][h],loc=0,
								scale=sqrt(sigma_left[key][h]/lambda_left[key][h]))) for h in range(self.H)])
							lambda_left[key] += 1.0
							sigma_left[key] += (pos ** 2)
					## Calculate the new second order allocation of the first cluster after the (imaginary) split
					lambda_right = {}; sigma_right = {}
					for key in ['s','r']:
						lambda_right[key] = np.copy(self.lambdank[key])
						lambda_right[key][self.v[zlost]] -= self.nk[zlost]
						sigma_right[key] = np.copy(self.sigmank[key])
						sigma_right[key][self.v[zlost]] -= np.sum(self.X[key][self.z == zlost,self.d:] ** 2,axis=0)
					prob_v_right = np.zeros(self.H)
					for node in np.random.permutation(np.where(self.z == zlost)[0]):
						for key in ['s','r']:
							pos = self.X[key][node,self.d:]
							prob_v_right += np.array([np.sum(t.logpdf(pos,df=lambda_right[key][h],loc=0,
								scale=sqrt(sigma_right[key][h]/lambda_right[key][h]))) for h in range(self.H)])
							lambda_right[key] += 1.0
							sigma_right[key] += (pos ** 2)
			## Resample the second order cluster allocation
			prob_v_left = exp(prob_v_left-max(prob_v_left)) / np.sum(exp(prob_v_left-np.max(prob_v_left)))
			left_sord = np.random.choice(range(self.H[sr] if self.coclust else self.H),p=prob_v_left)
			prob_v_right = exp(prob_v_right-max(prob_v_right)) / np.sum(exp(prob_v_right-max(prob_v_right)))
			right_sord = np.random.choice(range(self.H[sr] if self.coclust else self.H),p=prob_v_right)
			## Calculate the cumulative probability of allocation to the specific pair (left_sord and right_sord)
			prob_second_order = log(prob_v_left[left_sord]) + log(prob_v_right[right_sord])
			## Compute the proposed values of lambdank and sigmank
			if not self.directed or self.coclust:
				if self.coclust:
					lambdank_prop = np.copy(self.lambdank[sr])
					sigmank_prop = np.copy(self.sigmank[sr])
					v_prop = np.copy(self.v[sr])
					vk_prop = np.copy(self.vk[sr])
				else:
					lambdank_prop = np.copy(self.lambdank)
					sigmank_prop = np.copy(self.sigmank)
					v_prop = np.copy(self.v)
					vk_prop = np.copy(self.vk)
			else:
				lambdank_prop = {}; sigmank_prop = {}
				for key in ['s','r']:
					lambdank_prop[key] = np.copy(self.lambdank[key])
					sigmank_prop[key] = np.copy(self.sigmank[key])
				v_prop = np.copy(self.v)
				vk_prop = np.copy(self.vk)
			if split:
				## Propose new values for the right hand side of the matrix
				v_prop[zsplit] = left_sord
				v_prop = np.append(v_prop,right_sord)
				## Second order cluster counts
				vk_prop[self.v[sr][zsplit] if self.coclust else self.v[zsplit]] -= 1.0
				vk_prop[left_sord] += 1.0
				vk_prop[right_sord] += 1.0
				## Update lambda and sigma
				if not self.directed or self.coclust:
					if self.coclust:
						lambdank_prop[self.v[sr][zsplit]] -= self.nk[sr][zsplit]	
						lambdank_prop[left_sord] += nk_rest[0]
						lambdank_prop[right_sord] += nk_rest[1]
						sigmank_prop[self.v[sr][zsplit]] -= np.sum(self.X[sr][self.z[sr] == zsplit,self.d:] ** 2, axis=0)
						sigmank_prop[left_sord] += np.sum(self.X[sr][ind_left, self.d:] ** 2, axis=0)
						sigmank_prop[right_sord] += np.sum(self.X[sr][ind_right, self.d:] ** 2, axis=0)
					else:
						lambdank_prop[self.v[zsplit]] -= self.nk[zsplit]	
						lambdank_prop[left_sord] += nk_rest[0]
						lambdank_prop[right_sord] += nk_rest[1]
						sigmank_prop[self.v[zsplit]] -= np.sum(self.X[self.z == zsplit,self.d:] ** 2, axis=0)
						sigmank_prop[left_sord] += np.sum(self.X[ind_left, self.d:] ** 2, axis=0)
						sigmank_prop[right_sord] += np.sum(self.X[ind_right, self.d:] ** 2, axis=0)
				else:
					for key in ['s','r']:
						lambdank_prop[key][self.v[zsplit]] -= self.nk[zsplit]	
						lambdank_prop[key][left_sord] += nk_rest[0]
						lambdank_prop[key][right_sord] += nk_rest[1]
						sigmank_prop[key][self.v[zsplit]] -= np.sum(self.X[key][self.z == zsplit,self.d:] ** 2, axis=0)
						sigmank_prop[key][left_sord] += np.sum(self.X[key][ind_left, self.d:] ** 2, axis=0)
						sigmank_prop[key][right_sord] += np.sum(self.X[key][ind_right, self.d:] ** 2, axis=0)
			else:
				v_prop = np.delete(v_prop,zlost)
				samp_so = np.random.choice([0,1])
				second_order = [self.v[sr][zmerge],self.v[sr][zlost]][samp_so] if self.coclust else [self.v[zmerge],self.v[zlost]][samp_so]
				v_prop[zmerge] = second_order
				vk_prop[[self.v[sr][zmerge],self.v[sr][zlost]][1-samp_so] if self.coclust else [self.v[zmerge],self.v[zlost]][1-samp_so]] -= 1.0
				if not self.directed or self.coclust:
					if self.coclust:
						lambdank_prop[self.v[sr][zmerge]] -= self.nk[sr][zmerge]
						lambdank_prop[self.v[sr][zlost]] -= self.nk[sr][zlost]
						lambdank_prop[second_order] += self.nk[sr][zmerge] + self.nk[sr][zlost]
						Xzm = np.sum(self.X[sr][self.z[sr] == zmerge, self.d:] ** 2, axis=0)
						Xzl = np.sum(self.X[sr][self.z[sr] == zlost, self.d:] ** 2, axis=0)
						sigmank_prop[self.v[sr][zmerge]] -= Xzm
						sigmank_prop[self.v[sr][zlost]] -= Xzl
						sigmank_prop[second_order] += Xzm + Xzl
					else:
						lambdank_prop[self.v[zmerge]] -= self.nk[zmerge]
						lambdank_prop[self.v[zlost]] -= self.nk[zlost]
						lambdank_prop[second_order] += self.nk[zmerge] + self.nk[zlost]
						Xzm = np.sum(self.X[self.z == zmerge, self.d:] ** 2, axis=0)
						Xzl = np.sum(self.X[self.z == zlost, self.d:] ** 2, axis=0)
						sigmank_prop[self.v[zmerge]] -= Xzm
						sigmank_prop[self.v[zlost]] -= Xzl
						sigmank_prop[second_order] += Xzm + Xzl
				else:
					for key in ['s','r']:
						lambdank_prop[key][self.v[zmerge]] -= self.nk[zmerge]
						lambdank_prop[key][self.v[zlost]] -= self.nk[zlost]
						lambdank_prop[key][second_order] += self.nk[zmerge] + self.nk[zlost]
						Xzm = np.sum(self.X[key][self.z == zmerge, self.d:] ** 2, axis=0)
						Xzl = np.sum(self.X[key][self.z == zlost, self.d:] ** 2, axis=0)
						sigmank_prop[key][self.v[zmerge]] -= Xzm
						sigmank_prop[key][self.v[zlost]] -= Xzl
						sigmank_prop[key][second_order] += Xzm + Xzl
		## Calculate the acceptance probability
		if split:
			## Calculate the acceptance ratio
			## Left hand side of matrix
			accept_ratio = 0
			if not self.directed or self.coclust:
				if self.coclust:			
					accept_ratio += .5 * self.d * (log(self.kappa0[sr]) + log(self.kappank[sr][zsplit]) - np.sum(log(kappa_rest))) 
					accept_ratio += .5 * (self.nu0[sr] + self.d - 1) * self.Delta0_det[sr][self.d] + \
						.5 * (self.nunk[sr][zsplit] + self.d - 1) * self.Delta_k_det[sr][zsplit] - .5 * np.sum((nunk_rest + self.d - 1) * Delta_restricted_det)
					accept_ratio += np.sum(gammaln(.5 * (np.subtract.outer(nunk_rest + self.d,np.arange(self.d) + 1))))
					accept_ratio -= np.sum(gammaln(.5 * (self.nu0[sr] + self.d - np.arange(self.d) + 1))) + \
						np.sum(gammaln(.5 * (self.nunk[sr][zsplit] + self.d - np.arange(self.d) + 1))) 
				else:
					accept_ratio += .5 * self.d * (log(self.kappa0) + log(self.kappank[zsplit]) - np.sum(log(kappa_rest))) 
					accept_ratio += .5 * (self.nu0 + self.d - 1) * self.Delta0_det[self.d] + .5 * (self.nunk[zsplit] + self.d - 1) * self.Delta_k_det[zsplit] -\
						.5 * np.sum((nunk_rest + self.d - 1) * Delta_restricted_det)
					accept_ratio += np.sum(gammaln(.5 * (np.subtract.outer(nunk_rest + self.d,np.arange(self.d) + 1))))
					accept_ratio -= np.sum(gammaln(.5 * (self.nu0 + self.d - np.arange(self.d) + 1))) + \
						np.sum(gammaln(.5 * (self.nunk[zsplit] + self.d - np.arange(self.d) + 1))) 
			else:
				for key in ['s','r']:
					accept_ratio += .5 * self.d * (log(self.kappa0[key]) + log(self.kappank[key][zsplit]) - np.sum(log(kappa_rest[key]))) 
					accept_ratio += .5 * (self.nu0[key] + self.d - 1) * self.Delta0_det[key][self.d] + \
						.5 * (self.nunk[key][zsplit] + self.d - 1) * self.Delta_k_det[key][zsplit] - .5 * np.sum((nunk_rest[key] + self.d - 1) * Delta_restricted_det[key])
					accept_ratio += np.sum(gammaln(.5 * (np.subtract.outer(nunk_rest[key] + self.d,np.arange(self.d) + 1))))
					accept_ratio -= np.sum(gammaln(.5 * (self.nu0[key] + self.d - np.arange(self.d) + 1))) + \
						np.sum(gammaln(.5 * (self.nunk[key][zsplit] + self.d - np.arange(self.d) + 1))) 
			## Right hand side of matrix
			if self.equal_var:
				if not self.directed or self.coclust:
					if self.coclust:
						accept_ratio += np.sum([(self.m - self.d) * (gammaln(.5 * lambdank_prop[h]) - gammaln(.5 * self.lambda0[sr])) + \
							np.sum(.5*self.lambda0[sr]*log(self.prior_sigma[sr][self.d:]) - .5*lambdank_prop[h]*log(sigmank_prop[h])) for h in range(self.H[sr])])
						accept_ratio -= np.sum([(self.m - self.d) * (gammaln(.5 * self.lambdank[sr][h]) - gammaln(.5 * self.lambda0[sr])) + \
							np.sum(.5*self.lambda0[sr]*log(self.prior_sigma[sr][self.d:]) - .5*self.lambdank[sr][h]*log(self.sigmank[sr][h])) for h in range(self.H[sr])])
						accept_ratio += np.sum(gammaln(vk_prop + self.beta[sr] / self.H[sr])) - np.sum(gammaln(self.vk[sr] + self.beta[sr] / self.H[sr])) + \
							gammaln(self.K[sr] + self.beta[sr]) - gammaln(self.K[sr] + 1 + self.beta[sr])
						accept_ratio -= prob_second_order
					else:
						accept_ratio += np.sum([(self.m - self.d) * (gammaln(.5 * lambdank_prop[h]) - gammaln(.5 * self.lambda0)) + \
							np.sum(.5*self.lambda0*log(self.prior_sigma[self.d:]) - .5*lambdank_prop[h]*log(sigmank_prop[h])) for h in range(self.H)])
						accept_ratio -= np.sum([(self.m - self.d) * (gammaln(.5 * self.lambdank[h]) - gammaln(.5 * self.lambda0)) + \
							np.sum(.5*self.lambda0*log(self.prior_sigma[self.d:]) - .5*self.lambdank[h]*log(self.sigmank[h])) for h in range(self.H)])
						accept_ratio += np.sum(gammaln(vk_prop + self.beta / self.H)) - np.sum(gammaln(self.vk + self.beta / self.H)) + gammaln(self.K + self.beta) - \
							gammaln(self.K + 1 + self.beta)
						accept_ratio -= prob_second_order
				else:
					for key in ['s','r']:
						accept_ratio += np.sum([(self.m - self.d) * (gammaln(.5 * lambdank_prop[key][h]) - gammaln(.5 * self.lambda0[key])) + \
							np.sum(.5*self.lambda0[key]*log(self.prior_sigma[key][self.d:]) - .5*lambdank_prop[key][h]*log(sigmank_prop[key][h])) for h in range(self.H)])
						accept_ratio -= np.sum([(self.m - self.d) * (gammaln(.5 * self.lambdank[key][h]) - gammaln(.5 * self.lambda0[key])) + \
							np.sum(.5*self.lambda0[key]*log(self.prior_sigma[key][self.d:]) - .5*self.lambdank[key][h]*log(self.sigmank[key][h])) for h in range(self.H)])
					accept_ratio += np.sum(gammaln(vk_prop + self.beta / self.H)) - np.sum(gammaln(self.vk + self.beta / self.H)) + \
						gammaln(self.K + self.beta) - gammaln(self.K + 1 + self.beta)
					accept_ratio -= prob_second_order
			else:
				if not self.directed or self.coclust:
					if self.coclust:
						accept_ratio += (self.m - self.d) * (np.sum(gammaln(.5 * lambda_rest)) - gammaln(.5 * self.lambdank[sr][zsplit]) - gammaln(.5 * self.lambda0[sr]))
						accept_ratio += np.sum(.5 * self.lambda0[sr] * log(self.prior_sigma[sr][self.d:]) + .5 * self.lambdank[sr][zsplit] * log(self.sigmank[sr][zsplit]))
					else:
						accept_ratio += (self.m - self.d) * (np.sum(gammaln(.5 * lambda_rest)) - gammaln(.5 * self.lambdank[zsplit]) - gammaln(.5 * self.lambda0))
						accept_ratio += np.sum(.5 * self.lambda0 * log(self.prior_sigma[self.d:]) + .5 * self.lambdank[zsplit] * log(self.sigmank[zsplit]))
					accept_ratio -= .5 * np.sum(np.multiply(lambda_rest,log(sigma_rest).T).T)
				else:
					for key in ['s','r']:
						accept_ratio += (self.m - self.d) * (np.sum(gammaln(.5 * lambda_rest[key])) - \
							gammaln(.5 * self.lambdank[key][zsplit]) - gammaln(.5 * self.lambda0[key]))
						accept_ratio += np.sum(.5 * self.lambda0[key] * log(self.prior_sigma[key][self.d:]) + .5 * self.lambdank[key][zsplit] * log(self.sigmank[key][zsplit]))
						accept_ratio -= .5 * np.sum(np.multiply(lambda_rest[key],log(sigma_rest[key]).T).T)
			## Cluster allocations
			if self.coclust:
				accept_ratio += self.K[sr] * gammaln(float(self.alpha[sr]) / self.K[sr]) - (self.K[sr]+1) * gammaln(self.alpha[sr] / (self.K[sr]+1)) - \
					np.sum(gammaln(self.nk[sr] + float(self.alpha[sr])/self.K[sr]))
				accept_ratio += np.sum(gammaln(np.delete(self.nk[sr],zsplit) + self.alpha[sr]/(self.K[sr]+1.0))) + \
					np.sum(gammaln(nk_rest + self.alpha[sr] / (self.K[sr]+1.0)))	
			else:		
				accept_ratio += self.K * gammaln(float(self.alpha) / self.K) - (self.K+1) * gammaln(self.alpha / (self.K+1)) - \
					np.sum(gammaln(self.nk + float(self.alpha)/self.K))
				accept_ratio += np.sum(gammaln(np.delete(self.nk,zsplit) + self.alpha/(self.K+1.0))) + np.sum(gammaln(nk_rest + self.alpha / (self.K+1.0)))
			## Prior on K and q function
			accept_ratio += log(1.0 - self.omega) - prop_ratio
		else:
			## Merge the two clusters and calculate the acceptance ratio
			nk_sum = np.sum((self.nk[sr] if self.coclust else self.nk)[[zmerge,zlost]])
			if not self.directed or self.coclust:
				if self.coclust:
					nunk_sum = self.nu0[sr] + nk_sum
					kappank_sum = self.kappa0[sr] + nk_sum
					lambdank_sum = self.lambda0[sr] + nk_sum
					sum_x_sum = self.sum_x[sr][zmerge] + self.sum_x[sr][zlost]
					mean_k_sum = (self.prior_sum[sr][:self.d] + sum_x_sum) / kappank_sum
					squared_sum_x_sum = self.squared_sum_x[sr][zlost] + self.squared_sum_x[sr][zlost]
					Delta_det_merged = slogdet(self.Delta0[sr][:self.d,:self.d] + squared_sum_x_sum + self.prior_outer[sr][:self.d,:self.d] - \
						kappank_sum * np.outer(mean_k_sum,mean_k_sum))[1]
					## Caculate acceptance ratio
					## Left hand side of matrix
					accept_ratio = -.5 * self.d * (log(self.kappa0[sr]) + np.sum(log(self.kappank[sr][[zmerge,zlost]])) - log(self.kappa0[sr] + \
						np.sum(self.nk[sr][[zmerge,zlost]])))
					accept_ratio -= .5 * (self.nu0[sr]+self.d-1)*self.Delta0_det[sr][self.d] + .5 * (self.nu0[sr]+ \
						np.sum(self.nk[sr][[zmerge,zlost]]) + self.d - 1) * Delta_det_merged 
					accept_ratio += .5 * np.sum((self.nunk[sr][[zmerge,zlost]] + self.d - 1) * self.Delta_k_det[sr][[zmerge,zlost]])
					accept_ratio += np.sum(gammaln(.5 * (self.nu0[sr] + self.d - np.arange(self.d) + 1))) + \
						np.sum(gammaln(.5 * (self.nu0[sr] + np.sum(self.nk[sr][[zmerge,zlost]]) + self.d - np.arange(self.d)+1)))
					accept_ratio -= np.sum(gammaln(.5 * (np.subtract.outer(self.nunk[sr][[zmerge,zlost]] + self.d,np.arange(self.d) + 1))))
					## Right hand side of matrix
					if self.equal_var:
						accept_ratio += np.sum([(self.m - self.d) * (gammaln(.5 * lambdank_prop[h]) - gammaln(.5 * self.lambda0[sr])) + \
							np.sum(.5*self.lambda0[sr]*log(self.prior_sigma[sr][self.d:]) - .5*lambdank_prop[h]*log(sigmank_prop[h])) for h in range(self.H[sr])])
						accept_ratio -= np.sum([(self.m - self.d) * (gammaln(.5 * self.lambdank[sr][h]) - gammaln(.5 * self.lambda0[sr])) + \
							np.sum(.5*self.lambda0[sr]*log(self.prior_sigma[sr][self.d:]) - .5*self.lambdank[sr][h]*log(self.sigmank[sr][h])) for h in range(self.H[sr])])
						accept_ratio += np.sum(gammaln(vk_prop + self.beta[sr] / self.H[sr])) - np.sum(gammaln(self.vk[sr] + self.beta[sr] / self.H[sr])) + \
							gammaln(self.K[sr] + self.beta[sr]) - gammaln(self.K[sr] - 1 + self.beta[sr])
						accept_ratio += prob_second_order - (self.v[sr][zmerge] != self.v[sr][zlost])*log(.5)
					else:
						accept_ratio -= (self.m - self.d) * (np.sum(gammaln(.5 * self.lambdank[sr][[zmerge,zlost]])) - gammaln(.5 * self.lambda0[sr]) - \
							gammaln(.5 * (self.lambda0[sr] + np.sum(self.nk[sr][[zmerge,zlost]]))))
						accept_ratio -= np.sum(.5 * self.lambda0[sr] * log(self.prior_sigma[sr][self.d:])) - \
							.5 * np.sum(np.multiply(self.lambdank[sr][[zmerge,zlost]],log(self.sigmank[sr][[zmerge,zlost]]).T).T)
						accept_ratio -= np.sum(.5 * (self.lambda0[sr] + \
							np.sum(self.nk[sr][[zmerge,zlost]])) * log(np.sum(self.sigmank[sr][[zmerge,zlost]],axis=0) - self.prior_sigma[sr][self.d:]))
				else:
					nunk_sum = self.nu0 + nk_sum
					kappank_sum = self.kappa0 + nk_sum
					lambdank_sum = self.lambda0 + nk_sum
					sum_x_sum = self.sum_x[zmerge] + self.sum_x[zlost]
					## sigma_sum = self.sigmank[zmerge] + self.sigmank[zlost] - self.prior_sigma[self.d:]
					mean_k_sum = (self.prior_sum[:self.d] + sum_x_sum) / kappank_sum
					squared_sum_x_sum = self.squared_sum_x[zlost] + self.squared_sum_x[zlost]
					Delta_det_merged = slogdet(self.Delta0[:self.d,:self.d] + squared_sum_x_sum + self.prior_outer[:self.d,:self.d] - \
						kappank_sum * np.outer(mean_k_sum,mean_k_sum))[1]
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
						accept_ratio -= np.sum(.5 * (self.lambda0 + \
							np.sum(self.nk[[zmerge,zlost]])) * log(np.sum(self.sigmank[[zmerge,zlost]],axis=0) - self.prior_sigma[self.d:]))
			else:
				nunk_sum = {}; kappank_sum = {}; lambdank_sum = {}; sum_x_sum = {}; mean_k_sum = {}; squared_sum_x_sum = {}; Delta_det_merged = {}
				accept_ratio = 0
				for key in ['s','r']:
					nunk_sum[key] = self.nu0[key] + nk_sum		
					kappank_sum[key] = self.kappa0[key] + nk_sum
					lambdank_sum[key] = self.lambda0[key] + nk_sum
					sum_x_sum[key] = self.sum_x[key][zmerge] + self.sum_x[key][zlost]
					mean_k_sum[key] = (self.prior_sum[key][:self.d] + sum_x_sum[key]) / kappank_sum[key]
					squared_sum_x_sum[key] = self.squared_sum_x[key][zlost] + self.squared_sum_x[key][zlost]
					Delta_det_merged[key] = slogdet(self.Delta0[key][:self.d,:self.d] + squared_sum_x_sum[key] + self.prior_outer[key][:self.d,:self.d] - \
						kappank_sum[key] * np.outer(mean_k_sum[key],mean_k_sum[key]))[1]	
					## Caculate acceptance ratio
					## Left hand side of matrix
					accept_ratio += -.5 * self.d * (log(self.kappa0[key]) + np.sum(log(self.kappank[key][[zmerge,zlost]])) - log(self.kappa0[key] + \
						np.sum(self.nk[[zmerge,zlost]])))
					accept_ratio -= .5 * (self.nu0[key]+self.d-1)*self.Delta0_det[key][self.d] + .5 * (self.nu0[key] + \
						np.sum(self.nk[[zmerge,zlost]]) + self.d - 1) * Delta_det_merged[key]
					accept_ratio += .5 * np.sum((self.nunk[key][[zmerge,zlost]] + self.d - 1) * self.Delta_k_det[key][[zmerge,zlost]])
					accept_ratio += np.sum(gammaln(.5 * (self.nu0[key] + self.d - np.arange(self.d) + 1))) + \
						np.sum(gammaln(.5 * (self.nu0[key] + np.sum(self.nk[[zmerge,zlost]]) + self.d - np.arange(self.d)+1)))
					accept_ratio -= np.sum(gammaln(.5 * (np.subtract.outer(self.nunk[key][[zmerge,zlost]] + self.d,np.arange(self.d) + 1))))
				## Right hand side of matrix
				if self.equal_var:
					for key in ['s','r']:
						accept_ratio += np.sum([(self.m - self.d) * (gammaln(.5 * lambdank_prop[key][h]) - gammaln(.5 * self.lambda0[key])) + \
							np.sum(.5*self.lambda0[key]*log(self.prior_sigma[key][self.d:]) - \
							.5*lambdank_prop[key][h]*log(sigmank_prop[key][h])) for h in range(self.H)])
						accept_ratio -= np.sum([(self.m - self.d) * (gammaln(.5 * self.lambdank[key][h]) - gammaln(.5 * self.lambda0[key])) + \
							np.sum(.5*self.lambda0[key]*log(self.prior_sigma[key][self.d:]) - \
							.5*self.lambdank[key][h]*log(self.sigmank[key][h])) for h in range(self.H)])
					accept_ratio += np.sum(gammaln(vk_prop + self.beta / self.H)) - np.sum(gammaln(self.vk + self.beta / self.H)) + gammaln(self.K + self.beta) - \
								gammaln(self.K - 1 + self.beta)
					accept_ratio += prob_second_order - (self.v[zmerge] != self.v[zlost])*log(.5)
				else:
					for key in ['s','r']:
						accept_ratio -= (self.m - self.d) * (np.sum(gammaln(.5 * self.lambdank[key][[zmerge,zlost]])) - gammaln(.5 * self.lambda0[key]) - \
							gammaln(.5 * (self.lambda0[key] + np.sum(self.nk[[zmerge,zlost]]))))
						accept_ratio -= np.sum(.5 * self.lambda0[key] * log(self.prior_sigma[key][self.d:])) - \
							.5 * np.sum(np.multiply(self.lambdank[key][[zmerge,zlost]],log(self.sigmank[key][[zmerge,zlost]]).T).T)
						accept_ratio -= np.sum(.5 * (self.lambda0[key] + \
							np.sum(self.nk[[zmerge,zlost]])) * log(np.sum(self.sigmank[key][[zmerge,zlost]],axis=0) - self.prior_sigma[key][self.d:]))				
			## Cluster allocations
			if self.coclust:
				accept_ratio += self.K[sr] * gammaln(float(self.alpha[sr]) / self.K[sr]) - (self.K[sr]-1) * gammaln(self.alpha[sr] / (self.K[sr]-1.0)) - \
					np.sum(gammaln(self.nk[sr] + float(self.alpha[sr]) / self.K[sr]))
				accept_ratio += np.sum(gammaln(np.delete(self.nk[sr],[zmerge,zlost]) + float(self.alpha[sr])/(self.K[sr]-1.0))) + \
					gammaln(np.sum(nk_rest) + float(self.alpha[sr])/(self.K[sr]-1.0))
			else:
				accept_ratio += self.K * gammaln(float(self.alpha) / self.K) - (self.K-1) * gammaln(self.alpha / (self.K-1.0)) - \
					np.sum(gammaln(self.nk + float(self.alpha) / self.K))
				accept_ratio += np.sum(gammaln(np.delete(self.nk,[zmerge,zlost]) + float(self.alpha)/(self.K-1.0))) + \
					gammaln(np.sum(nk_rest) + float(self.alpha)/(self.K-1.0))
			## Prior on K and q function
			accept_ratio -= log(1.0 - self.omega) - prop_ratio
			#if ((np.sum(self.sigmank[[zmerge,zlost]],axis=0) - self.prior_sigma[self.d:]) <= 0.0).any():
			#	raise ValueError('ERROR')
		## Accept or reject the proposal
		accept = (-np.random.exponential(1) < accept_ratio)
		if accept:
			## Update the stored values
			if split:
				if not self.directed or self.coclust:
					if self.coclust:
						self.z[sr] = np.copy(z_prop)
						self.nk[sr][zsplit] = nk_rest[0]
						self.nk[sr] = np.append(self.nk[sr],nk_rest[1])
						self.nunk[sr] = self.nu0[sr] + self.nk[sr]
						self.kappank[sr] = self.kappa0[sr] + self.nk[sr]
						self.sum_x[sr][zsplit] = sum_rest[0]
						self.sum_x[sr] = np.vstack((self.sum_x[sr],sum_rest[1]))
						if self.equal_var:
							self.v[sr] = v_prop
							self.vk[sr] = vk_prop		
							self.lambdank[sr] = lambdank_prop
							self.sigmank[sr] = sigmank_prop
						else:
							self.lambdank[sr] = self.lambda0[sr] + self.nk[sr]
							self.sigmank[sr][zsplit] = sigma_rest[0]
							self.sigmank[sr] = np.vstack((self.sigmank[sr],sigma_rest[1]))
						self.mean_k[sr] = np.divide((self.prior_sum[sr][:self.d] + self.sum_x[sr]).T,self.kappank[sr]).T
						self.squared_sum_x[sr][zsplit] = squared_sum_restricted[0]
						self.squared_sum_x[sr][self.K[sr]] = squared_sum_restricted[1]
						self.Delta_k[sr][zsplit] = Delta_restricted[0]
						self.Delta_k[sr][self.K[sr]] = Delta_restricted[1]
						self.Delta_k_inv[sr][zsplit] = Delta_restricted_inv[0]
						self.Delta_k_inv[sr][self.K[sr]] = Delta_restricted_inv[1]
						self.Delta_k_det[sr][zsplit] = Delta_restricted_det[0]
						self.Delta_k_det[sr] = np.append(self.Delta_k_det[sr],Delta_restricted_det[1])
						## Update K 
						self.K[sr] += 1
					else:
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
					self.nk[zsplit] = nk_rest[0]
					self.nk = np.append(self.nk,nk_rest[1])
					for key in ['s','r']:
						self.nunk[key] = self.nu0[key] + self.nk 
						self.kappank[key] = self.kappa0[key] + self.nk
						self.sum_x[key][zsplit] = sum_rest[key][0]
						self.sum_x[key] = np.vstack((self.sum_x[key],sum_rest[key][1]))
					if self.equal_var:
						self.v = v_prop
						self.vk = vk_prop	
						for key in ['s','r']:	
							self.lambdank[key] = lambdank_prop[key]
							self.sigmank[key] = sigmank_prop[key]
					else:
						for key in ['s','r']:
							self.lambdank[key] = self.lambda0[key] + self.nk
							self.sigmank[key][zsplit] = sigma_rest[key][0]
							self.sigmank[key] = np.vstack((self.sigmank[key],sigma_rest[key][1]))
					for key in ['s','r']:	
						self.mean_k[key] = np.divide((self.prior_sum[key][:self.d] + self.sum_x[key]).T,self.kappank[key]).T
						self.squared_sum_x[key][zsplit] = squared_sum_restricted[key][0]
						self.squared_sum_x[key][self.K] = squared_sum_restricted[key][1]
						self.Delta_k[key][zsplit] = Delta_restricted[key][0]
						self.Delta_k[key][self.K] = Delta_restricted[key][1]
						self.Delta_k_inv[key][zsplit] = Delta_restricted_inv[key][0]
						self.Delta_k_inv[key][self.K] = Delta_restricted_inv[key][1]
						self.Delta_k_det[key][zsplit] = Delta_restricted_det[key][0]
						self.Delta_k_det[key] = np.append(self.Delta_k_det[key],Delta_restricted_det[key][1])
					## Update K 
					self.K += 1
			else:
				if not self.directed or self.coclust:
					if self.coclust:
						self.z[sr] = np.copy(z_prop)
						self.nk[sr][zmerge] += self.nk[sr][zlost]
						self.nunk[sr][zmerge] = self.nu0[sr] + self.nk[sr][zmerge]
						self.kappank[sr][zmerge] = self.kappa0[sr] + self.nk[sr][zmerge]
						self.sum_x[sr][zmerge] += self.sum_x[sr][zlost]
						if self.equal_var:
							self.v[sr] = v_prop
							self.vk[sr] = vk_prop
							self.lambdank[sr] = lambdank_prop
							self.sigmank[sr] = sigmank_prop
						else:
							self.lambdank[sr][zmerge] = self.lambda0[sr] + self.nk[sr][zmerge]
							self.sigmank[sr][zmerge] += self.sigmank[sr][zlost] - self.prior_sigma[sr][self.d:]
						self.mean_k[sr][zmerge] = (self.prior_sum[sr][:self.d] + self.sum_x[sr][zmerge]) / self.kappank[sr][zmerge]
						self.squared_sum_x[sr][zmerge] += self.squared_sum_x[sr][zlost]
						self.Delta_k[sr][zmerge] = self.Delta0[sr][:self.d,:self.d] + self.squared_sum_x[sr][zmerge] + self.prior_outer[sr][:self.d,:self.d] - \
							self.kappank[sr][zmerge] * np.outer(self.mean_k[sr][zmerge],self.mean_k[sr][zmerge])
						self.Delta_k_inv[sr][zmerge] = self.kappank[sr][zmerge] / (self.kappank[sr][zmerge] + 1.0) * inv(self.Delta_k[sr][zmerge])
						self.Delta_k_det[sr][zmerge] = slogdet(self.Delta_k[sr][zmerge])[1]
						## Delete components from vectors and dictionaries
						self.nk[sr] = np.delete(self.nk[sr],zlost)
						self.nunk[sr] = np.delete(self.nunk[sr],zlost)
						self.kappank[sr] = np.delete(self.kappank[sr],zlost)
						self.sum_x[sr] = np.delete(self.sum_x[sr],zlost,axis=0)
						if not self.equal_var:
							self.lambdank[sr] = np.delete(self.lambdank[sr],zlost)
							self.sigmank[sr] = np.delete(self.sigmank[sr],zlost,axis=0)
						self.mean_k[sr] = np.delete(self.mean_k[sr],zlost,axis=0)
						## Remove the element corresponding to the empty cluster
						del self.squared_sum_x[sr][zlost]
						del self.Delta_k[sr][zlost]
						del self.Delta_k_inv[sr][zlost]
						self.Delta_k_det[sr] = np.delete(self.Delta_k_det[sr],zlost)
						## Update the dictionaries and the allocations z
						if zlost != self.K[sr]-1:
							for k in range(zlost,self.K[sr]-1):
								self.squared_sum_x[sr][k] = self.squared_sum_x[sr][k+1]
								self.Delta_k[sr][k] = self.Delta_k[sr][k+1]
								self.Delta_k_inv[sr][k] = self.Delta_k_inv[sr][k+1]
							## Remove the final term
							del self.squared_sum_x[sr][self.K[sr]-1]
							del self.Delta_k[sr][self.K[sr]-1]
							del self.Delta_k_inv[sr][self.K[sr]-1] 
						## Update K
						self.K[sr] -= 1
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
				else:
					self.z = np.copy(z_prop)
					self.nk[zmerge] += self.nk[zlost]
					for key in ['s','r']:
						self.nunk[key][zmerge] = self.nu0[key] + self.nk[zmerge]
						self.kappank[key][zmerge] = self.kappa0[key] + self.nk[zmerge]
						self.sum_x[key][zmerge] += self.sum_x[key][zlost]
					if self.equal_var:
						self.v = v_prop
						self.vk = vk_prop
						for key in ['s','r']:
							self.lambdank[key] = lambdank_prop[key]
							self.sigmank[key] = sigmank_prop[key]
					else:
						for key in ['s','r']:
							self.lambdank[key][zmerge] = self.lambda0[key] + self.nk[zmerge]
							self.sigmank[key][zmerge] += self.sigmank[key][zlost] - self.prior_sigma[key][self.d:]
					for key in ['s','r']:	
						self.mean_k[key][zmerge] = (self.prior_sum[key][:self.d] + self.sum_x[key][zmerge]) / self.kappank[key][zmerge]
						self.squared_sum_x[key][zmerge] += self.squared_sum_x[key][zlost]
						self.Delta_k[key][zmerge] = self.Delta0[key][:self.d,:self.d] + self.squared_sum_x[key][zmerge] + self.prior_outer[key][:self.d,:self.d] - \
							self.kappank[key][zmerge] * np.outer(self.mean_k[key][zmerge],self.mean_k[key][zmerge])
						self.Delta_k_inv[key][zmerge] = self.kappank[key][zmerge] / (self.kappank[key][zmerge] + 1.0) * inv(self.Delta_k[key][zmerge])
						self.Delta_k_det[key][zmerge] = slogdet(self.Delta_k[key][zmerge])[1]
					## Delete components from vectors and dictionaries
					self.nk = np.delete(self.nk,zlost)
					for key in ['s','r']:
						self.nunk[key] = np.delete(self.nunk[key],zlost)
						self.kappank[key] = np.delete(self.kappank[key],zlost)
						self.sum_x[key] = np.delete(self.sum_x[key],zlost,axis=0)
						if not self.equal_var:
							self.lambdank[key] = np.delete(self.lambdank[key],zlost)
							self.sigmank[key] = np.delete(self.sigmank[key],zlost,axis=0)
						self.mean_k[key] = np.delete(self.mean_k[key],zlost,axis=0)
						## Remove the element corresponding to the empty cluster
						del self.squared_sum_x[key][zlost]
						del self.Delta_k[key][zlost]
						del self.Delta_k_inv[key][zlost]
						self.Delta_k_det[key] = np.delete(self.Delta_k_det[key],zlost)
						## Update the dictionaries and the allocations z
						if zlost != self.K-1:
							for k in range(zlost,self.K-1):
								self.squared_sum_x[key][k] = self.squared_sum_x[key][k+1]
								self.Delta_k[key][k] = self.Delta_k[key][k+1]
								self.Delta_k_inv[key][k] = self.Delta_k_inv[key][k+1]
							## Remove the final term
							del self.squared_sum_x[key][self.K-1]
							del self.Delta_k[key][self.K-1]
							del self.Delta_k_inv[key][self.K-1] 
					## Update K
					self.K -= 1
		if verbose:
			print 'Proposal: ' + ['MERGE','SPLIT'][split] + '\t' + 'Accepted: ' + str(accept)
		#if (self.sigmank < 0).any() or (self.lambdank < 0).any():
		#	raise ValueError('Error with sigma and/or lambda')
		#vv = Counter(self.v)
		#vvv = np.array([vv[key] for key in range(self.H)])
		#if (vvv != self.vk).any():
		#	raise ValueError('Error with vs')
		return None

	########################################################
	### d. Propose to add (or delete) an empty component ###
	########################################################
	def propose_empty(self,verbose=False):
		## For coclustering: Gibbs sample at random sources or receivers
		if self.coclust:
			sr = np.random.choice(['s','r'])
		## Propose to add or remove an empty cluster
		if (self.K[sr] if self.coclust else self.K) == 1:
			K_prop = 2
		elif (self.K[sr] == self.n[sr]) if self.coclust else (self.K == self.n):
			K_prop = self.n - 1
		else:
			K_prop = np.random.choice([(self.K[sr] if self.coclust else self.K)-1, (self.K[sr] if self.coclust else self.K)+1])
		## Assign values to the variable remove
		if K_prop < (self.K[sr] if self.coclust else self.K):
			remove = True
		else:
			remove = False
		## If there are no empty clusters and K_prop = K-1, reject the proposal
		if not ((self.nk[sr] if self.coclust else self.nk) == 0).any() and K_prop < (self.K[sr] if self.coclust else self.K):
			if verbose:
				print 'Proposal: ' + 'REMOVE' + '\t' + 'Accepted: ' + 'False'
			return None
		## Propose a new vector of cluster allocations
		if remove:
			## Delete empty cluster with largest index (or sample at random)
			ind_delete = np.random.choice(np.where((self.nk[sr] if self.coclust else self.nk) == 0)[0])##[-1]
			nk_prop = np.delete(self.nk[sr] if self.coclust else self.nk,ind_delete)
			## Remove the second order cluster
			if self.equal_var:
				vk_prop = np.copy(self.vk[sr] if self.coclust else self.vk)
				vk_prop[self.v[sr][ind_delete] if self.coclust else self.v[ind_delete]] -= 1
				v_prop = np.delete(self.v[sr] if self.coclust else self.v, ind_delete)
		else:
			nk_prop = np.append(self.nk[sr] if self.coclust else self.nk, 0)
			## Propose the new second order cluster
			if self.equal_var:
				v_prop = np.append(self.v[sr] if self.coclust else self.v, np.random.choice(self.H[sr] if self.coclust else self.H))
				vk_prop = np.copy(self.vk[sr] if self.coclust else self.vk)
				vk_prop[v_prop[-1]] += 1.0 
		## Common term for the acceptance probability
		if self.coclust:
			accept_ratio = self.K[sr] * gammaln(float(self.alpha[sr]) / self.K[sr]) - K_prop * gammaln(float(self.alpha[sr]) / K_prop) + \
								np.sum(gammaln(nk_prop + float(self.alpha[sr]) / K_prop)) - np.sum(gammaln(self.nk[sr] + float(self.alpha[sr]) / self.K[sr])) + \
								(K_prop - self.K[sr]) * log(1 - self.omega) * log(.5) * int(self.K[sr] == 1) - log(.5) * int(self.K[sr] == self.n[sr])
		else:
			accept_ratio = self.K*gammaln(float(self.alpha) / self.K) - K_prop * gammaln(float(self.alpha) / K_prop) + \
								np.sum(gammaln(nk_prop + float(self.alpha) / K_prop)) - np.sum(gammaln(self.nk + float(self.alpha) / self.K)) + \
								(K_prop - self.K) * log(1 - self.omega) * log(.5) * int(self.K == 1) - log(.5) * int(self.K == self.n)
		## Add the equal variance component
		if self.equal_var:
			if self.coclust:
				accept_ratio += gammaln(self.K[sr] + self.beta[sr]) - gammaln(K_prop + self.beta[sr]) - np.sum(gammaln(self.vk[sr] + \
					float(self.beta[sr]) / self.H[sr])) + np.sum(gammaln(vk_prop + float(self.beta[sr]) / self.H[sr]))
				if remove:
					accept_ratio += log(1.0 / self.H[sr]) - log(1.0 / (np.sum(self.nk[sr] == 0)))
				else:
					accept_ratio += log(1.0 / (np.sum(self.nk[sr] == 0)+1)) - log(1.0 / self.H[sr])
			else:
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
			if K_prop > (self.K[sr] if self.coclust else self.K):
				if self.coclust:
					self.nk[sr] = nk_prop
					self.kappank[sr] = np.append(self.kappank[sr],self.kappa0[sr])
					self.nunk[sr] = np.append(self.nunk[sr],self.nu0[sr])
					if self.equal_var:
						self.v[sr] = v_prop
						self.vk[sr] = vk_prop
					else:
						self.lambdank[sr] = np.append(self.lambdank[sr],self.lambda0[sr])
						self.sigmank[sr] = np.vstack((self.sigmank[sr],self.prior_sigma[sr][self.d:] * np.ones((1,self.m-self.d))))
					self.sum_x[sr] = np.vstack((self.sum_x[sr],np.zeros((1,self.d))))
					self.mean_k[sr] = np.vstack((self.mean_k[sr],self.prior_sum[sr][:self.d]/self.kappa0[sr]))
					self.squared_sum_x[sr][self.K[sr]] = np.zeros((self.d,self.d))
					self.Delta_k[sr][self.K[sr]] = self.Delta0[sr][:self.d,:self.d] + self.prior_outer[sr][:self.d,:self.d]
					self.Delta_k_inv[sr][self.K[sr]] = self.kappank[sr][self.K[sr]] / (self.kappank[sr][self.K[sr]] + 1.0) * inv(self.Delta_k[sr][self.K[sr]])
					self.Delta_k_det[sr] = np.append(self.Delta_k_det[sr],slogdet(self.Delta_k[sr][self.K[sr]])[1])
				else:
					self.nk = nk_prop
					if self.directed:
						for key in ['s','r']:
							self.kappank[key] = np.append(self.kappank[key],self.kappa0[key])
							self.nunk[key] = np.append(self.nunk[key],self.nu0[key])
					else:
						self.kappank = np.append(self.kappank,self.kappa0)
						self.nunk = np.append(self.nunk,self.nu0)
					if self.equal_var:
						self.v = v_prop
						self.vk = vk_prop
					else:
						if self.directed:
							for key in ['s','r']:
								self.lambdank[key] = np.append(self.lambdank[key],self.lambda0[key])
								self.sigmank[key] = np.vstack((self.sigmank[key],self.prior_sigma[key][self.d:] * np.ones((1,self.m-self.d))))
						else:
							self.lambdank = np.append(self.lambdank,self.lambda0)
							self.sigmank = np.vstack((self.sigmank,self.prior_sigma[self.d:] * np.ones((1,self.m-self.d))))
					if self.directed:
						for key in ['s','r']:
							self.sum_x[key] = np.vstack((self.sum_x[key],np.zeros((1,self.d))))
							self.mean_k[key] = np.vstack((self.mean_k[key],self.prior_sum[key][:self.d]/self.kappa0[key]))
							self.squared_sum_x[key][self.K] = np.zeros((self.d,self.d))
							self.Delta_k[key][self.K] = self.Delta0[key][:self.d,:self.d] + self.prior_outer[key][:self.d,:self.d]
							self.Delta_k_inv[key][self.K] = self.kappank[key][self.K] / (self.kappank[key][self.K] + 1.0) * inv(self.Delta_k[key][self.K])
							self.Delta_k_det[key] = np.append(self.Delta_k_det[key],slogdet(self.Delta_k[key][self.K])[1])
					else:
						self.sum_x = np.vstack((self.sum_x,np.zeros((1,self.d))))
						self.mean_k = np.vstack((self.mean_k,self.prior_sum[:self.d]/self.kappa0))
						self.squared_sum_x[self.K] = np.zeros((self.d,self.d))
						self.Delta_k[self.K] = self.Delta0[:self.d,:self.d] + self.prior_outer[:self.d,:self.d]
						self.Delta_k_inv[self.K] = self.kappank[self.K] / (self.kappank[self.K] + 1.0) * inv(self.Delta_k[self.K])
						self.Delta_k_det = np.append(self.Delta_k_det,slogdet(self.Delta_k[self.K])[1])
			else:
				if self.coclust:
					self.nk[sr] = nk_prop
					self.kappank[sr] = np.delete(self.kappank[sr],ind_delete)
					self.nunk[sr] = np.delete(self.nunk[sr],ind_delete)
					if self.equal_var:
						self.vk[sr] = vk_prop
						self.v[sr] = v_prop
					else:
						self.lambdank[sr] = np.delete(self.lambdank[sr],ind_delete)
						self.sigmank[sr] = np.delete(self.sigmank[sr],ind_delete,axis=0)
					self.sum_x[sr] = np.delete(self.sum_x[sr],ind_delete,axis=0)
					self.mean_k[sr] = np.delete(self.mean_k[sr],ind_delete,axis=0)
					## Remove the element corresponding to the empty cluster
					del self.squared_sum_x[sr][ind_delete]
					del self.Delta_k[sr][ind_delete]
					del self.Delta_k_inv[sr][ind_delete]
					self.Delta_k_det[sr] = np.delete(self.Delta_k_det[sr],ind_delete)
					## Update the dictionaries and the allocations z
					if ind_delete != K_prop:
						for k in range(ind_delete,K_prop):
							self.squared_sum_x[sr][k] = self.squared_sum_x[sr][k+1]
							self.Delta_k[sr][k] = self.Delta_k[sr][k+1]
							self.Delta_k_inv[sr][k] = self.Delta_k_inv[sr][k+1]
							self.z[sr][self.z[sr] == k+1] = k
						## Remove the final term
						del self.squared_sum_x[sr][K_prop]
						del self.Delta_k[sr][K_prop]
						del self.Delta_k_inv[sr][K_prop] 
				else:
					self.nk = nk_prop
					if self.directed:
						for key in ['s','r']:
							self.kappank[key] = np.delete(self.kappank[key],ind_delete)
							self.nunk[key] = np.delete(self.nunk[key],ind_delete)
					else:
						self.kappank = np.delete(self.kappank,ind_delete)
						self.nunk = np.delete(self.nunk,ind_delete)
					if self.equal_var:
						self.vk = vk_prop
						self.v = v_prop
					else:
						if self.directed:
							for key in ['s','r']:
								self.lambdank[key] = np.delete(self.lambdank[key],ind_delete)
								self.sigmank[key] = np.delete(self.sigmank[key],ind_delete,axis=0)
						else:
							self.lambdank = np.delete(self.lambdank,ind_delete)
							self.sigmank = np.delete(self.sigmank,ind_delete,axis=0)
					if self.directed:
						for key in ['s','r']:
							self.sum_x[key] = np.delete(self.sum_x[key],ind_delete,axis=0)
							self.mean_k[key] = np.delete(self.mean_k[key],ind_delete,axis=0)
							## Remove the element corresponding to the empty cluster
							del self.squared_sum_x[key][ind_delete]
							del self.Delta_k[key][ind_delete]
							del self.Delta_k_inv[key][ind_delete]
							self.Delta_k_det[key] = np.delete(self.Delta_k_det[key],ind_delete)
							## Update the dictionaries and the allocations z
							if ind_delete != K_prop:
								for k in range(ind_delete,K_prop):
									self.squared_sum_x[key][k] = self.squared_sum_x[key][k+1]
									self.Delta_k[key][k] = self.Delta_k[key][k+1]
									self.Delta_k_inv[key][k] = self.Delta_k_inv[key][k+1]
								## Remove the final term
								del self.squared_sum_x[key][K_prop]
								del self.Delta_k[key][K_prop]
								del self.Delta_k_inv[key][K_prop] 
						if ind_delete != K_prop:
							for k in range(ind_delete,K_prop):
								self.z[self.z == k+1] = k
					else:
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
			if self.coclust:
				self.K[sr] = K_prop
			else:
				self.K = K_prop
		if verbose:
			print 'Proposal: ' + ['ADD','REMOVE'][remove] + '\t' + 'Accepted: ' + str(accept)
		#if (self.sigmank < 0).any() or (self.lambdank < 0).any():
		#	raise ValueError('Error with sigma and/or lambda')
		#vv = Counter(self.v)
		#vvv = np.array([vv[key] for key in range(self.H)])
		#if (vvv != self.vk).any():
		#	raise ValueError('Error with vs')
		return None

	#####################################################################
	### Compute marginal likelihoods for all the possible values of d ###
	#####################################################################
	## Utility function: calculates the marginal likelihood for all the possible values of d given a set of allocations z
	def marginal_likelihoods_dimension(self):
		## Initialise sum, mean and sum of squares
		if self.directed:
			sums = {}; squares = {}; means = {}; Deltank = {}; Deltank_det = {}; Delta0_det = {}
			for key in ['s','r']:
				sums[key] = np.zeros((self.K[key] if self.coclust else self.K,self.m))
				squares[key] = np.zeros((self.sigmank[key].shape[0],self.m))
				means[key] = np.zeros((self.K[key] if self.coclust else self.K,self.m))
				Deltank[key] = {}
				Deltank_det[key] = np.zeros((self.K[key] if self.coclust else self.K,self.m+1))
				for k in range(self.K[key] if self.coclust else self.K):
					x = self.X[key][(self.z[key] if self.coclust else self.z) == k]
					sums[key][k] = x.sum(axis=0)
					means[key][k] = (sums[key][k] + self.prior_sum[key]) / ((self.nk[key][k] if self.coclust else self.nk[k]) + self.kappa0[key])
					if not self.equal_var:
						squares[key][k] = self.prior_sigma[key] + (x ** 2).sum(axis=0)
					Deltank[key][k] = self.Delta0[key] + np.dot(x.T,x) + self.prior_outer[key] - (self.kappa0[key] + \
						(self.nk[key][k] if self.coclust else self.nk[k])) * np.outer(means[key][k],means[key][k])
					for d in range(self.m+1):
						Deltank_det[key][k,d] = slogdet(Deltank[key][k][:d,:d])[1] 
				if self.equal_var:
					for h in range(self.H[key] if self.coclust else self.H):
						x = self.X[key][(self.v[key][self.z[key]] if self.coclust else self.v[self.z]) == h]
						squares[key][h] = self.prior_sigma[key] + (x ** 2).sum(axis=0)
				## Calculate the determinants sequentially for the prior
				Delta0_det[key] = np.zeros(self.m+1)
				for i in range(self.m+1):
					Delta0_det[key][i] = slogdet(self.Delta0[key][:i,:i])[1]
		else:
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
				if self.directed:
					for key in ['s','r']:
						for k in range(self.sigmank[key].shape[0]):
							self.mlik[d] += np.sum(.5 * self.lambda0[key] * log(self.prior_sigma[key][d:]) - \
								.5 * self.lambdank[key][k] * log(squares[key][k,d:]))
							self.mlik[d] += (self.m - d) * (gammaln(.5 * self.lambdank[key][k]) - gammaln(.5 * self.lambda0[key]))
							if self.equal_var:
								self.mlik[d] += -.5 * (np.sum(self.nk[key][self.v[key] == k]) if self.coclust else np.sum(self.nk[self.v == k])) * \
									(self.m - d) * log(np.pi)
							else:
								self.mlik[d] += -.5 * (self.nk[key][k] if self.coclust else self.nk[k]) * (self.m - d) * log(np.pi)
				else:
					for k in range(self.sigmank.shape[0]):
						self.mlik[d] += np.sum(.5 * self.lambda0 * log(self.prior_sigma[d:]) - .5 * self.lambdank[k] * log(squares[k,d:]))
						self.mlik[d] += (self.m - d) * (gammaln(.5 * self.lambdank[k]) - gammaln(.5 * self.lambda0))
						if self.equal_var:
							self.mlik[d] += -.5 * np.sum(self.nk[self.v == k]) * (self.m - d) * log(np.pi)
						else:
							self.mlik[d] += -.5 * self.nk[k] * (self.m - d) * log(np.pi)
			else:
				self.mlik[d] = 0
			## Calculate the marginal likelihood for the community allocations
			if d != 0: 
				if self.directed:
					for key in ['s','r']:
						for k in range(self.K[key] if self.coclust else self.K):
							self.mlik[d] += -.5 * (self.nk[key][k] if self.coclust else self.nk[k]) * d * log(np.pi) + \
								.5 * d * (log(self.kappa0[key]) - log((self.nk[key][k] if self.coclust else self.nk[k]) + self.kappa0[key])) + \
								.5 * (self.nu0[key] + d - 1) * Delta0_det[key][d] - .5 * (self.nu0[key] + \
								(self.nk[key][k] if self.coclust else self.nk[k]) + d - 1) * Deltank_det[key][k,d]
							self.mlik[d] += np.sum([(gammaln(.5 * (self.nu0[key] + (self.nk[key][k] if self.coclust else self.nk[k]) + d - i)) - \
								gammaln(.5 * (self.nu0[key] + d - i))) for i in range(1,d+1)])
				else:
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
		## For coclustering: Gibbs sample at random sources or receivers
		if self.coclust:
			sr = np.random.choice(['s','r'])
		## Update the second order allocations in random order
		for j in np.random.permutation(range(self.K[sr] if self.coclust else self.K)):
			vold = self.v[sr][j] if self.coclust else self.v[j]
			## Update parameters of the distribution
			if self.coclust:
				positions = self.X[sr][self.z[sr] == j,self.d:]
				self.vk[sr][vold] -= 1.0
				self.lambdank[sr][vold] -= self.nk[sr][j]
				self.sigmank[sr][vold] -= np.sum(positions ** 2,axis=0)
				lambdank_resamp = np.copy(self.lambdank[sr])
				sigmank_resamp = np.copy(self.sigmank[sr])
			else:
				if self.directed:
					positions = {}; lambdank_resamp = {}; sigmank_resamp = {}
					self.vk[vold] -= 1.0
					for key in ['s','r']:
						positions[key] = self.X[key][self.z == j,self.d:]
						self.lambdank[key][vold] -= self.nk[j]
						self.sigmank[key][vold] -= np.sum(positions[key] ** 2,axis=0)
						lambdank_resamp[key] = np.copy(self.lambdank[key])
						sigmank_resamp[key] = np.copy(self.sigmank[key])
				else:
					positions = self.X[self.z == j,self.d:]
					self.vk[vold] -= 1.0
					self.lambdank[vold] -= self.nk[j]
					self.sigmank[vold] -= np.sum(positions ** 2,axis=0)
					lambdank_resamp = np.copy(self.lambdank)
					sigmank_resamp = np.copy(self.sigmank)
			## Calculate the probabilities sequentially 
			prob_v = np.zeros(self.H[sr] if self.coclust else self.H)
			for i in np.random.permutation(range(positions['s'].shape[0]) if self.directed and not self.coclust else positions.shape[0]):
				if self.directed and not self.coclust:
					for key in ['s','r']:
						pos = positions[key][i]
						prob_v += np.array([np.sum(t.logpdf(pos,df=lambdank_resamp[key][h],loc=0,scale=sqrt(sigmank_resamp[key][h]/lambdank_resamp[key][h]))) \
							for h in range(self.H)])
						lambdank_resamp[key] += 1
						sigmank_resamp[key] += (pos ** 2)
				else:
					pos = positions[i]
					prob_v += np.array([np.sum(t.logpdf(pos,df=lambdank_resamp[h],loc=0,scale=sqrt(sigmank_resamp[h]/lambdank_resamp[h]))) \
						for h in range(self.H[sr] if self.coclust else self.H)])
					lambdank_resamp += 1
					sigmank_resamp += (pos ** 2)
			## Calculate last component of the probability of allocation
			if self.coclust:
				prob_v += log(self.vk[sr] + self.beta[sr] / self.H[sr]) - log(self.K[sr] - 1 + self.beta[sr])
			else:
				prob_v += log(self.vk + self.beta / self.H) - log(self.K - 1 + self.beta)
			prob_v = exp(prob_v-max(prob_v)) / np.sum(exp(prob_v-np.max(prob_v)))
			## Raise error if nan probabilities are computed or negative values appear in lambda or sigma
			#if (lambdank_resamp <= 0.0).any():
			#	print lambdank_resamp
			#	raise ValueError('Problem with lambda')
			#if (sigmank_resamp <= 0.0).any():
			#	raise ValueError('Problem with sigma')
			#if np.isnan(prob_v).any():
			#	raise ValueError("Error in the allocation probabilities.")
			## Sample the new community allocation
			vnew = int(np.random.choice(range(self.H[sr] if self.coclust else self.H),p=prob_v))
			if self.coclust:
				self.v[sr][j] = vnew
				## Update the Student's t parameters accordingly
				self.lambdank[sr][vnew] += self.nk[sr][j]
				self.sigmank[sr][vnew] += np.sum(positions ** 2, axis=0)
				self.vk[sr][vnew] += 1.0
			else:
				self.v[j] = vnew
				## Update the Student's t parameters accordingly
				if self.directed:
					for key in ['s','r']:
						self.lambdank[key][vnew] += self.nk[j]
						self.sigmank[key][vnew] += np.sum(positions[key] ** 2, axis=0)
				else:
					self.lambdank[vnew] += self.nk[j]
					self.sigmank[vnew] += np.sum(positions ** 2, axis=0)
				self.vk[vnew] += 1.0
		#if (self.sigmank < 0).any() or (self.lambdank < 0).any():
		#	raise ValueError('Error with sigma and/or lambda')
		#vv = Counter(self.v)
		#vvv = np.array([vv[key] for key in range(self.H)])
		#if (vvv != self.vk).any():
		#	raise ValueError('Error with vs')
		return None ## the global variables are updated within the function, no need to return anything

	###############################################################
	### 2b. Propose to split (or merge) two second order groups ###
	###############################################################
	def split_merge_second_order(self,verbose=False):
		## Stop if equal_var is set to false
		if not self.equal_var:
			raise ValueError('equal_var is set to false')
		## For coclustering: Gibbs sample at random sources or receivers
		if self.coclust:
			sr = np.random.choice(['s','r'])
		## Randomly choose two indices
		if (self.K[sr] if self.coclust else self.K) > 1:
			i,j = np.random.choice(self.K[sr] if self.coclust else self.K,size=2,replace=False)
		else:
			if verbose:
				print 'Proposal: ' + 'SPLIT-MERGE 2nd order' + '\t' + 'Accepted: ' + 'False'
			return None
		## Propose a split or merge move according to the sampled values
		if (self.v[sr][i] == self.v[sr][j]) if self.coclust else (self.v[i] == self.v[j]):
			split = True
			if not self.coclust:
				vsplit = self.v[i]
				## Obtain the indices that must be re-allocated
				S = np.delete(range(self.K),[i,j])[np.delete(self.v == vsplit,[i,j])]
				v_prop = np.copy(self.v)
				## Move j to a new cluster
				v_prop[j] = self.H
			else:
				vsplit = self.v[sr][i]
				## Obtain the indices that must be re-allocated
				S = np.delete(range(self.K[sr]),[i,j])[np.delete(self.v[sr] == vsplit,[i,j])]
				v_prop = np.copy(self.v[sr])
				## Move j to a new cluster
				v_prop[j] = self.H[sr]
		else:
			split = False
			## Choose to merge to the cluster with minimum index (vmerge) and remove the cluster with maximum index (vlost)
			if not self.coclust:
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
			else:
				vmerge = min([self.v[sr][i],self.v[sr][j]])
				vj = self.v[sr][j]
				vlost = max([self.v[sr][i],self.v[sr][j]])
				v_prop = np.copy(self.v[sr])
				v_prop[self.v[sr] == vlost] = vmerge
				if vlost != (self.H[sr]-1):
					for k in range(vlost,self.H[sr]-1):
						v_prop[self.v[sr] == k+1] = k
				S = np.delete(range(self.K[sr]),[i,j])[np.delete((self.v[sr] == vmerge) + (self.v[sr] == vlost),[i,j])]
		## Initialise the proposal ratio
		prop_ratio = 0
		## Construct vectors for sequential posterior predictives
		vk_rest = np.ones(2,int)
		if split:
			if self.coclust:
				lambda_rest = self.lambda0[sr] + np.ones(2)
				sigma_rest = np.zeros((2,self.m-self.d))
				sigma_rest[0] = self.prior_sigma[sr][self.d:] + np.sum(self.X[sr][v_prop[self.z[sr]] == vsplit,self.d:] ** 2, axis=0)
				sigma_rest[1] = self.prior_sigma[sr][self.d:] + np.sum(self.X[sr][v_prop[self.z[sr]] == self.H[sr],self.d:] ** 2, axis=0)
			else:
				if self.directed:
					lambda_rest = {}
					sigma_rest = {}
					for key in ['s','r']:
						lambda_rest[key] = self.lambda0[key] + np.ones(2)
						sigma_rest[key] = np.zeros((2,self.m-self.d))
						sigma_rest[key][0] = self.prior_sigma[key][self.d:] + np.sum(self.X[key][v_prop[self.z] == vsplit,self.d:] ** 2, axis=0)
						sigma_rest[key][1] = self.prior_sigma[key][self.d:] + np.sum(self.X[key][v_prop[self.z] == self.H,self.d:] ** 2, axis=0)
				else:
					lambda_rest = self.lambda0 + np.ones(2)
					sigma_rest = np.zeros((2,self.m-self.d))
					sigma_rest[0] = self.prior_sigma[self.d:] + np.sum(self.X[v_prop[self.z] == vsplit,self.d:] ** 2, axis=0)
					sigma_rest[1] = self.prior_sigma[self.d:] + np.sum(self.X[v_prop[self.z] == self.H,self.d:] ** 2, axis=0)
		else:
			if self.coclust:
				lambda_rest = self.lambda0[sr] + np.ones(2)
				sigma_rest = np.zeros((2,self.m-self.d))
				sigma_rest[0] = self.prior_sigma[sr][self.d:] + np.sum(self.X[sr][self.v[sr][self.z[sr]] == vmerge,self.d:] ** 2, axis=0)
				sigma_rest[1] = self.prior_sigma[sr][self.d:] + np.sum(self.X[sr][self.v[sr][self.z[sr]] == vlost,self.d:] ** 2, axis=0)
			else:
				if self.directed:
					lambda_rest = {}
					sigma_rest = {}
					for key in ['s','r']:
						lambda_rest[key] = self.lambda0[key] + np.ones(2)
						sigma_rest[key] = np.zeros((2,self.m-self.d))
						sigma_rest[0] = self.prior_sigma[key][self.d:] + np.sum(self.X[key][self.v[self.z] == vmerge,self.d:] ** 2, axis=0)
						sigma_rest[1] = self.prior_sigma[key][self.d:] + np.sum(self.X[key][self.v[self.z] == vlost,self.d:] ** 2, axis=0)
				else:
					lambda_rest = self.lambda0 + np.ones(2)
					sigma_rest = np.zeros((2,self.m-self.d))
					sigma_rest[0] = self.prior_sigma[self.d:] + np.sum(self.X[self.v[self.z] == vmerge,self.d:] ** 2, axis=0)
					sigma_rest[1] = self.prior_sigma[self.d:] + np.sum(self.X[self.v[self.z] == vlost,self.d:] ** 2, axis=0)
		## Randomly permute the indices in S and calculate the sequential allocations 
		for h in np.random.permutation(S):
			## Calculate the predictive probability
			if self.coclust:
				positions = self.X[sr][self.z[sr] == h,self.d:]
			else:
				if self.directed:
					positions = {}
					for key in ['s','r']:
						positions[key] = self.X[key][self.z == h,self.d:]
				else:
					positions = self.X[self.z == h,self.d:]
			## Calculate the new second order allocation of the first cluster after splitting
			if not self.directed or self.coclust: 
				lambda_left = np.copy(lambda_rest)
				sigma_left = np.copy(sigma_rest)
			else:
				lambda_left = {}; sigma_left = {}
				for key in ['s','r']:
					lambda_left[key] = np.copy(lambda_rest[key])
					sigma_left[key] = np.copy(sigma_rest[key])
			prob_v_left = np.zeros(2)
			if (positions['s'].shape[0] if self.directed and not self.coclust else positions.shape[0]) != 0:
				if not self.directed or self.coclust:
					for pos in np.random.permutation(positions):
						prob_v_left += np.array([np.sum(t.logpdf(pos,df=lambda_left[q],loc=0,scale=sqrt(sigma_left[q]/lambda_left[q]))) for q in range(2)])
						lambda_left += 1.0
						for q in range(2): 
							sigma_left[q] += (pos ** 2)
				else:
					for pos_ind in np.random.permutation(range(positions['s'].shape[0])):
						for key in ['s','r']:
							prob_v_left += np.array([np.sum(t.logpdf(positions[key][pos_ind],df=lambda_left[key][q],loc=0, \
									scale=sqrt(sigma_left[key][q]/lambda_left[key][q]))) for q in range(2)])
							lambda_left[key] += 1.0
							for q in range(2): 
								sigma_left[key][q] += (positions[key][pos_ind] ** 2)
			## Resample the second order cluster allocation
			pred_prob = exp(prob_v_left-max(prob_v_left)) / np.sum(exp(prob_v_left-np.max(prob_v_left)))
			## Calculate second order cluster allocation (restricted posterior predictive)
			if split:
				## Sample the new value
				vnew = np.random.choice(2,p=pred_prob)
				## Update proposal ratio
				prop_ratio += log(pred_prob[vnew])
				## Update proposed h
				v_prop[h] = [vsplit,self.H[sr] if self.coclust else self.H][vnew]
			else:
				## Determine the new value deterministically
				vnew = int((self.v[sr][h] if self.coclust else self.v[h]) == vj)
				## Update proposal ratio in the imaginary split
				prop_ratio += log(pred_prob[vnew])
				if pred_prob[vnew] == 0.0:
					warnings.warn('Imaginary split yields impossible outcome: merge proposal automatically rejected')
			## Update parameters 
			if not self.directed or self.coclust:
				vk_rest[vnew] += 1.0
				lambda_rest[vnew] += positions.shape[0]
				sigma_rest[vnew] += np.sum(positions ** 2, axis=0)
			else:
				vk_rest[vnew] += 1.0
				for key in ['s','r']:
					lambda_rest[key][vnew] += positions[key].shape[0]
					sigma_rest[key][vnew] += np.sum(positions[key] ** 2, axis=0)
		## Proposed values of lambdank and sigmank
		H_prop = (self.H[sr] if self.coclust else self.H) + (1 if split else -1)
		if H_prop > (self.K[sr] if self.coclust else self.K):
			if verbose:
				print 'Proposal: ' + ['MERGE 2nd order','SPLIT 2nd order'][split] + '\t' + 'Accepted: ' + 'False'
			return None
		if not self.directed or self.coclust:
			vk_prop = np.zeros(H_prop)
			lambdank_prop = np.zeros(H_prop)
			sigmank_prop = np.zeros((H_prop,self.m-self.d))
			for h in range(H_prop):
				## Second order cluster counts
				vk_prop[h] = np.sum(v_prop == h)
				lambdank_prop[h] = (self.lambda0[sr] + np.sum(self.nk[sr][v_prop==h]))  if self.coclust else (self.lambda0 + np.sum(self.nk[v_prop==h])) 
				sigmank_prop[h] = (self.prior_sigma[sr][self.d:] + np.sum(self.X[sr][v_prop[self.z[sr]]==h, self.d:]**2, \
						axis=0)) if self.coclust else (self.prior_sigma[self.d:] + np.sum(self.X[v_prop[self.z]==h, self.d:]**2, axis=0))
		else:
			vk_prop = np.zeros(H_prop)
			for h in range(H_prop):
				vk_prop[h] = np.sum(v_prop == h)
			lambdank_prop = {}; sigmank_prop = {}
			for key in ['s','r']:
				lambdank_prop[key] = np.zeros(H_prop)
				sigmank_prop[key] = np.zeros((H_prop,self.m-self.d))
				for h in range(H_prop):
				## Second order cluster counts
					lambdank_prop[key][h] = self.lambda0[key] + np.sum(self.nk[v_prop == h])
					sigmank_prop[key][h] = self.prior_sigma[key][self.d:] + np.sum(self.X[key][v_prop[self.z] == h, self.d:]**2, axis=0)
		#### Alternative way to evaluate sigma_prop for the undirected graph
		#if split:
		#	sigmank_prop[vsplit] = sigma_rest[0]
		#	sigmank_prop = np.vstack((sigmank_prop,sigma_rest[1]))
		#else:
		#	sigmank_prop[vmerge] += sigmank_prop[vlost] - self.prior_sigma[self.d:]
		#	sigmank_prop = np.delete(sigmank_prop,vlost,axis=0)
		## Calculate the acceptance probability
		accept_ratio = 0.0
		if self.coclust:
			accept_ratio += np.sum([(self.m - self.d) * (gammaln(.5 * lambdank_prop[h]) - gammaln(.5 * self.lambda0[sr])) + \
							np.sum(.5*self.lambda0[sr]*log(self.prior_sigma[sr][self.d:]) - .5*lambdank_prop[h]*log(sigmank_prop[h])) for h in range(H_prop)])
			accept_ratio -= np.sum([(self.m - self.d) * (gammaln(.5 * self.lambdank[sr][h]) - gammaln(.5 * self.lambda0[sr])) + \
							np.sum(.5*self.lambda0[sr]*log(self.prior_sigma[sr][self.d:]) - .5*self.lambdank[sr][h]*log(self.sigmank[sr][h])) for h in range(self.H[sr])])
			accept_ratio += np.sum(gammaln(vk_prop + self.beta[sr] / self.H[sr])) - np.sum(gammaln(self.vk[sr] + self.beta[sr] / self.H[sr])) + \
							gammaln(self.K[sr] + self.beta[sr]) - gammaln(self.K[sr] + 1 + self.beta[sr])
		else:
			if self.directed:
				for key in ['s','r']:
					accept_ratio += np.sum([(self.m - self.d) * (gammaln(.5 * lambdank_prop[key][h]) - gammaln(.5 * self.lambda0[key])) + \
							np.sum(.5*self.lambda0[key]*log(self.prior_sigma[key][self.d:]) - .5*lambdank_prop[key][h]*log(sigmank_prop[key][h])) for h in range(H_prop)])
					accept_ratio -= np.sum([(self.m - self.d) * (gammaln(.5 * self.lambdank[key][h]) - gammaln(.5 * self.lambda0[key])) + \
							np.sum(.5*self.lambda0[key]*log(self.prior_sigma[key][self.d:]) - .5*self.lambdank[key][h]*log(self.sigmank[key][h])) for h in range(self.H)])
					accept_ratio += np.sum(gammaln(vk_prop + self.beta / self.H)) - np.sum(gammaln(self.vk + self.beta / self.H)) + \
							gammaln(self.K + self.beta) - gammaln(self.K + 1 + self.beta)
			else:
				accept_ratio += np.sum([(self.m - self.d) * (gammaln(.5 * lambdank_prop[h]) - gammaln(.5 * self.lambda0)) + \
							np.sum(.5*self.lambda0*log(self.prior_sigma[self.d:]) - .5*lambdank_prop[h]*log(sigmank_prop[h])) for h in range(H_prop)])
				accept_ratio -= np.sum([(self.m - self.d) * (gammaln(.5 * self.lambdank[h]) - gammaln(.5 * self.lambda0)) + \
							np.sum(.5*self.lambda0*log(self.prior_sigma[self.d:]) - .5*self.lambdank[h]*log(self.sigmank[h])) for h in range(self.H)])
				accept_ratio += np.sum(gammaln(vk_prop + self.beta / self.H)) - np.sum(gammaln(self.vk + self.beta / self.H)) + gammaln(self.K + self.beta) - \
							gammaln(self.K + 1 + self.beta)
		# Prior on H and q function
		accept_ratio += (1 if split else -1) * prop_ratio ##(log(1.0 - self.csi) - prop_ratio)
		## Accept or reject the proposal
		accept = (-np.random.exponential(1) < accept_ratio)
		if accept:
			## Update the stored values
			if self.coclust:
				self.v[sr] = v_prop
				self.vk[sr] = vk_prop
				self.lambdank[sr] = lambdank_prop
				self.sigmank[sr] = sigmank_prop
				## Update H 
				self.H[sr] = H_prop
			else:
				self.v = v_prop
				self.vk = vk_prop
				self.lambdank = lambdank_prop
				self.sigmank = sigmank_prop
				## Update H 
				self.H = H_prop
		if verbose:
			print 'Proposal: ' + ['MERGE 2nd order','SPLIT 2nd order'][split] + '\t' + 'Accepted: ' + str(accept)
		#if (self.sigmank < 0).any() or (self.lambdank < 0).any():
		#	raise ValueError('Error with sigma and/or lambda')
		#vv = Counter(self.v)
		#vvv = np.array([vv[key] for key in range(self.H)])
		#if (vvv != self.vk).any():
		#	raise ValueError('Error with vs')
		return None

	######################################################################
	### 2c. Propose to add (or delete) an empty second order component ###
	######################################################################
	def propose_empty_second_order(self,verbose=False):
		## Stop if equal_var is set to false
		if not self.equal_var:
			raise ValueError('equal_var is set to false')
		## For coclustering: Gibbs sample at random sources or receivers
		if self.coclust:
			sr = np.random.choice(['s','r'])
		## Propose to add or remove an empty cluster
		if (self.H[sr] if self.coclust else self.H) == 1:
			H_prop = 2
		elif (self.H[sr] == self.n[sr]) if self.coclust else (self.H == self.n):
			H_prop = (self.H[sr] if self.coclust else self.H) - 1
		else:
			H_prop = np.random.choice([(self.H[sr] if self.coclust else self.H) - 1, (self.H[sr] if self.coclust else self.H) + 1])
		## Assign values to the variable remove
		if H_prop < (self.H[sr] if self.coclust else self.H):
			remove = True
		else:
			remove = False
		## If there are no empty clusters and K_prop = K-1, reject the proposal
		if not ((self.vk[sr] if self.coclust else self.vk) == 0).any() and H_prop < (self.H[sr] if self.coclust else self.H):
			if verbose:
				print 'Proposal: ' + 'REMOVE 2nd order' + '\t' + 'Accepted: ' + 'False'
			return None
		## If there are no empty clusters and K_prop = K-1, reject the proposal
		if H_prop > (self.K[sr] if self.coclust else self.K):
			if verbose:
				print 'Proposal: ' + 'REMOVE 2nd order' + '\t' + 'Accepted: ' + 'False'
			return None
		## Propose a new vector of cluster allocations
		if remove:
			## Delete empty cluster with largest index (or sample at random)
			ind_delete = np.random.choice(np.where((self.vk[sr] if self.coclust else self.vk) == 0)[0]) ##[-1]
			vk_prop = np.delete(self.vk[sr] if self.coclust else self.vk,ind_delete)
		else:
			## Add an empty second order cluster
			vk_prop = np.append(self.vk[sr] if self.coclust else self.vk,0)
		## Common term for the acceptance probability
		if self.coclust:
			accept_ratio = self.H[sr] * gammaln(float(self.beta[sr]) / self.H[sr]) - H_prop * gammaln(float(self.beta[sr]) / H_prop) + \
							np.sum(gammaln(vk_prop + float(self.beta[sr]) / H_prop)) - np.sum(gammaln(self.vk[sr] + float(self.alpha[sr]) / self.H[sr])) ## + \
							## (H_prop - self.H[sr]) * log(1 - self.csi) * log(.5) * int(self.H[sr] == 1) - log(.5) * int(self.H[sr] == self.n[sr])
		else:
			accept_ratio = self.H * gammaln(float(self.beta) / self.H) - H_prop * gammaln(float(self.beta) / H_prop) + \
							np.sum(gammaln(vk_prop + float(self.beta) / H_prop)) - np.sum(gammaln(self.vk + float(self.alpha) / self.H)) ## + \
							## (H_prop - self.H) * log(1 - self.csi) * log(.5) * int(self.H == 1) - log(.5) * int(self.H == self.n)
		## Accept or reject the proposal
		accept = (-np.random.exponential(1) < accept_ratio)
		## Scale all the values if an empty cluster is added
		if accept:
			if H_prop > (self.H[sr] if self.coclust else self.H):
				if self.coclust:
					self.vk[sr] = vk_prop
					self.lambdank[sr] = np.append(self.lambdank[sr],self.lambda0[sr])
					self.sigmank[sr] = np.vstack((self.sigmank[sr],self.prior_sigma[sr][self.d:] * np.ones((1,self.m-self.d))))
				else:
					self.vk = vk_prop
					if self.directed:
						for key in ['s','r']:
							self.lambdank[key] = np.append(self.lambdank[key],self.lambda0[key])
							self.sigmank[key] = np.vstack((self.sigmank[key],self.prior_sigma[key][self.d:] * np.ones((1,self.m-self.d))))
					else:
						self.lambdank = np.append(self.lambdank,self.lambda0)
						self.sigmank = np.vstack((self.sigmank,self.prior_sigma[self.d:] * np.ones((1,self.m-self.d))))
			else:
				if self.coclust:
					self.vk[sr] = vk_prop
					self.lambdank[sr] = np.delete(self.lambdank[sr],ind_delete)
					self.sigmank[sr] = np.delete(self.sigmank[sr],ind_delete,axis=0)
				else:
					self.vk = vk_prop
					if self.directed:
						for key in ['s','r']:
							self.lambdank[key] = np.delete(self.lambdank[key],ind_delete)
							self.sigmank[key] = np.delete(self.sigmank[key],ind_delete,axis=0)
					else:
						self.lambdank = np.delete(self.lambdank,ind_delete)
						self.sigmank = np.delete(self.sigmank,ind_delete,axis=0)
				if ind_delete != H_prop:
					for h in range(ind_delete,H_prop):
						if self.coclust:
							self.v[sr][self.v[sr] == h+1] = h
						else:
							self.v[self.v == h+1] = h
			if self.coclust:
				self.H[sr] = H_prop
			else:
				self.H = H_prop
		if verbose:
			print 'Proposal: ' + ['ADD 2nd order','REMOVE 2nd order'][remove] + '\t' + 'Accepted: ' + str(accept)
		#if (self.sigmank < 0).any() or (self.lambdank < 0).any():
		#	raise ValueError('Error with sigma and/or lambda')
		#vv = Counter(self.v)
		#vvv = np.array([vv[key] for key in range(self.H)])
		#if (vvv != self.vk).any():
		#	raise ValueError('Error with vs')
		return None

	###################################################################################################
	### Maximise PEAR (posterior expected adjusted Rand index) from the posterior similarity matrix ###
	###################################################################################################
	# Import required packages
	#from rpy2.robjects.packages import importr
	#import rpy2.robjects as ro
	#import rpy2.robjects.numpy2ri
	#from sklearn.cluster import AgglomerativeClustering 
	# Import R's packages
	#base = importr('base')
	#utils = importr('utils')
	#mcclust = importr('mcclust')
	#rpy2.robjects.numpy2ri.activate()
	## The function takes the posterior similarity matrix (psm) as argument (obtained from MCMC chains)
	#def estimate_clustering(psm,k=None):
	#	if k is None:
	#		## If k is not specified, maxpear is used -- from the R package 'mcclust' (Fritsch and Ickstadt, 2009) 
	#		Br = ro.r.matrix(psm, nrow=psm.shape[0], ncol=psm.shape[1])
	#		ro.r.assign("B", Br)
	#		cl = mcclust.maxpear(Br)
	#		clust = np.array(cl[cl.names.index('cl')])
	#	else:
	#		## If k is specified, use agglomerative clustering 
	#		cluster_model = AgglomerativeClustering(n_clusters=int(k), affinity='precomputed', linkage='average') 
	#		clust = cluster_model.fit_predict(1-psm).labels_
	#	return clust

