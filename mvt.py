#!/usr/bin/env python
import sys
from scipy.stats import t
from scipy.special import gammaln
import numpy as np
from numpy import pi,log,sqrt
from numpy.linalg import slogdet,inv

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
