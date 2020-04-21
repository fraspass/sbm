#!/usr/bin/env python
import sys
import numpy as np
from scipy.stats import norm

########################################################################
### Estimate the dimension using the method of Zhu and Ghodsi (2006) ###
########################################################################

### Function to estimate the dimension using the automatic method of Zhu and Ghodsi (2006)
def estimate_dimension(vals,gaps=1):
	## Sort vals if it is not sorted
	vals = np.sort(vals)[::-1]
	## Number of possible dimensions
	m = len(vals)
	## List of profile likelihoods
	profile_lik = []
	## Loop for all the possible values of d
	for d in range(1,m+1):
		mu1 = np.mean(vals[:d]) 
		mu2 = np.mean(vals[d:])
		sigma = np.sqrt(np.sum((vals[:d] - mu1) ** 2) + np.sum((vals[d:] - mu2) ** 2) / (m-1))
		profile_lik += [np.sum(norm.pdf(vals[:d], loc=mu1, scale=sigma)) + np.sum(norm.pdf(vals[d:], loc=mu2, scale=sigma))]
	## Obtain the gaps
	return np.argsort(-np.array(profile_lik))[:gaps]+1