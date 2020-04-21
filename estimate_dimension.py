#!/usr/bin/env python
import sys
import numpy as np
from scipy.stats import norm

########################################################################
### Estimate the dimension using the method of Zhu and Ghodsi (2006) ###
########################################################################

### Function to estimate the dimension using the automatic method of Zhu and Ghodsi (2006)
def zg_method(vals):
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
		sigma = np.sqrt((np.sum((vals[:d] - mu1) ** 2) + np.sum((vals[d:] - mu2) ** 2)) / (m-2))
		profile_lik += [np.sum(norm.logpdf(vals[:d], loc=mu1, scale=sigma)) + np.sum(norm.logpdf(vals[d:], loc=mu2, scale=sigma))]
	## Obtain the gaps
	return profile_lik

## Sequential ZG method
def sequential_zg(vals, k=3, pythonic=True):
	## Calculate the first dimension using the ZG method
	d = [np.argsort(zg_method(vals))[-1] + 1]
	## Repeat for the remaining values 
	for _ in range(k-1):
		d += [d[-1] + np.argsort(zg_method(np.sort(vals)[::-1][d[-1]:]))[-1] + 1]
	## Return list of values of the dimension
	return np.array(d)-(1 if pythonic else 0)