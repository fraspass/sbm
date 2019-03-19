#!/usr/bin/env python
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from sklearn.cluster import AgglomerativeClustering 
import numpy as np

###################################################################################################
### Maximise PEAR (posterior expected adjusted Rand index) from the posterior similarity matrix ###
###################################################################################################

# Import R's packages
base = importr('base')
utils = importr('utils')
mcclust = importr('mcclust')
rpy2.robjects.numpy2ri.activate()

## The function takes the posterior similarity matrix (psm) as argument (obtained from MCMC chains)
def estimate_clustering(psm,k=None,min_clust=0):
	psm = psm / np.max(psm)
	if k is None:
		## If k is not specified, maxpear is used -- from the R package 'mcclust' (Fritsch and Ickstadt, 2009) 
		Br = ro.r.matrix(psm, nrow=psm.shape[0], ncol=psm.shape[1])
		ro.r.assign("B", Br)
		cl = mcclust.maxpear(Br)
		clust = np.array(cl[cl.names.index('cl')])
	else:
		## If k is specified, use agglomerative clustering 
		cluster_model = AgglomerativeClustering(n_clusters=int(k), affinity='precomputed', linkage='average') 
		clust = cluster_model.fit_predict(1-psm) + 1
	return clust if min_clust!=0 else clust-1  


