# Bayesian estimation of the latent dimension and communities in stochastic blockmodels

## Methodology

The model is described in *Sanna Passino, F. and Heard, N. A., "Bayesian estimation of the latent dimension and communities in stochastic blockmodels"* ([link to arXiv - CURRENTLY NOT AVAILABLE](https://arxiv.org/abs/0000.00000)). 

## Understanding the code

The main part of the code is contained in the file `mcmc_sampler_sbm.py`, which contains a *python* class for the main proposals used in the MCMC sampler. The class uses some functions contained in `mvt.py`. The estimated clusters can be obtained from the estimated posterior similarity matrix (called `psm` in the code) using the function `estimate_clustering` in `estimate_cluster.py`, which uses *rpy2*. The file `synthetic_sbm.py` contains code to simulate a stochastic blockmodel using the RDPG construction, and gives an example on how to run the MCMC sampler. The files `santander.py` and `enron.py` give examples on real world networks. 

Note that `estimate_cluster.py` uses *rpy2*. In order to install the required packages in *python*:
```
from rpy2.robjects.packages import importr
utils = importr('utils')
utils.install_packages('mcclust')
```

## Running the code

Running the code is easy: `/synthetic_sbm.py --help`, `/synthetic_sbm.py --help` and `/synthetic_sbm.py --help` give detailed instruction on the possible options. For example, if the user wants to obtain 25000 MCMC samples from the posterior for the Enron Email Dataset, with no burnin, using coclustering and the second-level community allocations, and store the results in a **pre-existing folder** `Results`:

```
./enron.py -M 25000 -B 0 -c yes -s yes -f Results

```

<!--
## References

* Heard, N.A., Rubin-Delanchy, P.T.G. and Lawson, D.J. (2014). "Filtering automated polling traffic in computer network flow data". Proceedings - 2014 IEEE Joint Intelligence and Security Informatics Conference, JISIC 2014, 268-271. ([Link](https://ieeexplore.ieee.org/document/6975589/))
-->

