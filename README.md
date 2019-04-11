# Bayesian estimation of the latent dimension and communities in stochastic blockmodels

## Methodology

The model is described in *Sanna Passino, F. and Heard, N. A., "Bayesian estimation of the latent dimension and communities in stochastic blockmodels"* ([link to arXiv](https://arxiv.org/abs/1904.05333)). 

## Understanding the code

The main part of the code is contained in the file `mcmc_sampler_sbm.py`, which contains a *python* class for the main proposals used in the MCMC sampler. The class uses some functions contained in `mvt.py`. The estimated clusters can be obtained from the estimated posterior similarity matrix (called `psm` in the code) using the function `estimate_clustering` in `estimate_cluster.py`, which uses *rpy2*. The file `synthetic_sbm.py` contains code to simulate a stochastic blockmodel using the RDPG construction, and gives an example on how to run the MCMC sampler. The files `santander.py` and `enron.py` give examples on real world networks. 

Note that `estimate_cluster.py` uses *rpy2*. In order to install the required packages in *python*:
```
from rpy2.robjects.packages import importr
utils = importr('utils')
utils.install_packages('mcclust')
```

## Running the code

Running the code is easy: `/synthetic_sbm.py --help`, `/synthetic_sbm.py --help` and `/synthetic_sbm.py --help` give detailed instruction on the possible options. For example, for `/synthetic_sbm.py --help`:

* `-n` is the number of nodes. Additionally, `-n2` can be used for bipartite graphs,
* `-d` is the true latent dimension in the simulated graph,
* `-K` (and `-K2` for co-clustering) represents the number of clusters, 
* `-g` and `-b` are Boolean variables used for generating directed and bipartite graphs respectively,
* `-l` is a Boolean variable used for making inference on the Laplacian spectral embedding (LSE). The adjacency spectral embedding (ASE) is the default.
* `-c` and `-s` are Boolean variables used for co-clustering and second-level clustering respectively,
* `-r` is a Boolean variable: if it is set to `1`, then MCMC is run on the simulated graphs, otherwise summary plots are produced, 
* `-M` and `-B` denote the number of samples in the MCMC chain, and the burnin,
* `-f` is the name of the destination folder for the output files. It must be **pre-existent**. 

For example, if the user wants to obtain 25000 MCMC samples from the posterior for the Enron Email Dataset, with no burnin, using coclustering and the second-level community allocations, and store the results in a **pre-existing folder** `Output`:

```
./enron.py -M 25000 -B 0 -c yes -s yes -f Output

```
The output consists in multiple files: for example, `Ko.txt` contains the MCMC chain for the number of non-empty clusters <img alt="$K_\varnothing$" src="svgs/c09a28e6f1aeb430bd603a5562d11a90.svg" align="middle" width="24.235233pt" height="22.4657235pt"/>, and `psm.txt` is the estimated posterior similarity matrix. The names of the output files are self-explanatory and consistent with the notation used in the paper.  

Reproducing the figures in the paper is also easy. For example, for Figure 5:
```
./synthetic_sbm.py -n 250 -n2 300 -g 1 -b 1
```
The resulting plots are in the repository in the directory `Results` (default for `-f`). Note that the above code produces plot in `matplotlib`. The figures in the paper are postprocessed using `matplotlib2tikz`. 
```
from matplotlib2tikz import save as tikz_save
tikz_save('foo.tex')
```

**- Note -** For the Boolean variables, the admissable values are `yes`, `true`, `t`, `y`, `1` (and uppercase counterparts) for the positive class, and `no`, `false`, `f`, `n`, `0` (and uppercase counterparts) for the negative class. 

<!--
## References

* Heard, N.A., Rubin-Delanchy, P.T.G. and Lawson, D.J. (2014). "Filtering automated polling traffic in computer network flow data". Proceedings - 2014 IEEE Joint Intelligence and Security Informatics Conference, JISIC 2014, 268-271. ([Link](https://ieeexplore.ieee.org/document/6975589/))
-->

