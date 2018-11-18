# Estimating the stochastic blockmodel on a latent space of unknown dimension for an unknown number of communities

## Methodology

<p align="center"><img src="https://rawgit.com/fraspass/sbm/master/svgs/d35ad7fc4ab3685de91bed2397cfdf67.svg?invert_in_darkmode" align=middle width=72.5798865pt height=14.2027941pt/></p>

## Understanding the code

The main part of the code is contained in the file `gibbs_sampler.jl`. The code also contains the implementation of an EM algorithm for a uniform - wrapped normal mixture model, used to initialise the algorithm. Finally, the code in `fft.jl` is used for periodicity detection from the sequence of raw data. For further extensions of this model, with a more flexible approach for modelling the human component, see the repository `fraspass/human_activity`.

## References

* Heard, N.A., Rubin-Delanchy, P.T.G. and Lawson, D.J. (2014). "Filtering automated polling traffic in computer network flow data". Proceedings - 2014 IEEE Joint Intelligence and Security Informatics Conference, JISIC 2014, 268-271. ([Link](https://ieeexplore.ieee.org/document/6975589/))
