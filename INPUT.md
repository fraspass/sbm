# Estimating the stochastic blockmodel on a latent space of unknown dimension for an unknown number of communities

## Methodology

For simplicity, the embeddings will be generically denoted as ${\mathbf X}=[\boldsymbol x_1,\dots,\boldsymbol x_n]^\top\in\mathbb R^{n\times m},\ \boldsymbol x_i\in\mathbb R^m$ for some $m>K$, and the non-identifiable latent position $\boldsymbol v^\star_j$ are more practically renamed $\boldsymbol v_j$. The notation $\boldsymbol x_{i:d}$ denotes the first $d$ elements $(x_1,\dots,x_d)$ of the vector $\boldsymbol x_i$, and similarly $\boldsymbol x_{id:}$ denotes the last $m-d$ elements $(x_{d+1},\dots,x_{m})$ of the vector. We assume that, given the number of communities $K$ and the latent dimension $d$, the embeddings are generated from $K$ $d$-dimensional community-specific Gaussians in the first $d$ components, say $\mathbf X_{:d}$, and from a generic $(m-d)$-dimensional Gaussian (which is not cluster-specific) for the remaining components $\mathbf X_{d:}$. Therefore, introducing latent community assignments $\boldsymbol z=(z_1,\dots,z_n)$, the model can be expressed as follows:

\begin{align*}
\boldsymbol x_i \vert d,K,z_i,\boldsymbol v_{z_i},\bm\Sigma_{z_i}, \boldsymbol v_r, \bm\Sigma_r &\overset{d}{\sim} \mathbb N_m \left( \begin{bmatrix} \boldsymbol v_{z_i} \\ \boldsymbol v_r \end{bmatrix}, \begin{bmatrix} \bm\Sigma_{z_i} & \boldsymbol 0 \\ \boldsymbol 0 & \bm\Sigma_r \end{bmatrix} \right),\ i=1,\dots,n, \\
(\boldsymbol v_{k},\bm\Sigma_{k})\vert d, K &\overset{iid}{\sim} \mathrm{NIW}_d(\boldsymbol v_{0:d},\kappa_0,\nu_0+d-1,\bm\Delta_{0:d}),\ k=1,\dots,K \\
(\boldsymbol v_{r},\bm\Sigma_{r})\vert d, K &\overset{d}{\sim} \mathrm{NIW}_{m-d}(\boldsymbol v_{0d:},\kappa_0,\nu_0+m-d-1,\bm\Delta_{0d:}), \\
z_i\vert\bm\theta, K &\overset{iid}{\sim}\mathrm{Multinoulli}(\bm\theta),\ i=1,\dots,n, \ \bm\theta\in\mathcal S_{K-1}, \\
\bm\theta \vert K &\overset{d}{\sim} \mathrm{Dirichlet}\left(\frac{\alpha}{K},\dots,\frac{\alpha}{K}\right), \\
k &\overset{d}{\sim} \mathrm{Geometric}(\omega), \\
d &\overset{d}{\sim} \mathrm{Geometric}(\delta).
\end{align*}

where $\mathcal S_{K-1}$ is the $K-1$ probability simplex. Note that the inverse Wishart has been partially re-parametrised using a parameter $\nu_0>0$ and adding the corresponding dimension to obtain the required constraint $\nu>d-1$ for the generic parametrisation and interpretation of $\nu$ in the inverse Wishart. Also note that $m$ can be chosen to be equal to $K$, when fixed, for parsimony, or equal to $n$ to have the maximum possible dimension of the embeddings.

## Understanding the code

The main part of the code is contained in the file `gibbs_sampler.jl`. The code also contains the implementation of an EM algorithm for a uniform - wrapped normal mixture model, used to initialise the algorithm. Finally, the code in `fft.jl` is used for periodicity detection from the sequence of raw data. For further extensions of this model, with a more flexible approach for modelling the human component, see the repository `fraspass/human_activity`.

## References

* Heard, N.A., Rubin-Delanchy, P.T.G. and Lawson, D.J. (2014). "Filtering automated polling traffic in computer network flow data". Proceedings - 2014 IEEE Joint Intelligence and Security Informatics Conference, JISIC 2014, 268-271. ([Link](https://ieeexplore.ieee.org/document/6975589/))
