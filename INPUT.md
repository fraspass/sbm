# Separating human and automated activity in computer network traffic data

This reposit contains *Julia* code used to separate human and automated activity on a single edge within a computer network. 

The methodology builds up on the algorithm for detection of periodicities suggested in Heard, Rubin-Delanchy and Lawson (2014). A given edge can be classified as automated with significant level $\alpha$ according to the the $p$-value obtained from a Fourier's $g$-test. Using this method, the entire activity observed from the edge is discarded from further analysis. In many instances though, the the activity on edges in NetFlow data is a mixture between human and automated connections. Therefore, a mixture model for classification of the automated and human events on the edge is proposed. 

## Methodology

### Fourier analysis

Assume that $t_1,\dots,t_N$ are the raw arrival times of events on an edge $X\to Y$, where $X$ and $Y$ are client and server IP addresses. In NetFlow data, the $t_i$'s are expressed in seconds from the Unix epoch (January 1, 1970). From the raw arrival times, the counting process $N(t)$ counts the number of events up to time $t$ from the beginning of the observation period. 

Given $\{N(t),\ t=0,1,\dots,T\}$, where $T$ is the total observation time, the periodogram $\hat{S}^{(p)}(f)$ can be calculated as follows:
\begin{equation*}
\hat{S}^{(p)}(f) = \left\vert\frac{1}{T}\sum_{t=1}^T \left(N(t)-N(t-1)-\frac{N(T)}{T}\right)e^{-2\pi\imath ft}\right\vert^2
\end{equation*}

The periodogram can be easily evaluated at the Fourier frequencies $f_k = k/T,\ k=0,\dots,\lfloor T/2 \rfloor$ using the Fast Fourier Transform (FFT). The presence of periodicities can be assessed using the Fourier's $g$-statistic:
\begin{equation*}
g = \frac{\max_{1\leq k\leq\lfloor T/2\rfloor} \hat{S}^{(p)}(f_k)}{\sum_{1\leq k\leq\lfloor T/2\rfloor} \hat{S}^{(p)}(f_k)}
\end{equation*}

The $p$-value associated with an observed value $g^\star$ of the test statistic for the null hypothesis $H_0$ of no periodicities is:
\begin{equation*}
\mathbb P(g>g^\star) = \sum_{j=1}^{\min\{\lfloor 1/g^\star\rfloor,m\}} (-1)^{j-1}\binom{m}{j}(1-jg^\star)^{m-1} \approx 1-(1-\exp\{-mg^\star\} )^{m}
\end{equation*}

where $m = \lfloor T/2\rfloor$.

### Mixture modelling

The following mixture model is used to make inference on the $z_i$'s:
\begin{equation*}
f(t_i|z_i)\propto f_A(x_i)^{z_i} f_H(y_i)^{1-z_i}
\end{equation*}

The distribution of $f_A(\cdot)$ is chosen to be **wrapped normal**, and for $f_H(\cdot)$, a **histogram step function** with fixed number $B$ of changepoints $\tau$ (specified a priori) is used. Conjugate priors are used for efficient implementation. In the code, a full Gibbs sampler is used. 

## Understanding the code

The main part of the code is contained in the file `gibbs_sampler.jl`. The code also contains the implementation of an EM algorithm for a uniform - wrapped normal mixture model, used to initialise the algorithm. Finally, the code in `fft.jl` is used for periodicity detection from the sequence of raw data. For further extensions of this model, with a more flexible approach for modelling the human component, see the repository `fraspass/human_activity`.

## References

* Heard, N.A., Rubin-Delanchy, P.T.G. and Lawson, D.J. (2014). "Filtering automated polling traffic in computer network flow data". Proceedings - 2014 IEEE Joint Intelligence and Security Informatics Conference, JISIC 2014, 268-271. ([Link](https://ieeexplore.ieee.org/document/6975589/))
