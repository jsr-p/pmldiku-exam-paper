# Questions for the exercise class

- In question B, is it correct that $y_i = f(x_i)$, so $y_i$ is non-stochastic, which implies:
  - $y$ does not have a likelihood function
  - $p(\mathcal{D}) = 1$ (trivially, $\mathcal{D}$ is non-stochastic too while $S = \{-1, -1/2, \ldots, 1\}$)
  - etc.
- Is it correct that we should sample from $p(f^{\star} \mid x^{\star}, \mathcal{D})$ (with $\theta$ deliberately not in the condioning set) i.e. we have to "weight" $\theta$ out as in [this derivation](#predictive-posterior)?
  - Or is $\theta$ in the conditoning set such that we just sample one $\theta^{(i)} \sim p(\theta \mid \mathcal{D})$, insert it into the GP and then construct the posterior predictive?

## Questions e-mail

Hi Oswin

At the exercise class today we asked whether it was on purpose that
$\theta$ did not enter the conditioning set of $p(f^{\star} \mid x^{\star}, \mathcal{D})$ (in question B) and got an affirmative answer.
This implies some things, for instance that we sample from the posterior predictive using an average as also written in the Stan manual: <https://mc-stan.org/docs/stan-users-guide/computing-the-posterior-predictive-distribution.html>.

But, then in question B.2, hint 1 says:
"To obtain a sample $f^{\star}$ only a single sample from $p(\theta \mid \mathcal{D})$ is needed".

We know from question B.1 that
$$
f^{\star} \sim p(f^{\star} \mid x^{\star}, \mathcal{D}) \approx S^{-1} \sum_{i=1}^{S} p(f^{\star} \mid x^{\star}, \mathcal{D}, \theta^{(i)}).
$$
This is also the way that Algorithm 1 says that $f^{\star}$ should be sampled in line 2 of the algorithm.

Our question is then: how is it possible to sample an $f^{\star}$ using only a single sample from the posterior $\theta^{(i)} \sim p(\theta \mid \mathcal{D})$ without having this sample in the conditioning set of $p(f^{\star} \mid x^{\star}, \mathcal{D})$?

That is, shouldn't Algorithm 1 specify, using hint 1, that

$$
f^{\star} \sim p(f^{\star} \mid x^{\star}, \mathcal{D}, \theta^{(i)})
$$
?

kind regards

Jonas, Julius & Jeppe

### Derivations for Gaussian Processes

#### Full Bayesian model

![Deriv1](figs/derivations/deriv1.jpg)

#### Predictive posterior

![Deriv2](figs/derivations/deriv2.jpg)

#### Predictive posterior

![Deriv3](figs/derivations/deriv3.jpg)

#### Average out

![Deriv4](figs/derivations/deriv4.jpg)
