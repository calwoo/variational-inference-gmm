## variational inference with GMMs

This is a mini-project to understand variational inference better with a Gaussian mixture model (which we will call GMM from now on).

### background

The goal of VI is to provide computationally tractible ways to compute posterior distributions coming from probabilistic graphical models. For example in a GMM, there is a discrete (categorical) latent variable `z` taking values in $1,...,M$ where $M$ is the number of clusters. Given a `z`, the conditional distribution `p(x|z)` is a gaussian:

`x ~ p(x|z) = N(\mu, \sigma^2)`

A probabilistic graphical model provides a model for the joint distribution `p(x,z) = p(x|z)p(z)` from which we wish to perform inference on. Why inference? Usually, x is the observed data and what we are really are looking for are the hidden features learned by the model, which is given by the latent variables `z`. For example, given an observed data point `x`, we'd like to figure out which cluster is comes from. In particular, we are interested in computing

`p(z|x) = p(x,z) / p(x)`

But this is really hard! For example, even with a GMM model, the marginal 

`p(x) = \sum_{i=1,...,M} p(z=i)N(x;\mu_i, \sigma_i^2)`

is generally a difficult beast to compute. In the cases where the prior `p(z)` is a continuous distribution, this may be analytically impossible. What do we do instead?

### the elbo and vi to the rescue

We will often instead opt to compute an easier lower bound to the posterior we are trying to compute. This lower bound is called the elbo (derivation in attached notes).

`log p(x) >= E[log p(x,z)] - E[log q(z)]`



