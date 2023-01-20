# How to run the code:

First, create an conda environment following the guide [here](../README.md)
and install the `pmldiku` package.

## A: 
The results for question A are created by running the scripts inside [scripts](./scripts/) in the order:

1. [run_diffusion_models.py](./scripts/run_diffusion_models.py)
2. [run_vae_models.py](./scripts/run_vae_models.py)
3. [frechet_inception_distance.py](./scripts/frechet_inception_distance.py)
4. [estimate_all_marginal_loglik.py](./scripts/estimate_all_marginal_loglik.py)

## B: 
The results for question B are created by running the notebook [function_fitting.ipynb](./notebooks/B/function_fitting.ipynb)
