# Post-Hoc Reversal: Are We Selecting Models Prematurely?

This repository has code for the paper "Post-Hoc Reversal: Are We Selecting Models Prematurely" (TODO: add arxiv link).

## Install

We recommend using [mamba](https://github.com/mamba-org/mamba), a fast implementation of conda.

Create a new environment:
```bash
mamba env create -f env.yml
mamba activate post-hoc-reversal
```

## Experiments

The essential code for all experiments in the paper can be found in `src/expt`. Please see `nbs/cifar.ipynb` for a reproduction of the CIFAR-N experiments in the paper (including plots for post-hoc reversal and a table for post-hoc selection). Experiments for other datasets are similar.

All datasets are downloaded and processed automatically on first usage of the corresponding scripts.

## Citation

TODO: add citation
