# Unofficial Implementation of [Classifier-free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
The Pytorch implementation is adapted from [openai/guided-diffusion](https://github.com/openai/guided-diffusion) with modifications for classifier-free conditioned generation. The dataset used for training is cifar-10.
## How to run
First, you need to do some preparations:
```bash
mkdir sample
mkdir model
ln -s absolute/path/to/cifar-10 ./cifar-10-batches-py
```
### How to train
```bash
make train
```
**NOTICE** : hyperparameter settings are in the file *train.py*
### How to sample
```bash
make sample
```
**NOTICE** : hyperparameter settings are in the file *sample.py*
