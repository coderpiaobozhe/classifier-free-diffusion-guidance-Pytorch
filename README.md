# Unofficial Implementation of Classifier-free Diffusion Guidance
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
**NOTICE** : hyperparameter settings are in the train.py
### How to sample
```bash
make sample
```
**NOTICE** : hyperparameter settings are in the sample.py