# Bridging Ordinary-Label Learning and Complementary-Label Learning

## Dependencies
- Python 3.6
- Pytorch 1.2
- TorchVision 0.4

## Prepare experimental dataset with complementary labels

```
$ python generate_comp_dataset.py -dn mnist
$ python generate_comp_dataset.py -dn kmnist
$ python generate_comp_dataset.py -dn fmnist
$ python generate_comp_dataset.py -dn cifar10
```

## Demo

To run the demo for MNIST with K=10 and N=9 with MLP model for one-versus-all classification:

```
$ python demo.py -K 10 -N 9 -ml ova -dn mnist -m mlp
```

To run the whole demo demonstrated in our paper, simply:

```
$ bash auto_demo.sh 1000
```
