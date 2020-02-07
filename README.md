# Bridging Ordinary-Label Learning and Complementary-Label Learning

## Dependencies
- Python 3.6
- Pytorch 1.2
- TorchVision 0.4
- Numpy 1.17
- Matplotlib 3.1

## Reproduce the Experiment demonstrated in our Paper

Experimental dataset requires to be provided multiple *complemenary labels* (or equivalently, *candidate labels*) by an *annotator*, according to the data-generation probability model defined in our paper.
The pretrained annotators used in our experiment are available [here](https://drive.google.com/file/d/11oNgzIsh9aGfmlyy9Cmb5uZsFy1C29nk/view?usp=sharing).
Simply locate the directory `/annotator`  at the root level of this git repository. The directory should include annotators for `MNIST`, `Fashion-MNIST`, `Kuzushi-MNIST`, and `CIFAR-10` for 10 and 5 classification.

To reproduce our experiment:

```
$ bash auto_demo.sh 1000
```

Note that the experiment is carried out stochastically, thus the results can be a little different from those demonstrated in our paper.
Our experiment was demonstrated on GPU, that is `NVIDIA Corporation TU102 (rev a1)`.


## Script Explanation

To train an annotator and provde candidate labels to the dataset:

```
$ python generate_dataset.py -dn <dset_name> -K <num_classes> -N <num_candidates>
```

Here is the explanation of the arguments.  

- `dset_name`: dataset name ('mnist', 'fashion', 'kuzushi', or 'cifar10')
- `num_classes`: number of the whole classes (10 or 5)
- `num_candidates`: number of candidate labels
- `resume`: resume training of the annotator

If the checkpoint `/annotators/<dset_name>/K<num_classes>_annotator.ckpt` does not exist, the script starts training the annotator.
If the pretrained annotators are downloaded from [here](https://drive.google.com/file/d/11oNgzIsh9aGfmlyy9Cmb5uZsFy1C29nk/view?usp=sharing), the script immediately starts annotation skipping the training procedure.

To run the experiment:

```
$ python demo.py -dn <dset_name> -K <num_classes> -N <num_candidates> -m <model> -ml <multi_loss> 
```

Here is the explanation of the arguments.  

- `dset_name`: dataset name ('mnist', 'fashion', 'kuzushi', or 'cifar10')
- `num_classes`: number of the whole classes (10 or 5)
- `num_candidates`: number of candidate labels
- `num_data`: number of data for each class (randomly chosen)
- `num_trial`: number of trials
- `model`: classification model ('mlp', 'linear', 'densenet', or 'resnet')
- `multi_loss`: loss function ('ova' or 'pc')


