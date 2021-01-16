# Candidate-Label Learning

## Dependencies

- Python 3.6
- Pytorch 1.2
- TorchVision 0.4
- Numpy 1.17
- Matplotlib 3.1

## Reproduce the Experiment Demonstrated in our Paper

The dataset should be provided with multiple candidate labels based on the data-generation probability model defined in our paper. Locate the directory `/annotator`  at the root level of this git repository. The directory should include annotators for `MNIST`, `Fashion-MNIST`, `Kuzushiji-MNIST`, and `CIFAR-10` for 10 and 5 classifications. The pretrained annotators used in our experiments are available [here](https://drive.google.com/file/d/11oNgzIsh9aGfmlyy9Cmb5uZsFy1C29nk/view?usp=sharing).  

Our experiments were demonstrated on GPU `NVIDIA Corporation TU102 (rev a1)`. To reproduce our experiment presented in Section 6:  
```
$ bash auto_demo_exp1.sh
```
To reproduce our experiment presented in Section G:  
```
$ bash auto_demo_exp2.sh
```

## Script Explanation

To train an annotator and provide candidate labels to the dataset:

```
$ python generate_dataset.py -ds <dset_name> -K <num_classes> -N <num_candidates>
```

Here is the explanation of the arguments.

- `dset_name`: dataset name ('mnist', 'fashion', 'kuzushiji', or 'cifar10')
- `num_classes`: number of the whole classes (10 or 5)
- `num_candidates`: number of candidate labels
- `resume`: resume training of the annotator

If the checkpoint `/annotators/<dset_name>/K<num_classes>_annotator.ckpt` does not exist, the script automatically starts training the annotator. If the pretrained annotators are downloaded from [here](https://drive.google.com/file/d/11oNgzIsh9aGfmlyy9Cmb5uZsFy1C29nk/view?usp=sharing), the script skips the training procedure and immediately starts annotation.

To run the experiment:
```
$ python demo.py -ds <dset_name> -K <num_classes> -N <num_candidates> -m <model> -ml <multi_loss> 
```

Here is the explanation of the arguments.  

- `dset_name`: dataset name ('mnist', 'fashion', 'kuzushi', or 'cifar10')
- `num_classes`: number of the whole classes (10 or 5)
- `num_candidates`: number of candidate labels
- `num_data`: number of data for each class (randomly chosen)
- `num_trial`: number of trials
- `model`: classification model ('mlp', 'linear', 'densenet', or 'resnet')
- `multi_loss`: loss function ('ova' or 'pc')
- `unbiased`: loss function is set to estimate unbiased risk (loss functions for OVA and PC are automatically set to be unbiased)

Learning logs will be stored in `/logs/unbiased` if the option `unbiased` is set, otherwise it will be store in `/logs/denoise` and the denoising method will be conducted.

## Citation

Related work:  
Yasuhiro Katsura and Masato Uchida. "Bridging Ordinary-Label Learning and Complementary-Label Learning." In Proceedings of The 12th Asian Conference on Machine Learning, pages 161–176, 2020.