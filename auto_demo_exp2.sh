#!/bin/bash

# bash code for the experiment presented in Section G

n=1000

dset_name_list=('cifar10')
K_list=(10 5)
loss_list=('ce')


for K in ${K_list[@]}
do
    if [ $K -eq 10 ];then
        N_list=(1 2 3 4 5 6 7 8 9)
    elif [ $K -eq 5 ];then
        N_list=(1 2 3 4)
    fi

    for dset_name in ${dset_name_list[@]}
    do
        echo $dset_name
        if [ $dset_name != 'cifar10' ];then
            model='mlp'
            weight_decay=0.0001
            learning_rate=5e-5
        else
            model='densenet'
            weight_decay=5e-4
            learning_rate=0.05
        fi

        # dataset generation
        # skip the following if the dataset is already annotated
        for N in ${N_list[@]}
        do
            python generate_dataset.py -K $K -N $N -ds $dset_name
        done

        # demo
        for N in ${N_list[@]}
        do
            for loss in ${loss_list[@]}
            do
                echo -e "\n--\nNumber of candidates: " $N "/" $K
                # denoise the probability model Q(y|x)
                python demo.py -K $K -N $N -n $n -ds $dset_name -m $model -ml $loss -wd $weight_decay -lr $learning_rate -nt 3 -ne 300
            done
        done
    done
done


