#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuDNN/lib64:/usr/local/cuda/lib64

dataDir=../sample_data/mnist/ # DNN directory


mkdir -p exp/mnist/1
cnn/trainCNN_ex.py  --num-class 10 --lr 0.001 --num-epoch 10 --minibatch 100 --valEpoch 1 \
			$dataDir $dataDir ../exp/mnist/1 

cnn/predCNN_ex.py $dataDir $dataDir ../exp/mnist/1


