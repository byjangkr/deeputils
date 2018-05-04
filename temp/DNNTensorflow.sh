#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuDNN/lib64:/usr/local/cuda/lib64

dnnDir=$1 # DNN directory
trDataFile="$1/train.dat" # train data file 
trLabelFile="$1/train_$4.lab" # train label file
tsDataFile="$1/test.dat" # test data file
tsLabelFile="$1/test_$4.lab" # test label file
tsPredFile="$1/predTrain_$4.lab" # predict label file of test data
trPredFile="$1/predTest_$4.lab" # predict label file of train data

DNNTensorflow/trainDNN.py --feat-dim $2 --num-class $3 $trDataFile $trLabelFile $tsDataFile $tsLabelFile $tsPredFile $trPredFile

