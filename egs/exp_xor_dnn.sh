#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuDNN/lib64:/usr/local/cuda/lib64

dnnDir=../sample_data # DNN directory
trDataFile="$dnnDir/xor/train.dat" # train data file 
trLabelFile="$dnnDir/xor/train.lab" # train label file
tsDataFile="$dnnDir/xor/test.dat" # test data file
tsLabelFile="$dnnDir/xor/test.lab" # test label file
tsPredProbFile="$dnnDir/xor/pred.prob" # predict probability file of test data
tsPredLabFile="$dnnDir/xor/pred.lab" # predict label file of test data

mkdir -p ../exp/xor/1
dnn/trainDNN_ex.py  --num-class 2 --lr 0.001 --num-epoch 10 --minibatch 10 --valEpoch 1 \
			$trDataFile $trLabelFile ../exp/xor/1 

dnn/predDNN_ex.py --out-predprob $tsPredProbFile --out-predlab $tsPredLabFile \
			$tsDataFile $tsLabelFile ../exp/xor/1

#dnn_base/trainDNN.py --feat-dim 2 --num-class 2 \
#			--num-epoch 100 --minibatch 5 \
#			--mdl-dir exp/xor/1 \
#			$trDataFile $trLabelFile exp/xor/2

#dnn_base/predDNN.py $tsDataFile $tsLabelFile exp/xor/2 $tsPredFile
