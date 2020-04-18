#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M eddyhu@gmail.com
model=$1;shift
source ~/miniconda3/bin/activate
ipcluster start -n 20 --cluster-id="$model-fix" &
sleep 45
ipython fix.py $model
ipcluster stop
