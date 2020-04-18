#!/bin/bash
#$ -cwd
#$ -m abe
#$ -pe onenode 4
#$ -M eddyhu@gmail.com
model=$1;shift
year=$1;shift
source ~/miniconda3/bin/activate
ipcluster start -n 7 --cluster-id="$model-$year" &
sleep 45
ipython est.py $model $year
ipcluster stop
