#!/bin/bash
#$ -cwd
#$ -m abe
#$ -pe onenode 8
#$ -l m_mem_free=3G
#$ -M eddyhu@gmail.com
model=$1;shift
year=$1;shift
source ~/miniconda3/bin/activate
ipcluster start -n 15 --cluster-id="$model-$year" &
sleep 45
ipython est.py $model $year
ipcluster stop
