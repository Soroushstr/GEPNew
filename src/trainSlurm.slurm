#!/bin/bash
#
#  usage: sbatch ./gpu_test.scrpt
#
#SBATCH -J cfix4
#SBATCH -N 1                           #use -N only if you use both GPUs on the$
#SBATCH --partition zen3_0512_a100x2   #For A40 cards, use --partition zen2_025$
#SBATCH --qos zen3_0512_a100x2
#SBATCH --gres=gpu:2                   #or --gres=gpu:1 if you only want to use$

module purge
module load cuda/11.8.0-gcc-12.2.0-bplw5nu

python run_experiment.py > results/run_experiment_mcc_v1.txt 2>&1
