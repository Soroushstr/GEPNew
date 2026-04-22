#!/bin/bash
#
#  usage: sbatch run_p1.slurm
#
#SBATCH -J gep_p1
#SBATCH -N 1
#SBATCH --partition zen3_0512_a100x2
#SBATCH --qos zen3_0512_a100x2
#SBATCH --gres=gpu:2

module purge
module load cuda/11.8.0-gcc-12.2.0-bplw5nu

mkdir -p ../results

python run_experiment_p1.py > ../results/run_p1.txt 2>&1
