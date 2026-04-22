#!/bin/bash
#
#  usage: sbatch run_p2c.slurm
#
#SBATCH -J gep_p2c
#SBATCH -N 1
#SBATCH --partition zen3_0512_a100x2
#SBATCH --qos zen3_0512_a100x2
#SBATCH --gres=gpu:2

module purge
module load cuda/11.8.0-gcc-12.2.0-bplw5nu

mkdir -p ../results

python run_experiment_p2c.py > ../results/run_p2c.txt 2>&1
