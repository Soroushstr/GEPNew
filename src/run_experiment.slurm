#!/bin/bash
#
#  usage: sbatch run_experiment.slurm
#
#SBATCH -J gep_mcc_v1
#SBATCH -N 1
#SBATCH --partition zen3_0512_a100x2
#SBATCH --qos zen3_0512_a100x2
#SBATCH --gres=gpu:2

module purge
module load cuda/11.8.0-gcc-12.2.0-bplw5nu

# results/ directory is created automatically by run_experiment.py;
# stdout + stderr are captured here for convenience
mkdir -p ../results

python run_experiment.py > ../results/run_experiment_mcc_v1.txt 2>&1
