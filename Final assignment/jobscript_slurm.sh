#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=3
#SBATCH --partition=gpu_a100
#SBATCH --time=01:30:00

srun apptainer exec --nv --env-file .env container.sif /bin/bash main.sh