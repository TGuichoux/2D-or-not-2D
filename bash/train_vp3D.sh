#!/bin/bash

#SBATCH --partition=hard

#SBATCH --exclude=lizzy,thin,zeppelin,led

#SBATCH --job-name=vp3D

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=1-12:00:00

#SBATCH --output=bash/out/%x.out

 
source activate tedyoon

python VP3D_train_TED.py