#!/bin/bash

#SBATCH --partition=hard

#SBATCH --exclude=zz,top,aerosmith,zeppelin

#SBATCH --job-name=2D-diff

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=1-04:00:00

#SBATCH --output=bash/out/%x-%j.out

#SBATCH --error=bash/out/%x-%j.err
 
source activate tedyoon

python DiffGesture_train_TED.py  \
    --config=config/pose_diffusion_ted_2D.yml --unconditional=False --name=diffgesture-2D