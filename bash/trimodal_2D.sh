#!/bin/bash

#SBATCH --partition=hard

#SBATCH --exclude=zz,top,aerosmith,zeppelin

#SBATCH --job-name=2D_trim-both

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=1-04:00:00

#SBATCH --output=bash/out/%x.out

#SBATCH --error=bash/out/%x-%j.err
 
source activate tedyoon

python trimodal_train_TED.py  \
    --config=config/trimodal_2D.yml --input_context=both --name=trimodal-2D-both