#!/bin/bash

#SBATCH --partition=jazzy

#SBATCH --job-name=generate_pickles

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=12:00:00

#SBATCH --output=bash/out/%x.out

#SBATCH --error=bash/out/%x-%j.err

source activate tedyoon

python scripts/DiffGesture_test_TED_2DVers.py --mode=long --dimension=2 --postprocessing=lifting --model=multimodal_context --context=both 
python scripts/DiffGesture_test_TED_2DVers.py --mode=long --dimension=2 --postprocessing=lifting --model=pose_diffusion --context=audio

