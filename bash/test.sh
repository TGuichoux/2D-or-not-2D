#!/bin/bash

#SBATCH --partition=hard

#SBATCH --exclude=zz,top,aerosmith,zeppelin

#SBATCH --job-name=eval

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=12:00:00

#SBATCH --output=bash/out/%x.out

#SBATCH --error=bash/out/%x-%j.err

source activate tedyoon

output_file="output/eval/output_all_val_diff_2.txt" 

# Check if output file already exists and remove it to start fresh
if [ -f "$output_file" ]; then
    rm "$output_file"
fi

# Define the arrays for context values and dimensions
contexts=("audio" "none")
dimensions=("3" "2")
postprocessing=("none" "lowering" "lifting")

# Loop over each dimension, context, and postprocessing, calling the Python script with the parameters
for dimension in "${dimensions[@]}"; do
    for context in "${contexts[@]}"; do
        for pp in "${postprocessing[@]}"; do
            # Skip combinations where pp=lowering and dimension=2 or pp=lifting and dimension=3
            if { [ "$pp" = "lowering" ] && [ "$dimension" = "2" ]; } || { [ "$pp" = "lifting" ] && [ "$dimension" = "3" ]; }; then
                echo "Skipping combination: context=$context, dimension=$dimension, postprocessing=$pp" >> "$output_file"
                continue
            fi
            echo "Running with context=$context, dimension=$dimension, postprocessing=$pp" >> "$output_file"
            python test_TED.py --mode=eval --dimension=$dimension --postprocessing=$pp --model=pose_diffusion --context=$context >> "$output_file"
            echo "------------------------------------------------" >> "$output_file"
        done
    done
done

echo "All operations completed. Check the output in $output_file"