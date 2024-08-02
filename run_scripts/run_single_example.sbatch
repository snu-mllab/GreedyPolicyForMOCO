#!/bin/bash
#SBATCH --job-name=regex_4_experiments  # Job name
#SBATCH --output=output_%j.log          # Standard output and error log
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=12               # Number of CPU cores per task
#SBATCH --mem=60000                       # Job memory request
#SBATCH --time=100:00:00                 # Time limit hrs:min:sec
#SBATCH --partition rtx3090             # 3090 partition
#SBATCH --gres gpu:1                     # single GPU Job
#SBATCH --array=0-9                     # Array of seeds 0 through 9
#SBATCH --qos normal                    # normal priority


# Load conda and activate environment
source activate setbench

# Define directory to save results
DIR2SAVE="/home/deokjae/GreedyPolicyForMOCO/results/table1_example/"  # Update this path as needed

# Run experiments
declare -a algorithms=(
                        "setrl"
                        "moreinforce" 
                        "moreinforce" 
                        "greedyrl" 
                        "rand_hill" 
                        "rand_sample"
                    )
declare -a extra_args=(
                        "algorithm.pi_lr=1e-4 algorithm.random_action_prob=0.0" 
                        "algorithm.pi_lr=1e-4 algorithm.random_action_prob=0.0 algorithm.reward_type=tchebycheff algorithm.simplex_bins=20" 
                        "algorithm.pi_lr=1e-5 algorithm.random_action_prob=0.0 algorithm.reward_type=convex algorithm.simplex_bins=20" 
                        "algorithm.pi_lr=1e-5 algorithm.random_action_prob=0.0" 
                        "" 
                        ""
                    )

for i in "${!algorithms[@]}"; do
  algorithm=${algorithms[$i]}
  args=${extra_args[$i]}

  python run_scripts/run_single.py \
    algorithm.state_save_path=$DIR2SAVE \
    task=regex_4 \
    algorithm=$algorithm \
    algorithm.max_size=16 \
    trial_id=$SLURM_ARRAY_TASK_ID \
    $args
done
