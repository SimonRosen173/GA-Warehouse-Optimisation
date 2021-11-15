#!/bin/bash
# specify a partition
#SBATCH -p stampede
# specify number of nodes
#SBATCH -N 1
# specify the job name
#SBATCH -J ga_1
# specify the filename to be used for writing output
# NOTE: You must replace the username `hwang' with your own account name!!
#SBATCH -o /home-mscluster/srosen/research/logs/slurm.%N.%j.out
# specify the filename for stderr
#SBATCH -e /home-mscluster/srosen/research/logs/slurm.%N.%j.err

echo ------------------------------------------------------
echo -n 'Job is running on node ' $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SLURM: sbatch is running on $SLURM_SUBMIT_HOST
echo SLURM: job ID is $SLURM_JOB_ID
echo SLURM: submit directory is $SLURM_SUBMIT_DIR
echo SLURM: number of nodes allocated is $SLURM_JOB_NUM_NODES
echo SLURM: number of cores is $SLURM_NTASKS
echo SLURM: job name is $SLURM_JOB_NAME
echo ------------------------------------------------------
cd $SLURM_SUBMIT_DIR

# "pop_size,n_generations,mut_tile_size,mut_tile_no,"
# "n_agents,n_timesteps,"
# "cluster_node,run_notes,run_name,"
# "wandb_mode,log_interval,save_interval"

source activate gaenv

python3 multi_obj_train.py 64 1000 2 1 60 600 $SLURM_JOB_NODELIST "Actual Test" "GA Final 1.9" "online" 5 5