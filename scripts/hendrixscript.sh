!/bin/bash
filename: slurm_script
# Example SLURM script to run a job on the cluster.
# The lines beginning #SBATCH set various queuing parameters.
# Set name of submitted job

SBATCH -J example_run

# Ask for 3 cores

SBATCH -n 3

# Submit with maximum 24 hour walltime HH:MM:SS

SBATCH -t 24:00:00


echo 'Your job is running on node(s):'
echo $SLURM_JOB_NODELIST
echo 'Cores per node:'
echo $SLURM_TASKS_PER_NODE