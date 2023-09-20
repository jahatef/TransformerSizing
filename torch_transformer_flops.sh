#!/bin/bash
#SBATCH --job-name="benchmarks"
#SBATCH --partition=g40x
#SBATCH --mem-per-cpu=16GB        # Amount of CPU memory
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8          # Crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=6           # Number of cores per tasks
#SBATCH --hint=nomultithread         # We get physical cores not logical
#SBATCH --gres=gpu:8                 # Number of gpus
#SBATCH --output=%x_%j.out      # Set this dir where you want slurm outs to go
#SBATCH --error=%x_%j.out      # Set this dir where you want slurm outs to go
#SBATCH --exclusive      # Turn off node sharing
#SBATCH --account=neox
#SBATCH --exclude=ip-26-0-155-10,ip-26-0-155-46,ip-26-0-154-253,ip-26-0-148-202

# setup the environment using the script we created before
source /fsx/home-jacob/setup.sh


export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# Hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=$TRAIN_PATH/tmp/torch-elastic-error.json

# Move to the gpt-neox install
TRAIN_PATH=/fsx/home-jacob/TransformerSizing
cd $TRAIN_PATH

# Write the hostfile for this job
#/fsx/shiv/zphang/scripts/write_hostfile.sh
#export DLTS_HOSTFILE=/fsx/shiv/zphang/hostfiles/hosts_$SLURM_JOBID
bash /fsx/home-jacob/write_hostfile.sh
export DLTS_HOSTFILE=/fsx/home-jacob/hostfiles/hosts_$SLURM_JOBID
#export DLTS_HOSTFILE=/fsx/quentin/hostfiles/hosts_test_2

#sudo mkdir -p /home/quentin/.cache/torch_extensions
#sudo chmod -R 777 /home/quentin

#spawn one process 
python torch_transformer_flops.py | tee -a results/flash_attn_hsweep.out 


#to spawn a process for each of 8 gpus:
#for i in {1..8}
#do 
#    CUDA_VISIBLE_DEVICES=$i python multi_process_scripts/torch_transformer_flops_copy_${i}.py | tee -a results/3d_plot${i}.out &
#end
#wait
