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
#source /fsx/home-jacob/setup.sh


#export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
#export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=localhost
export MASTER_PORT=12802
#export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# Hide duplicated errors using this hack - will be properly fixed in pt-1.12
#export TORCHELASTIC_ERROR_FILE=$TRAIN_PATH/tmp/torch-elastic-error.json

# Move to the gpt-neox install
TRAIN_PATH=/workspace/TransformerSizing
cd $TRAIN_PATH

# Write the hostfile for this job
#/fsx/shiv/zphang/scripts/write_hostfile.sh
#export DLTS_HOSTFILE=/fsx/shiv/zphang/hostfiles/hosts_$SLURM_JOBID
#bash /fsx/home-jacob/write_hostfile.sh
#export DLTS_HOSTFILE=/fsx/home-jacob/hostfiles/hosts_$SLURM_JOBID
#export DLTS_HOSTFILE=/fsx/quentin/hostfiles/hosts_test_2

#sudo mkdir -p /home/quentin/.cache/torch_extensions
#sudo chmod -R 777 /home/quentin

for i in {0..3}
do
    echo $i
    MASTER_PORT=$((6000 + $i)) CUDA_VISIBLE_DEVICES=$i python torch_transformer_flops${i}.py | tee -a results/max_h_sweep_h100_${i}.out &
done
wait

#python torch_transformer_flops.py | tee -a results/flash_attn_proportion-large.out &
#python torch_transformer_flops_copy_2.py | tee -a results/3d_plot2.out &
#python torch_transformer_flops_copy_3.py | tee -a results/3d_plot3.out &
#python torch_transformer_flops_copy_4.py | tee -a results/3d_plot4.out &
#python torch_transformer_flops_copy_5.py | tee -a results/3d_plot5.out &
#python torch_transformer_flops_copy_6.py | tee -a results/3d_plot6.out &
#python torch_transformer_flops_copy_7.py | tee -a results/3d_plot7.out &
#python torch_transformer_flops_copy_8.py | tee -a results/3d_plot8.out &
wait
