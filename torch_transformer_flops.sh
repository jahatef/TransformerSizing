#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -A csc439

# setup the environment using the script we created before
source /lustre/orion/csc439/scratch/jahatef/transformerSizing/setup.sh


export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`


# Move to the gpt-neox install
TRAIN_PATH=/lustre/orion/csc439/scratch/jahatef/transformerSizing/TransformerSizing
cd $TRAIN_PATH

# Write the hostfile for this job
#/fsx/shiv/zphang/scripts/write_hostfile.sh
#export DLTS_HOSTFILE=/fsx/shiv/zphang/hostfiles/hosts_$SLURM_JOBID
bash /lustre/orion/csc439/scratch/jahatef/write_hostfile.sh
export DLTS_HOSTFILE=/lustre/orion/csc439/scratch/jahatef/hostfiles/hosts_$SLURM_JOBID
#export DLTS_HOSTFILE=/fsx/quentin/hostfiles/hosts_test_2

#sudo mkdir -p /home/quentin/.cache/torch_extensions
#sudo chmod -R 777 /home/quentin

for i in 6
do
	echo $i
	ROCR_VISIBLE_DEVICES=$i python torch_transformer_flops${i}.py | tee -a results/max_h_sweep_mi250_${i}.out &
done
wait

#for i in 0 2
#do
#	echo $i
#	MASTER_PORT=$((6000 + $i)) ROCR_VISIBLE_DEVICES=$i python torch_transformer_flops_3d_${i}.py | tee -a results/3d_plot_mi250_${i}.out &
#done
#wait