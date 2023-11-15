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

# setup the environment using the script we created before
source /fsx/home-jacob/setup.sh

export NCCL_DEBUG=WARN
export NCCL_TREE_THRESHOLD=0
export NCCL_PROTO=simple
# Network issues without the following two NCCL vars set; See https://github.com/NVIDIA/nccl/issues/676
export NCCL_IBEXT_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64

export PYTHONFAULTHANDLER=1

export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_btl="^openib"

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

python mm_flops.py >> results/vocab.out
