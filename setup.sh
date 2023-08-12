module load openmpi cuda/11.7

CONDA_HOME=/fsx/quentin/jacob/gpt-neox-stuff/GEMMs_project/transformer_sizing/experiments/scripts/miniconda3-plot
#CONDA_HOME=/fsx/quentin/jacob/miniconda3-test
#CONDA_HOME=/fsx/quentin/jacob/gpt-neox/conda/envs/improved-t5
CUDNN_HOME=/fsx/quentin/jacob/cudnn-linux-x86_64-8.6.0.163_cuda11-archive

export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDNN_HOME/include:$CPATH

export PATH=$CONDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CONDA_HOME/include:$CPATH

#source /fsx/quentin/jacob/gpt-neox/conda/bin/activate improved-t5
#conda list

export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.7/efa/lib:/usr/local/cuda-11.7/lib:/usr/local/cuda-11.7/lib64:/usr/local/cuda-11.7:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:/usr/local/cuda-11.7/bin:$PATH
#export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"

#conda activate gpt-neox
