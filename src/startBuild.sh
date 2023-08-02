#!/bin/bash
#SBATCH --job-name=hpc_img_clssfr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128

echo "Changing dirctories to Work..."
cd /lustre/work/client/users/rherrin

echo "Setting up spack environment..." 
source ./spack/share/spack/setup-env.sh

echo "Activating spack environment..."  
spack env activate hpc-final

echo "Changing directory to project source..."
cd ./final_project/src

echo "Launching Python Build..."
python main.py


