#!/bin/bash
 
#SBATCH -N 1 --ntasks-per-node=16  
#SBATCH -t 06:00:00                                                                                                                                        
#SBATCH -p normal

#SBATCH -o slurm_output/slurm-%j.out # STDOUT                                                                                                                           
#SBATCH -e slurm_output/slurm-%j.err # STDERR

cd $HOME/AMC/

#module load openmpi/gnu
#module load python/2.7.9

module load pre2019

module unload GCCcore
module load Python/2.7.12-intel-2016b
module load slurm-tools

export SLURM_CPU_BIND=none

time mpirun -np 16 python2.7 RunMPI_PrepDist.py -index 0
