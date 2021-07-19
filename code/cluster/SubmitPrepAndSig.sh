#!/bin/bash
 
#SBATCH -N 1 --ntasks-per-node=16  
#SBATCH -t 12:00:00                                                                                                                                        
#SBATCH -p normal
# #SBATCH -p short

#SBATCH -o /home/kavanagh/AMC/slurm_output/slurm-%j.out # STDOUT                                                                                                                           
#SBATCH -e /home/kavanagh/AMC/slurm_output/slurm-%j.err # STDERR

cd $HOME/AMC/code/cluster

#module load openmpi/gnu
#module load python/2.7.9

module load pre2019

module unload GCCcore
module load Python/2.7.12-intel-2016b
module load slurm-tools

export SLURM_CPU_BIND=none

time mpirun -np 16 python2.7 RunMPI_PrepAndSig.py
