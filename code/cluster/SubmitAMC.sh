#!/bin/bash

# Time should probably be about 60 hours

#SBATCH -N 1 --ntasks-per-node=16  
#SBATCH -t 30:00:00
#SBATCH -p normal

# #SBATCH -t 00:04:30
# #SBATCH -p short

#SBATCH -o /home/kavanagh/AMC/slurm_output/slurm-%j.out # STDOUT                                                                                                                           
#SBATCH -e /home/kavanagh/AMC/slurm_output/slurm-%j.err # STDERR


cd $HOME/AMC/cluster

#module load openmpi/gnu
#module load python/2.7.9

export SLURM_CPU_BIND=none

module load pre2019

module unload GCCcore
module load Python/2.7.12-intel-2016b
module load slurm-tools

NAMC=100000

if [ $1 -eq 1 ]; then
    RMIN=0.1
    RMAX=0.44
elif [ $1 -eq 2 ]; then
    RMIN=0.48
    RMAX=2.13
elif [ $1 -eq 3 ]; then
    RMIN=2.35
    RMAX=10.32
elif [ $1 -eq 4 ]; then
    RMIN=11.39
    RMAX=50.0
fi

time mpirun -np 16 python2.7 RunMPI_AMC.py -N_AMC $NAMC -R_ini $RMIN -R_fin $RMAX -profile $2 -circ
#time mpirun -np 16 python2.7 RunMPI_AMC.py -N_AMC $NAMC -R_ini 0.48 -R_fin 2.13 -profile $2
#time mpirun -np 16 python2.7 RunMPI_AMC.py -N_AMC $NAMC -R_ini 2.35 -R_fin 10.32 -profile $2
#time mpirun -np 16 python2.7 RunMPI_AMC.py -N_AMC $NAMC -R_ini 11.39 -R_fin 50.0 -profile $2

#0.1 0.44
#0.48 2.13
#2.35 10.32
#11.39 50.0
