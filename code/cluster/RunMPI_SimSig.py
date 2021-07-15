#!/usr/bin/env python
from mpi4py import MPI
from subprocess import call
import numpy as np

import sys

comm = MPI.COMM_WORLD
#Get total number of MPI processes
nprocs = comm.Get_size()
#Get rank of current process
rank = comm.Get_rank()

#Directory where the calc files are located
myDir = "/home/kavanagh/AMC/code/"
cmd = "cd "+myDir+" ; python3 simulate_signal.py -Ne 1e6 "

if (rank in [0,4]):
    cmd += " -profile PL -unperturbed 0"
if (rank in [1,5]):
    cmd += " -profile PL -unperturbed 1"
if (rank in [2,6]):
    cmd += " -profile NFW -unperturbed 0"
if (rank in [3,7]):
    cmd += " -profile NFW -unperturbed 1"

if (rank in [4,5,6,7]):
    cmd += " -AScut"
    
if (rank in [0, 1, 2, 3, 4, 5, 6, 7]):
    sts = call(cmd,shell=True)
    
comm.Barrier()
