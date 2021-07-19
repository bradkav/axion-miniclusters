#!/usr/bin/env python
from mpi4py import MPI
from subprocess import call
import numpy as np

import sys

import argparse

comm = MPI.COMM_WORLD
#Get total number of MPI processes
nprocs = comm.Get_size()
#Get rank of current process
rank = comm.Get_rank()

#Directory where the calc files are located
myDir = "/home/kavanagh/AMC/code/"
cmd = "cd "+myDir+" ; python3 prepare_distributions_delta.py"

if (rank in [0, 6]):
    cmd += " -profile PL -mass_choice a"
if (rank in [1, 7]):
    cmd += " -profile PL -mass_choice c"
if (rank in [2, 8]):
    cmd += " -profile PL -unperturbed 1 -circ"
    
if (rank in [3, 9]):
    cmd += " -profile NFW -mass_choice a"
if (rank in [4, 10]):
    cmd += " -profile NFW -mass_choice c"
if (rank in [5, 11]):
    cmd += " -profile NFW -unperturbed 1 -circ"

if (rank < 6):
    cmd += ' -AScut'

    #10000
cmd += ' -max_rows 10000'

#-Ne 1e6
cmd += "; python3 simulate_signal_delta.py -Ne 1e6 "

if (rank in [0, 6]):
    cmd += " -profile PL -mass_choice a"
if (rank in [1, 7]):
    cmd += " -profile PL -mass_choice c"
if (rank in [2, 8]):
    cmd += " -profile PL -unperturbed 1 -circ"
    
if (rank in [3, 9]):
    cmd += " -profile NFW -mass_choice a"
if (rank in [4, 10]):
    cmd += " -profile NFW -mass_choice c"
if (rank in [5, 11]):
    cmd += " -profile NFW -unperturbed 1 -circ"

if (rank < 6):
    cmd += ' -AScut'
    
if (rank in [0, 1, 3, 4, 6, 7, 9, 10]):
#if (rank in [1, 2, 4, 5, 7, 8, 10, 11]):
    sts = call(cmd,shell=True)
#if (rank in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
#    sts = call(cmd,shell=True)
comm.Barrier()
