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

parser = argparse.ArgumentParser(description='...')
parser.add_argument('-index','--index', help='index', type=int,default = 0)
args = parser.parse_args()

rank_temp = args.index

#Directory where the calc files are located
myDir = "/home/kavanagh/AMC/code/"
cmd = "cd "+myDir+" ; python3 prepare_distributions.py"

if (rank == 0):
    cmd += " -profile PL"
if (rank == 1):
    cmd += " -profile PL -circ"
if (rank == 2):
    cmd += " -profile PL -unperturbed 1 -circ"
    
if (rank == 3):
    cmd += " -profile NFW"
if (rank == 4):
    cmd += " -profile NFW -circ"
if (rank == 5):
    cmd += " -profile NFW -unperturbed 1 -circ"

cmd += ' -max_rows 100000 -AScut'

if (rank in [0, 1, 2, 3, 4, 5]):
    sts = call(cmd,shell=True)
comm.Barrier()
