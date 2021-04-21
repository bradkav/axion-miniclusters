#!/usr/bin/env python
from mpi4py import MPI
from subprocess import call
import numpy as np

import sys

import argparse

#Parse the arguments!
parser = argparse.ArgumentParser(description='...')
parser.add_argument('-N_AMC','--N_AMC', help='Number of AMCs to simulate', type=int,default = 100000)
parser.add_argument('-R_ini', '--R_ini', help='Value of R to start at', type=float, required=True)
parser.add_argument('-R_fin', '--R_fin', help='Value of R to end at', type=float, required=True)
parser.add_argument('-profile', '-profile', help="AMC profile - `PL` or `NFW`", type=str, default="PL")

parser.add_argument("-circ", "--circular", dest="circular", action='store_true', help="Use the circular flag to force e = 0 for all orbits.")
parser.set_defaults(circular=False)

args = parser.parse_args()


comm = MPI.COMM_WORLD
#Get total number of MPI processes
nprocs = comm.Get_size()
#Get rank of current process
rank = comm.Get_rank()

R_list = np.logspace(np.log10(args.R_ini), np.log10(args.R_fin), nprocs)


#MC_script.py -R 1 -N 20000

#Directory where the calc files are located
myDir = "/home/kavanagh/AMC/code/"
cmd = "cd "+myDir+" ;time python3 MC_script_ecc.py "
cmd += "-a " + str(R_list[rank])
cmd += " -N " + str(args.N_AMC)
cmd += " -profile " + str (args.profile)

if (args.circular):
    cmd += " -circ"

sts = call(cmd,shell=True)
comm.Barrier()
