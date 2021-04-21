#!/usr/bin/env python3
import os

#Specify the locations of the Monte Carlo simulations and data directories here.
#Edit these variables to point to the right place for you.

data_dir = "/Users/bradkav/Projects/AMC_encounters/axion-miniclusters/data/"
montecarlo_dir = "/Users/bradkav/Projects/AMC_encounters/code/AMC_montecarlo_data/"

if (os.environ['HOME'] == "/home/kavanagh"):
    data_dir = "/home/kavanagh/AMC/data/"
    montecarlo_dir =  "/home/kavanagh/AMC/AMC_montecarlo_data/"

