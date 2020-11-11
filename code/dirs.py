#!/usr/bin/env python3
import os

#Specify the locations of the Monte Carlo simulations and data directories here.

data_dir = "/Users/bradkav/Projects/AMC_encounters/axion-miniclusters/data/"
montecarlo_dir = "/Users/bradkav/Projects/AMC_encounters/code/AMC_montecarlo_data/"

if (os.environ['HOME'] == "/home/kavanagh"):
    data_dir = "../data/"
    montecarlo_dir =  "../AMC_montecarlo_data/"

