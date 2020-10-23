#!/usr/bin/env python3
import os

#print(os.environ['HOME'])

#montecarlo_dir = "../AMC_montecarlo_data/"

data_dir = "/Users/bradkav/Projects/AMC_encounters/axion-miniclusters/data/"
montecarlo_dir = "/Users/bradkav/Projects/AMC_encounters/code/AMC_montecarlo_data/"

if (os.environ['HOME'] == "/home/kavanagh"):
    data_dir = "../data/"
    montecarlo_dir =  "../AMC_montecarlo_data/"

#data_dir = "/Users/bradkav/Projects/AMC_encounters/axion-miniclusters/data/"
#dists_dir = "/Users/bradkav/Dropbox/Projects/Axion star radio detection/current/data_ecc/distributions/"
#data_dir = "/Users/bradkav/Dropbox/Projects/Axion star radio detection/current/data_ecc/"
#print(data_dir)
