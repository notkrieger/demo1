# first attempt without tensors
import numpy as np
import matplotlib.pyplot as plt
import torch
import multiprocessing as mp

depth = 250
step = 0.0005
start = 2.8
end = 4
length = int((end - start)/step)
rs = torch.arange(start, end, step) # constants

zs = torch.ones(length + 1) * 0.3 # set start value to arbitrary number between 0 and 1

# mulitprocessing for parallelism??
numProcesses = 10
pool = mp.Pool(numProcesses)
gs = 8

def run(zs, rs, depth):
    for i in range(depth - 1): # got through depth
        zs_ = zs * rs * (1 - zs) # find next iteration
        zs = zs_ # redefine zs
        if i > 3*depth/4: # need to plot all "stable" points
            # without this step only one point for each r value is measured
            plt.plot(rs, zs, 'ko', ms = 0.025)

run(zs, rs, depth)
plt.show()
