# first attempt without tensors
import numpy as np
import matplotlib.pyplot as plt
import torch

depth = 250
halfDepth = int(depth/2)
step = 0.0005
start = 2.8
end = 4
length = int((end - start)/step)
rs = torch.arange(start, end, step) # growth rate

#zs = torch.zeros((length, halfDepth)) # for integer start/end values
zs = torch.zeros((length + 1, halfDepth)) # for decimal start/end values

for j in range(length):
    r = rs[j] # define growth constant
    x_ = 0.3  # arbitrary initial value

    xs = torch.zeros(depth)
    xs[0] = x_

    for i in range(1, depth):
        xs[i] = xs[i-1] * r * (1 - xs[i-1])
        if i > halfDepth:
            zs[j][i - halfDepth] = xs[i]

plt.plot(rs, zs, 'kx', ms = 0.005)
plt.show()

