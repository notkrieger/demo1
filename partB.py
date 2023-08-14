import numpy as np
import torch
import matplotlib.pyplot as plt

#X, Y = np.mgrid[-1.3:1.3:0.004, -2:1:0.004] # mandelbrot set
X, Y = np.mgrid[-1:1.3:0.004, -2:2:0.004] # julia set


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(y, x) # swapped axis to make work ???

zs = z.clone()
ns = torch.zeros_like(z)

z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

depth = 200
c = -0.956 + 0.22j # for julia set

for  i in range(depth):
    #zs_ = zs*zs + z
    zs_ = zs * zs + c # julia set code
    not_diverged = torch.abs(zs_) < 4.0
    ns += not_diverged
    zs = zs_

fig = plt.figure(figsize=(16, 10))

def processFractal(a):
    a_cyclic = (2*np.pi*a/20).reshape(list(a.shape) + [1])
    img = np.concatenate([10 + 20*np.cos(a_cyclic),
                          30 + 50*np.sin(a_cyclic),
                          155 - 80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

plt.imshow(processFractal(ns.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()



