import numpy as np
import torch
import matplotlib.pyplot as plt

print("Pytorch Version: ", torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X, Y = np.mgrid[-4:4:0.05, -4:4:0.05]

X = torch.Tensor(X)
Y = torch.Tensor(Y)

x = X.to(device)
y = Y.to(device)

amp = 2
freq = 10
phase = np.pi/2


#z = amp * torch.sin(freq * (x + y) + phase) # 2d sin curve
z = torch.exp(-(x**2+y**2)/2) * amp * torch.sin(freq * (x + y) + phase) #2d sin curve

plt.imshow(z.cpu().numpy())
plt.tight_layout()
plt.show()

