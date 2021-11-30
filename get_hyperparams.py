import torch
from Generator import Generator
from lib.solvers import anderson, broyden


checkpoint = torch.load('runs/pokemon/11-28-2021-161708/gen1.pth.tar')
network = Generator(3, [3,3,3], eval('broyden'), eval('broyden'), 30, 40, 64, 100).to('cuda')
network.load_state_dict(checkpoint['state_dict'])
optimizer = torch.optim.Adam(network.parameters(), lr=0.01, betas=(0.9,0.99))
optimizer.load_state_dict(checkpoint['optimizer'])

for elements in optimizer.state_dict():
  print(elements, "\t", optimizer.state_dict()[elements])