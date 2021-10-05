import torch.nn as nn
from collections import OrderedDict


class Discriminator(nn.Module):
    def __init__(self, num_branches, num_channels, features_d, original_img_size):
        super(Discriminator, self).__init__()
        self.num_branches = num_branches
        self.num_channels = num_channels

        self.classifier = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(num_channels[i], features_d * 4, kernel_size=3, stride=1, padding=0)),
                ('inorm1', nn.InstanceNorm2d(features_d*4)),
                ('lr1', nn.LeakyReLU(0.2)),
                ('conv2', nn.Conv2d(features_d * 4, features_d * 8, kernel_size=3, stride=1, padding=0)),
                ('inorm2', nn.InstanceNorm2d(features_d * 8)),
                ('lr2', nn.LeakyReLU(0.2)),
                ('conv3', nn.Conv2d(features_d * 8, 1, kernel_size=3, stride=1, padding=0))
            ])) for i in range(num_branches)])

        self.final = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('linear', nn.Linear( (original_img_size // (2 ** i) - 6) * (original_img_size // (2 ** i) - 6), 1))
            ]))for i in range(num_branches)
        ])

    def forward(self, x_list):
        ys = []
        for i in range(self.num_branches):
            y = self.classifier[i](x_list[i])
            y = y.reshape([y.shape[0], y.shape[1] * y.shape[2] * y.shape[3]])
            y = self.final[i](y)
            ys.append(y)
        return ys

'''
from lib.solvers import anderson, broyden
import torch.optim as opt
from Generator import Generator

num_branches = 1
num_channels = [1]
features_d = 32
img_size = 28
batch_size = 1

f_solver = eval('broyden')
b_solver = eval('broyden')
f_thres = 3
b_thres = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

disc = Discriminator(num_branches, num_channels, features_d, img_size).to(device)
disc.train()
gen = Generator(num_branches, num_channels, f_solver, b_solver, f_thres, b_thres, 32).to(device)
gen.train()


noise = torch.randn(batch_size, num_channels[0], img_size, img_size).to(device)
fake, jac_loss = gen(noise)
op = disc(fake)
print(op)

opt_disc = opt.Adam(disc.parameters(), lr=1e-4)
lossD = torch.mean(op[0].reshape(-1))
print('Disc Loss:', lossD)
disc.zero_grad()
lossD.backward(retain_graph=True)
print("Disc Backward done")
opt_disc.step()
print("Disc Finished")

opt_gen = opt.Adam(gen.parameters(), lr=1e-4)
output = disc(fake)[0].reshape(-1)
lossG = -torch.mean(output)
print('Gen Loss:', lossG)
gen.zero_grad()
lossG.backward()
print("Gen Backward done")
opt_gen.step()
'''