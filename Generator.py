import torch.nn as nn
import torch
from lib.jacobian import jac_loss_estimate
import torch.autograd as autograd
from lib.optimization import VariationalHidDropout2d
from lib.MDEQ import MDEQModule
from lib.utils import list2vec, vec2list
from lib.solvers import anderson, broyden


class Generator(nn.Module):
    def __init__(self, num_branches, num_channels, f_solver, b_solver, f_thres, b_thres, feature_g, z_dim):
        super(Generator, self).__init__()
        self.num_branches = num_branches
        self.num_channels = num_channels
        self.f_solver = f_solver
        self.b_solver = b_solver
        self.f_thres = f_thres
        self.b_thres = b_thres

        self.init_gen = nn.Sequential(
            # input: N x z_dim x 1 x 1
            self._block(z_dim, feature_g * 16, 4, 1, 0),  # N x f_g*16 x 4 x 4
            self._block(feature_g * 16, feature_g * 8, 4, 2, 1),
            self._block(feature_g * 8, feature_g * 4, 4, 2, 1),
            self._block(feature_g * 4, feature_g * 2, 4, 2, 1),
            self._block(feature_g * 2, num_channels[0], 4, 2, 1)
        )

        self.fullstage = self._make_stage(num_branches, num_channels, feature_g)
        self.iodrop = VariationalHidDropout2d(0.0)

        self.hook = None

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def _make_stage(self, num_branches, num_channels, feature_g):
        """
        Build an MDEQ block with the given hyperparameters
        """
        return MDEQModule(num_branches, num_channels, feature_g)

    def forward(self, x):
        x = self.init_gen(x)
        # print(x.shape)

        x_list = [x]
        for i in range(1, self.num_branches):
            bsz, C, H, W = x_list[-1].shape
            x_list.append(torch.zeros(bsz, C, H//2, W//2).to(x)) #Change this later according to image resize
        z_list = [torch.zeros_like(element) for element in x_list]
        z1 = list2vec(z_list)
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z_list]

        func = lambda z: list2vec(self.fullstage(vec2list(z, cutoffs), x_list))

        # jac_loss = torch.tensor(0.0).to(x)

        with torch.no_grad():
            res = self.f_solver(func, z1, threshold=self.f_thres, stop_mode='abs')
            z1 = res['result']

        new_z1 = z1
        new_z1 = func(z1.requires_grad_())
        
        # jac_loss = jac_loss_estimate(new_z1, z1)
        
        # if self.training:
        #     def backward_hook(grad):
        #         if self.hook is not None:
        #             self.hook.remove()
        #             torch.cuda.synchronize()
        #         result = self.b_solver(lambda y: autograd.grad(new_z1, z1, y, retain_graph=True)[0] + grad,
        #                             torch.zeros_like(grad),
        #                             threshold=self.b_thres, stop_mode='abs', name="backward")
        #         print(result['result'].shape)
        #         return result['result']

        #     self.hook = new_z1.register_hook(backward_hook)
        
        y_list = self.iodrop(vec2list(new_z1, cutoffs))
        # return y_list, jac_loss.view(1, -1)
        return y_list

'''
num_branches = 2
num_channels = [1, 1]
f_solver = eval('broyden')
b_solver = eval('broyden')
f_thres = 30
b_thres = 40
feature_g = 64
gen = Generator(num_branches, num_channels,f_solver, b_solver, f_thres, b_thres, feature_g)

x = torch.randn(8, 1, 28, 28)
x_list = [x]
for i in range(1, num_branches):
    bsz, C, H, W = x_list[-1].shape
    x_list.append(torch.randn(bsz, C, H//2, W//2))
op, loss = gen(x)
print(len(op[0]))
'''

