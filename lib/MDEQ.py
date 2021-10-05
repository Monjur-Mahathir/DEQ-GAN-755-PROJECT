import torch.nn as nn
from collections import OrderedDict


class GeneratorBlock(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super().__init__()
        #input_size: N x z_dim x H x W

        self.conv1 = nn.Conv2d(z_dim, features_g * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(features_g * 4)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(features_g * 4, features_g * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(features_g * 2)

        self.conv3 = nn.Conv2d(features_g * 2, channels_img, kernel_size=(3, 3), padding=(1, 1))

        self.transpose1 = nn.ConvTranspose2d(channels_img, features_g * 2, kernel_size=(4, 4), stride=(2, 2),
                                             padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(features_g * 2)

        self.transpose2 = nn.ConvTranspose2d(features_g * 2, features_g * 4, kernel_size=(4, 4), stride=(2, 2),
                                             padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(features_g * 4)

        self.transpose3 = nn.ConvTranspose2d(features_g * 4, z_dim, kernel_size=(3, 3), padding=(1, 1))

        self.tanh = nn.Tanh()

    def forward(self, z, injection=None):
        if injection is None:
            injection = 0
        z = z + injection

        first_block = self.relu(self.bn1(self.conv1(z)))
        second_block = self.relu(self.bn2(self.conv2(first_block)))
        first_t_block = self.conv3(second_block)

        third_block = self.relu(self.bn3(self.transpose1(first_t_block)))
        fourth_block = self.relu(self.bn4(self.transpose2(third_block)))
        second_t_block = self.tanh(self.transpose3(fourth_block))

        # second_t_block = self.tanh(fourth_block)

        return second_t_block


class BranchNet(nn.Module):
    def __init__(self, blocks):
        """
        The residual block part of each resolution stream
        """
        super().__init__()
        self.blocks = blocks

    def forward(self, x, injection=None):
        blocks = self.blocks
        y = blocks[0](x, injection)
        for i in range(1, len(blocks)):
            y = blocks[i](y)
        return y


class DownsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res):
        super(DownsampleModule, self).__init__()
        # downsample (in_res=j, out_res=i)
        convs = []
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        self.level_diff = level_diff = out_res - in_res

        kwargs = {"kernel_size": 3, "stride": 2, "padding": 1, "bias": False}
        for k in range(level_diff):
            intermediate_out = out_chan if k == (level_diff - 1) else inp_chan
            components = [('conv', nn.Conv2d(inp_chan, intermediate_out, **kwargs)),
                          ('bnorm', nn.BatchNorm2d(intermediate_out))]
            if k != (level_diff - 1):
                components.append(('relu', nn.ReLU()))
            if k == (level_diff - 1):
                components.append(('tanh', nn.Tanh()))
            convs.append(nn.Sequential(OrderedDict(components)))
        self.net = nn.Sequential(*convs)

    def forward(self, x):
        return self.net(x)


class UpsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res):
        """
        An upsample step from resolution j (with in_res) to resolution i (with out_res).
        Simply a 1x1 convolution followed by an interpolation.
        """
        super(UpsampleModule, self).__init__()
        # upsample (in_res=j, out_res=i)
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        self.level_diff = level_diff = in_res - out_res

        self.net = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(inp_chan, out_chan, kernel_size=1, bias=False)),
            ('gnorm', nn.BatchNorm2d(out_chan)),
            ('upsample', nn.Upsample(scale_factor=2 ** level_diff, mode='nearest'))]))

    def forward(self, x):
        return self.net(x)


class MDEQModule(nn.Module):
    def __init__(self, num_branches, num_channels, features_g):
        super(MDEQModule, self).__init__()
        self.num_branches = num_branches
        self.num_channels = num_channels

        self.branches = self._make_branches(num_branches, num_channels, features_g)

        self.fuse_layers = self._make_fuse_layers()
        self.post_fuse_layers = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('relu', nn.ReLU(False)),
                ('conv', nn.Conv2d(num_channels[i], num_channels[i], kernel_size=1, bias=False)),
                ('gnorm', nn.BatchNorm2d(num_channels[i]))
            ])) for i in range(num_branches)])

    def _make_one_branch(self, branch_index, num_channels, features_g):
        """
        Make a specific branch indexed by `branch_index`. This branch contains `num_blocks` residual blocks of type `block`.
        """
        n_channel = num_channels[branch_index]
        return GeneratorBlock(n_channel, n_channel, features_g)

    def _make_branches(self, num_branches, num_channels, features_g):
        """
        Make the residual block (s; default=1 block) of MDEQ's f_\theta layer. Specifically,
        it returns `branch_layers[i]` gives the module that operates on input from resolution i.
        """
        #branch_layers = [self._make_one_branch(num_channels, features_g) for i in range(num_branches)]
        branch_layers = [self._make_one_branch(i, num_channels, features_g) for i in range(num_branches)]
        return nn.ModuleList(branch_layers)

    def _make_fuse_layers(self):
        """
        Create the multiscale fusion layer (which does simultaneous up- and downsamplings).
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_channels = self.num_channels
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []  # The fuse modules into branch #i
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(None)  # Identity if the same branch
                else:
                    module = UpsampleModule if j > i else DownsampleModule
                    fuse_layer.append(module(num_channels, in_res=j, out_res=i))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        # fuse_layers[i][j] gives the (series of) conv3x3s that convert input from branch j to branch i
        return nn.ModuleList(fuse_layers)

    def forward(self, x, injection):
        """
        The two steps of a multiscale DEQ module (see paper): a per-resolution residual block and
        a parallel multiscale fusion step.
        """
        if injection is None:
            injection = [0] * len(x)
        if self.num_branches == 1:
            return [self.branches[0](x[0], injection[0])]

        # Step 1: Per-resolution residual block
        x_block = []
        for i in range(self.num_branches):
            x_block.append(self.branches[i](x[i], injection[i]))

        # Step 2: Multiscale fusion
        x_fuse = []
        for i in range(self.num_branches):
            y = 0
            # Start fusing all #j -> #i up/down-samplings
            for j in range(self.num_branches):
                y += x_block[j] if i == j else self.fuse_layers[i][j](x_block[j])
            x_fuse.append(self.post_fuse_layers[i](y))
        return x_fuse