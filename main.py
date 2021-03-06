import torch
print(torch.__version__)
exit(0)
import torchvision.transforms.functional as F
import torch.optim as opt
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from Discriminator import Discriminator
from Generator import Generator
from lib.solvers import anderson, broyden
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILENAME = 'gen1.pth.tar'
DISC_MODEL = 'disc.pth.tar'
LOAD_MODEL = True

lr = 1e-4
image_size = 64
batch_size = 1
num_epochs = 10
features_disc = 32
features_gen = 64
critic_iterations = 3
lambda_ = 10
z_dim = 100

num_branches = 3
num_channels = [3, 3, 3]
f_solver = eval('broyden')
b_solver = eval('broyden')
f_thres = 30
b_thres = 40

transforms = transforms.Compose(
    [transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(
        [0.5 for _ in range(num_channels[0])], [0.5 for _ in range(num_channels[0])]
    )]
)

#dataset = datasets.Celeba(root="dataset/", train=True, transform=transforms, download=True)
dataset = datasets.ImageFolder(root="dataset/celeb_dataset", transform=transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

disc = Discriminator(num_branches, num_channels, features_disc, image_size).to(device)
gen = Generator(num_branches, num_channels, f_solver, b_solver, f_thres, b_thres, features_gen, z_dim).to(device)

opt_disc = opt.Adam(disc.parameters(), lr=lr, betas=(0,0.9))
opt_gen = opt.Adam(gen.parameters(), lr=lr, betas=(0,0.9))


fixed_noise = torch.randn(batch_size, z_dim, 1, 1).to(device)

write_fake = SummaryWriter(f"runs/IGANGP_MNIST/fake")
write_real = SummaryWriter(f"runs/IGANGP_MNIST/real")


def gradient_penalty(disc, real, fake, device="cpu"):
    #interpolated_images = real[0] * alpha + fake[0] * (1 - alpha)

    # Calculate critic scores
    ip = []
    for i in range(num_branches):
        BATCH_SIZE, C, H, W = real[i].shape
        alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
        new = real[i] * alpha + fake[i] * (1 - alpha)
        ip.append(new)

    mixed_scores = disc(ip)

    gp = []
    for i in range(num_branches):
        ms = mixed_scores[i]

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=ip[i],
            outputs=ms,
            grad_outputs=torch.ones_like(ms),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_pen = torch.mean((gradient_norm - 1) ** 2)
        gp.append(gradient_pen)

    return gp


def save_checkpoint(state, filename=FILENAME):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def train():
    step = 0

    if LOAD_MODEL:
        checkpoint = torch.load(FILENAME)
        gen.load_state_dict(checkpoint['state_dict'])
        opt_gen.load_state_dict(checkpoint['optimizer'])
        checkpoint1 = torch.load(DISC_MODEL)
        disc.load_state_dict(checkpoint1['state_dict'])
        opt_disc.load_state_dict(checkpoint1['optimizer'])

    gen.train()
    disc.train()

    for epoch in range(num_epochs):
        cp = {'state_dict': gen.state_dict(), 'optimizer': opt_gen.state_dict()}
        save_checkpoint(cp, FILENAME)
        cp1 = {'state_dict': disc.state_dict(), 'optimizer': opt_disc.state_dict()}
        save_checkpoint(cp1, DISC_MODEL)

        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            batch_size = real.shape[0]

            real_x = [real]
            for i in range(1, num_branches):
                bsz, C, H, W = real_x[-1].shape
                new = F.resize(real, (H//2, W//2))
                real_x.append(new)

            for _ in range(critic_iterations):
                noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
                fake = gen(noise)
                print(fake[0])
                exit(0)

                #fake, jac_loss = gen(noise)
                #jac_loss = jac_loss.mean()

                disc_real = disc(real_x)
                disc_fake = disc(fake)
                gp = gradient_penalty(disc, real_x, fake, device=device)
                lossD = 0
                for i in range(0, 1):
                    dr = disc_real[i].reshape(-1)
                    df = disc_fake[i].reshape(-1)
                    ld = -(torch.mean(dr) - torch.mean(df)) + lambda_ * gp[i]
                    lossD += ld
                #lossD /= num_branches
                disc.zero_grad()
                lossD.backward(retain_graph=True)
                opt_disc.step()

            output = disc(fake)
            lossG = 0
            for i in range(0, 1):
                op = output[i].reshape(-1)
                lg = -torch.mean(op)
                lossG += lg
            # lossG /= num_branches
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                              Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    fake = fake[0]
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:8], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:8], normalize=True
                    )

                    write_real.add_image("Real", img_grid_real, global_step=step)
                    write_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1


def test():
    if LOAD_MODEL:
        checkpoint = torch.load(FILENAME)
        gen.load_state_dict(checkpoint['state_dict'])
    gen.eval()
    noise = torch.randn(1, z_dim, 1, 1).to(device)
    fake = gen(noise)
    print(fake[0])
    fake = fake[0].detach().cpu()
    fake = np.squeeze(fake, axis=0)
    fake = fake.permute(1, 2, 0)
    plt.imshow(fake)
    exit(0)


if __name__ == '__main__':
    train()


