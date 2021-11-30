import numpy as np
from matplotlib.image import imread, imsave
from os import path
"""
Take a Tensorboard image grid and separate it into separate image files.
"""

imfile = '/home/ehrensam/Downloads/beta0.png'
grid = imread(imfile)
h, w, _ = grid.shape
for i in range(h//64):
    for j in range(w//64):
        img = grid[i*66+2:(i+1)*66,j*66+2:(j+1)*66]
        imsave(f'out-beta0/{i}_{j}.png', img)
