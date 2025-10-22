# python -m pen_examples.example1

import math
from pen_base import PenroseP3, Fatt, psi, save_svg

# A simple example starting with a BL tile

scale = 1000.

theta = 2*math.pi / 5
exp_j_theta = math.cos(theta) + 1j*math.sin(theta)
A = scale/2 + 0j
B = -scale/2 * exp_j_theta
C = -scale/2 / psi + 0j

config = {'draw-arcs': False, 'reflect-x': False, 'draw-rhombuses': False}

tiling = PenroseP3([Fatt(A, B, C)])

for n in range(7):
    tiling.inflate(times=1)
    save_svg(tiling=tiling, additional_config=config, filename=f'example1-{n}.svg')
