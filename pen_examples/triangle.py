
import math
from pen_base import PenroseP3, Fatt, psi
from pen_svg import save_svg


scale = 1000.

theta = 2*math.pi / 5
exp_j_theta = math.cos(theta) + 1j*math.sin(theta)
A = scale/2 + 0j
B = -scale/2 * exp_j_theta
C = -scale/2 / psi + 0j

# A simple example starting with a BL tile
triangle_tiling = PenroseP3([Fatt(A, B, C)])
triangle_tiling.inflate(0)

config1 = {'draw-arcs': False, 'reflect-x': False, 'draw-rhombuses': False}

if __name__ == '__main__':
    save_svg(tiling=triangle_tiling, additional_config=config1, filename='example1.svg')