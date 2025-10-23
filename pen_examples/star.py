# example3.py
import math
import cmath
from pen_base import PenroseP3, Fatt, psi
from pen_svg import save_svg

# A star with five-fold symmetry

# The Golden ratio
scale = 1000


t = Fatt(0j, scale+0j, scale/psi*cmath.exp(1j * math.pi/5))
tiles = [t.rotated(2 * math.pi * k / 5) for k in range(5)]

star_tiling = PenroseP3(tiles)
star_tiling.add_x_flipped()
star_tiling.inflate(times=4)

config3 = {'draw-arcs': True,
          'Aarc-colour': '#ff5e25',
          'Carc-colour': 'none',
          'Stile-colour': '#090',
          'Ltile-colour': '#9f3',
          'rotate': math.pi/2}

if __name__ == '__main__':
    save_svg(star_tiling, config3, 'example3.svg')
