# example3.py
import math
from pen_base import PenroseP3, Fatt, psi, save_svg

# A star with five-fold symmetry

# The Golden ratio
phi = 1 / psi
scale = 1000
config = {'draw-arcs': True,
          'Aarc-colour': '#ff5e25',
          'Carc-colour': 'none',
          'Stile-colour': '#090',
          'Ltile-colour': '#9f3',
          'rotate': math.pi/2}

theta = 2*math.pi / 5
rot = math.cos(theta) + 1j*math.sin(theta)

B1 = scale
p = B1 * rot
q = p*rot

C5 = -scale * phi
r = C5 / rot
s = r / rot
A = [0]*5
B = [scale, p, p, q, q]
C = [s, s, r, r, C5]

tiling = PenroseP3([Fatt(*v) for v in zip(A, B, C)])
tiling.inflate(times=5)

save_svg(tiling, config, 'example3.svg')

