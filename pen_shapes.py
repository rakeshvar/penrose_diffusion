import math
from cmath import exp

from pen_base import Fatt, Thin, PenroseP3, psi

two_piby5 = 2 * math.pi / 5
ej2piby5 = exp(1j * two_piby5)
piby5 = math.pi / 5
ejpiby5 = exp(1j * piby5)

scale = 1000.

#----------------------------------------
# Triange
#----------------------------------------
A = scale/2 + 0j
B = -scale / 2 * ej2piby5
C = -scale/2 / psi + 0j

triangle_tiling = PenroseP3([Fatt(A, B, C)])

#----------------------------------------
# Star
#----------------------------------------
A = 0j
B = scale + 0j
C = scale / psi * ejpiby5
t = Fatt(A, B, C)

star_tiling = PenroseP3([t.rotated(k * two_piby5) for k in range(5)])
star_tiling.add_x_flipped()

#----------------------------------------
# Circle
#----------------------------------------
A1 = scale + 0.j
B = 0 + 0j
C1 = C2 = A1 * ejpiby5
A2 = A3 = C1 * ejpiby5
C3 = C4 = A3 * ejpiby5
A4 = A5 = C4 * ejpiby5
C5 = -A1

circle_tiling = PenroseP3([
    Thin(A1, B, C1),
    Thin(A2, B, C2),
    Thin(A3, B, C3),
    Thin(A4, B, C4),
    Thin(A5, B, C5)])

circle_tiling.add_x_flipped()

#----------------------------------------
# Configurations
#----------------------------------------

configs = [
    {'draw-arcs': False},
    
    {'tile-opacity': 1.0, 
     'Aarc-colour': '#f04040',
     'Carc-colour': '#4040f0'
    },

    {
     'Aarc-colour': '#ff5e25',
     'Carc-colour': 'none',
     'Stile-colour': '#009000',
     'Ltile-colour': '#90f030',
    }
]

if __name__ == '__main__':
    from pen_svg import save_svg

    for j in range(5):
        triangle_tiling.inflate(1)
        circle_tiling.inflate(1)
        star_tiling.inflate(1)

        for i, config in enumerate(configs):
            save_svg(triangle_tiling, f'pics/triangle_{i}_{j}.svg', config, 50)
            save_svg(star_tiling, f'pics/star_{i}_{j}.svg', config, 50)
            save_svg(circle_tiling, f'pics/circle_{i}_{j}.svg', config, 50)
