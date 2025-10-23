# example2.py
import math
from pen_base import PenroseP3, Thin
from pen_svg import save_svg

# A "sun", with randomly-coloured tiles and including arc paths

scale = 1000
theta = math.pi / 5
alpha = math.cos(theta)
rot = math.cos(theta) + 1j*math.sin(theta)
A1 = scale + 0.j
B = 0 + 0j
C1 = C2 = A1 * rot
A2 = A3 = C1 * rot
C3 = C4 = A3 * rot
A4 = A5 = C4 * rot
C5 = -A1

circle_tiling = PenroseP3([
    Thin(A1, B, C1), 
    Thin(A2, B, C2),
    Thin(A3, B, C3), 
    Thin(A4, B, C4),                          
    Thin(A5, B, C5)])

circle_tiling.add_x_flipped()
circle_tiling.inflate(0)

config2={'draw-arcs': True, 'tile-opacity': 0.6,
        'random-tile-colours': True, 'Aarc-colour': '#f44',
        'Carc-colour': '#44f'}

if __name__ == '__main__':
    save_svg(tiling=circle_tiling, additional_config=config2, filename='example2.svg')
