# example2.py
import math
from pen_base import PenroseP3, Thin, save_svg

# A "sun", with randomly-coloured tiles and including arc paths

scale = 100
config={'draw-arcs': True, 'tile-opacity': 0.6,
        'random-tile-colours': True, 'Aarc-colour': '#f44',
        'Carc-colour': '#44f'}

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

tiling = PenroseP3([
    Thin(A1, B, C1), 
    Thin(A2, B, C2),
    Thin(A3, B, C3), 
    Thin(A4, B, C4),                          
    Thin(A5, B, C5)])

tiling.inflate(4)
save_svg(tiling=tiling, additional_config=config, scale=scale, filename='example2.svg')
