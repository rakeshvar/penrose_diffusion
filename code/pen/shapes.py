import math
from cmath import exp

from code.pen.base import Fatt, Thin, TriangleGrid, psi

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

triangle_tiling = TriangleGrid([Fatt(A, B, C)])

#----------------------------------------
# Star
#----------------------------------------
A = 0j
B = scale + 0j
C = scale / psi * ejpiby5
t = Fatt(A, B, C)

star_tiling = TriangleGrid([t.rotated(k * two_piby5) for k in range(5)])
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

circle_tiling = TriangleGrid([
    Thin(A1, B, C1),
    Thin(A2, B, C2),
    Thin(A3, B, C3),
    Thin(A4, B, C4),
    Thin(A5, B, C5)])

circle_tiling.add_x_flipped()
