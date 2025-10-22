# example4.py
import math
from pen_base import PenroseP3, Thin, save_svg

# A "sun"


piby5 = math.pi / 5
ejpiby5 = math.cos(piby5) + 1j*math.sin(piby5)

B = 0 + 0j      # Tips at origin

A1 = 1 + 0.j
C1 = C2 = A1 * ejpiby5
A2 = A3 = C1 * ejpiby5
C3 = C4 = A3 * ejpiby5
A4 = A5 = C4 * ejpiby5
C5 = -A1

tiling = PenroseP3([
    Thin(A1, B, C1), 
    Thin(A2, B, C2),
    Thin(A3, B, C3), 
    Thin(A4, B, C4),
    Thin(A5, B, C5)]) # All above the x-axis

tiling.add_conjugate_elements() # Reflect to get those below x-axis

tiling.inflate(times=10)
save_svg(tiling, config, 'example4.svg')
