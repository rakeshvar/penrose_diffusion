import sys
from code.pen.shapes import circle_tiling
from code.pen.pregen import get_pen_mother_tiles
import copy

try:
    halfside = float(sys.argv[1])
    pen_side = float(sys.argv[2])
except IndexError:
    print(f"Usage: python {sys.argv[0]} <halfside> <pen_side>")
    print("Using default values")
    pen_side = 1/10.
    halfside = 2.

print(f"\thalfside: {halfside}")
print(f"\tpen_side: {pen_side}")

mtiles = get_pen_mother_tiles(halfside, pen_side)

from code.pen.svg import save_svg
save_svg(mtiles, f"pen_mother_tiles_{len(mtiles)}.svg")

# Save original tiling for comparison
comparisiongrid = copy.deepcopy(circle_tiling)
comparisiongrid.inflate(5)
save_svg(comparisiongrid, f"library/tiles/pen_mother_tiles_{len(comparisiongrid)}.svg")