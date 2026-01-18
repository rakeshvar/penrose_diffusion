import sys
from code.hex.pregen import get_hex_mother_tiles

try:
    halfside = float(sys.argv[1])
    hex_side = float(sys.argv[2])
except IndexError:
    print(f"Usage: python {sys.argv[0]} <halfside> <hex_side>")
    print("Using default values")
    hex_side = 1/10.
    halfside = 3.

print(f"\thalfside: {halfside}")
print(f"\thex_side: {hex_side}")

mtiles = get_hex_mother_tiles(halfside, hex_side)

from code.hex.svg import save_svg
save_svg(mtiles, f"library/tiles/hex_mother_tiles_{len(mtiles)}.svg")