import copy

from utils import print_tile_stats, inscribed_square_halfside
from pen_shapes import circle_tiling
from pen_base import PenGrid

TOL = 1e-6

def get_pen_mother_tiles(target_halfside, target_pen_side):
    trianglegrid = copy.deepcopy(circle_tiling)
    target_elements = target_halfside / target_pen_side

    while True:
        tiss = inscribed_square_halfside(trianglegrid)/target_elements
        print(f"Target Inscribed side: {tiss:6.1f} Scaled side: {trianglegrid.side:7.1f}")
        if trianglegrid.side < tiss:
            break
        trianglegrid.inflate(1)

    pengrid = PenGrid(trianglegrid)

    original_side = pengrid.side
    pengrid.scale(target_pen_side/original_side)
    print_tile_stats(pengrid)
    inscribed_square_halfside(pengrid)

    print(f"Pen Side: Original: {original_side} -> Final: {pengrid.side} scale factor: {original_side / pengrid.side}")

    return pengrid

if __name__ == '__main__':
    import sys
    try:
        halfside = float(sys.argv[1])
        pen_side = float(sys.argv[2])
    except:
        print(f"Usage: python {sys.argv[0]} <halfside> <pen_side>")
        print("Using default values")
        pen_side = 1/10.
        halfside = 2.

    print(f"\thalfside: {halfside}")
    print(f"\tpen_side: {pen_side}")

    mtiles = get_pen_mother_tiles(halfside, pen_side)

    from pen_svg import save_svg
    save_svg(mtiles, f"pen_mother_tiles_{len(mtiles)}.svg")

    # Save original tiling for comparison
    comparisiongrid = copy.deepcopy(circle_tiling)
    comparisiongrid.inflate(5)
    save_svg(comparisiongrid, f"pen_mother_tiles_{len(comparisiongrid)}.svg")