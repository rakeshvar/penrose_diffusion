import copy

from code.utils.geom import inscribed_square_halfside
from code.utils.basic import print_tile_stats
from code.pen.shapes import circle_tiling
from code.pen.base import PenGrid

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

    print(f"Pen Side: Original: {original_side} -> Final: {pengrid.side} scale factor: {original_side / pengrid.side:.2f}")

    return pengrid
