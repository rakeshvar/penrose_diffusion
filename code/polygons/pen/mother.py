import copy
import math

from code.utils.geom import inscribed_square_halfside
from code.utils.basic import print_tile_stats
from .shapes import circle_tiling
from .xya import PenGrid

TOL = 1e-6

def polar_key(r):
    x, y = r.x, r.y
    radius = round(math.hypot(x, y), 5)
    theta = math.atan2(y, x)
    theta_cw = (-theta + math.pi) % (2*math.pi)
    return (radius, theta_cw)


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

    # Index
    pengrid.rhombuses.sort(key=polar_key)
    for i, rh in enumerate(pengrid.rhombuses):
        rh.index = i


    return pengrid
