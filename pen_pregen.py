import copy
import math

from utils import deg
from pen_shapes import circle_tiling

TOL = 1e-6

def get_pen_mother_tiles(target_count):
    tiling = copy.deepcopy(circle_tiling)
    print(f"Starting with {len(tiling.elements)} tiles.")

    # Get at least 16 times the target count to ensure enough tiles
    # for random translation and rotation
    # 2 is for removing mirror images
    mother_count = 2 * 16 * target_count
    while len(tiling.elements) < mother_count:
        tiling.inflate(1)
        print(f"Inflated to {len(tiling.elements)} tiles.")

    rhombus_tiles = tiling.get_rhombus_tiles()
    # distances_to_the_closest_neighbor(rhombus_tiles)

    sides = sorted(set(TOL*round(rh.side_length/TOL) for rh in rhombus_tiles))
    assert len(sides) == 1, f"Expected same tile sizes, got {sides}"
    side = sides.pop()

    # So that approximately target_count tiles fit in unit area
    target_side = 1/math.sqrt(target_count)

    # Scale tiles accordingly
    scale_factor = target_side / side
    for rh in rhombus_tiles:
        rh.center *= scale_factor
        rh.side *= scale_factor

    # Calculate and print some tile statistics
    xmin = min(rh.center.real for rh in rhombus_tiles)
    xmax = max(rh.center.real for rh in rhombus_tiles)
    ymin = min(rh.center.imag for rh in rhombus_tiles)
    ymax = max(rh.center.imag for rh in rhombus_tiles)
    area = (xmax - xmin) * (ymax - ymin)
    set_of_tilts = sorted(set(round(deg(rh.tilt)) for rh in rhombus_tiles))
    print(f"""Tile 
          Count: {len(rhombus_tiles)}
          Side: {side:.4f} -> {target_side:.4f}
          Area: {area:.4f}
          Tilts: {set_of_tilts}
          xmin: {xmin:.4f} xmax: {xmax:.4f}
          ymin: {ymin:.4f} ymax: {ymax:.4f}
    """)

    return rhombus_tiles

if __name__ == '__main__':
    import sys
    from pen_svg import save_svg

    target_count=100 if len(sys.argv) < 2 else int(sys.argv[1])
    mtiles = get_pen_mother_tiles(target_count)
    save_svg(mtiles, f"mother_tiles_{target_count}.svg")

    # Save original tiling for comparison
    tiling = copy.deepcopy(circle_tiling)
    tiling.inflate(5)
    save_svg(tiling.elements, f"mother_tiles_{target_count}_orig.svg")