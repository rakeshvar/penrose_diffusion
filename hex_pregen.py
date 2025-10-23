import sys
from collections import Counter
from hex_base import *
from hex_svg import save_svg

def get_hex_mother_tiles(target_count):
    hexagons = HexagonGrid.from_count(target_count * 2)
    hexgrid  = HexGrid(hexagons)
    print(f"Generated {len(hexgrid)} hexagon mother tiles for target count {target_count}.")

    target_side = 1 / math.sqrt(target_count) #* (3**0.25))
    original_side = hexgrid.side
    hexgrid.scale(target_side / original_side)

    # Calculate and print some tile statistics
    xs = [h.x for h in hexgrid]
    ys = [h.y for h in hexgrid]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    colors = Counter(h.color for h in hexgrid)

    print(f"""Tile 
          Count: {len(hexgrid)}
          Side: {original_side:.4f} -> {target_side:.4f}
          Colors: {dict(colors)}  
          xmin: {xmin:.4f} xmax: {xmax:.4f}   
          ymin: {ymin:.4f} ymax: {ymax:.4f}
    """)

    return hexgrid

if __name__ == '__main__':
    target_count=100 if len(sys.argv) < 2 else int(sys.argv[1])
    mtiles = get_hex_mother_tiles(target_count)
    save_svg(mtiles, f"hex_mother_tiles_{target_count}.svg")