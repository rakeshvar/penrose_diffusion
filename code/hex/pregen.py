
from code.hex.base import HexagonGrid, HexGrid
from code.utils import print_tile_stats, inscribed_square_halfside

def get_hex_mother_tiles(total_halfside, target_hex_side):
    hexagons = HexagonGrid.from_halfside(target_hex_side, total_halfside)
    hexgrid  = HexGrid(hexagons)

    original_side = hexgrid.side
    hexgrid.scale(target_hex_side / original_side)
    print_tile_stats(hexgrid)
    print(f"Hex Side: Original: {original_side} -> Target: {target_hex_side} scale factor: {original_side / target_hex_side}")
    inscribed_square_halfside(hexgrid)
    return hexgrid
