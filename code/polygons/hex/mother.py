
from .qrs import HexQRSGrid
from .xya import HexGrid
from code.utils.geom import inscribed_square_halfside
from code.utils.basic import print_tile_stats

def get_hex_mother_tiles(total_halfside, target_hex_side):
    hexagons = HexQRSGrid.from_halfside(target_hex_side, total_halfside)
    hexgrid  = HexGrid(hexagons)

    original_side = hexgrid.side
    hexgrid.scale(target_hex_side / original_side)
    print_tile_stats(hexgrid)
    print(f"Hex Side: Original: {original_side} -> Target: {target_hex_side} scale factor: {original_side / target_hex_side}")
    inscribed_square_halfside(hexgrid)
    return hexgrid
