
from hex_base import HexagonGrid, HexGrid
from utils import print_tile_stats, inscribed_square_halfside

def get_hex_mother_tiles(total_halfside, target_hex_side):
    hexagons = HexagonGrid.from_halfside(target_hex_side, total_halfside)
    hexgrid  = HexGrid(hexagons)

    original_side = hexgrid.side
    hexgrid.scale(target_hex_side / original_side)
    print_tile_stats(hexgrid)
    print(f"Hex Side: Original: {original_side} -> Target: {target_hex_side} scale factor: {original_side / target_hex_side}")
    inscribed_square_halfside(hexgrid)
    return hexgrid

if __name__ == '__main__':
    import sys
    try:
        halfside = float(sys.argv[1])
        hex_side = float(sys.argv[2])
    except:
        print("Usage: python hex_pregen.py <halfside> <hex_side>")
        print("Using default values")
        hex_side = 1/10.
        halfside = 3.
    
    print(f"\thalfside: {halfside}")
    print(f"\thex_side: {hex_side}")

    mtiles = get_hex_mother_tiles(halfside, hex_side)

    from hex_svg import save_svg
    save_svg(mtiles, f"hex_mother_tiles_{len(mtiles)}.svg")