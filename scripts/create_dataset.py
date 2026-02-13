import math
import argparse
from code.data.generator import Generator5, Generator6
from code.data.imageset import ImageSet
from code.utils.basic import npz_stats

from code.data.create import generate_and_save


def calculate_default_unit_side(symmetry, num_tiles):
    """
    Calculate default unit_side based on symmetry and num_tiles.
    So that we get a standard deviation of 1. for x, y
    """
    if symmetry == 6:
        return round(math.sqrt(3.24/num_tiles), 2)
    else:
        return round(math.sqrt(10/num_tiles), 2)

if True:
    parser = argparse.ArgumentParser(
        description='Create Dataset with options.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required positional arguments
    parser.add_argument('symmetry', type=int, help='Symmetry value')
    parser.add_argument('num_tiles', type=int, help='Number of tiles')
    parser.add_argument('num_copies', type=int, help='Number of copies')
    
    # Optional positional argument
    parser.add_argument('unit_side', type=float, nargs='?', default=None,
                        help='Unit side length (auto-calculated if not provided)')

    # Boolean flags with short options
    parser.add_argument('-x', '--no-save-xya', dest='save_xya', action='store_false', 
                        help='Do not save XYA')
    parser.add_argument('-c', '--no-save-colors', dest='save_colors', action='store_false',
                        help='Do not save colors')
    parser.add_argument('-l', '--no-save-labels', dest='save_labels', action='store_false',
                        help='Do not save labels')
    parser.add_argument('-I', '--save-indices', dest='save_indices', action='store_true',
                        help='Save indices')

    parser.set_defaults(save_xya=True, save_colors=True, save_labels=True, save_indices=False)

    args = parser.parse_args()

SYMMETRY = args.symmetry
NUM_TILES = args.num_tiles
NUM_COPIES = args.num_copies

assert SYMMETRY in [5, 6], "SYMMETRY must be 5 or 6"

if args.unit_side is None:
    UNIT_SIDE = calculate_default_unit_side(SYMMETRY, NUM_TILES)
else:
    UNIT_SIDE = round(args.unit_side, 2)

print(f"SYMMETRY: {SYMMETRY} \nNUM_TILES: {NUM_TILES} \nNUM_COPIES: {NUM_COPIES} \nUNIT_SIDE: {UNIT_SIDE}")
print("Saving...")
print(f"\txya: {args.save_xya} \n\tcolors: {args.save_colors} \n\tlabels: {args.save_labels} \n\tindices: {args.save_indices}")

prefix = "datasets/"
prefix += "hex" if SYMMETRY == 6 else "pen"
prefix += "xy" if args.save_xya else ""
prefix += "ind" if args.save_indices else ""
print("Prefix: ", prefix, "\n")



folder = "library/MPEG7/gifs"
imageset = ImageSet(folder)
SAMPLES_PER_CLASS = 20

if SYMMETRY == 6:
    GEN = Generator6
else:
    GEN = Generator5

gen = GEN(imageset, num_tiles=NUM_TILES, target_halfside=5., unit_side=UNIT_SIDE)
file = generate_and_save(gen, SAMPLES_PER_CLASS, NUM_COPIES, prefix=prefix,\
                         save_xya=args.save_xya, save_colors=args.save_colors, 
                         save_labels=args.save_labels, save_indices=args.save_indices)

npz_stats(file)

"""
Pen5
Total  Fatt Thin
1   .618034 .381966
10	      6	4
100	     62	38
162     100	62
1000	618	382

512	    316	196
768	    475	293
1024	633	391
		
52	32	20
104	64	40
207	128	79
414	256	158
828	512	316
809	500	309
828	512	316
"""