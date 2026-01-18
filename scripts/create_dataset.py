from code.data.generator import Generator5, Generator6
from code.data.imageset import ImageSet
from code.utils import npz_stats

from code.data.create import generate_and_save

import sys
if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} symmetry num_tiles num_copies [unit_side]")
    exit(0)


folder = "library/MPEG7/gifs"
imageset = ImageSet(folder)
SAMPLES_PER_CLASS = 20

SYMMETRY = int(sys.argv[1])
NUM_TILES = int(sys.argv[2])
NUM_COPIES = int(sys.argv[3])

if len(sys.argv) > 4:
    UNIT_SIDE = float(sys.argv[4])
else:                               # Aiming for a std of 1. for x, y
    if SYMMETRY == 6:
        UNIT_SIDE = .18 * (96/NUM_TILES)**.5
    else:
        UNIT_SIDE = .1 * (512/NUM_TILES)**.5
UNIT_SIDE = round(UNIT_SIDE, 2)

print(f"SYMMETRY: {SYMMETRY} \nNUM_TILES: {NUM_TILES} \nNUM_COPIES: {NUM_COPIES} \nUNIT_SIDE: {UNIT_SIDE}")


if SYMMETRY == 6:
    gen6 = Generator6(imageset, num_tiles=NUM_TILES, target_halfside=5., unit_side=UNIT_SIDE)
    file = generate_and_save(gen6, SAMPLES_PER_CLASS, NUM_COPIES, prefix="datasets/hex")
else:
    gen5 = Generator5(imageset, num_tiles=500, target_halfside=5., unit_side=UNIT_SIDE)
    file = generate_and_save(gen5, SAMPLES_PER_CLASS, NUM_COPIES, prefix="datasets/pen")

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