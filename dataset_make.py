import numpy as np
from tqdm import tqdm

from data_generator import Generator5, Generator6
from data_imageset import ImageSet
from utils import npz_stats


def calculate_num_copies_per_sample(num_tiles, target_mb=1024., num_classes=70, samples_per_class=20):
    """
    Calculates how many random augmentations per image are needed  to hit the target file size.
    """
    bytes_per_val = 2    # float16
    columns = 3.5         # xya (float16) + colors (uint8)
    total_base_images = num_classes * samples_per_class

    bytes_per_sample = num_tiles * columns * bytes_per_val
    target_bytes = target_mb * (1024**2)

    num_copies_per_sample = int(target_bytes / (total_base_images * bytes_per_sample))
    return num_copies_per_sample

def generate_and_save(generator, samples_per_class, num_copies, prefix):
    num_tiles = generator.num_tiles
    num_classes = generator.imageset.num_classes
    total_samples = num_classes * samples_per_class * num_copies
    print(f"\nTotal Samples = {num_classes} classes * {samples_per_class} samples * {num_copies} copies = {total_samples}.")

    print(f"xya: ({total_samples}, {num_tiles}, 3) [float16]")
    print(f"colors: ({total_samples}, {num_tiles}) [uint8]")
    print(f"labels: ({total_samples},) [uint8]")
    xya = np.zeros((total_samples, num_tiles, 3), dtype=np.float16)
    colors = np.zeros((total_samples, num_tiles), dtype=np.uint8)
    labels = np.zeros((total_samples,), dtype=np.uint8)

    print("\nFilling data...")
    i = 0
    for _ in tqdm(range(num_copies)):
        for s_id in range(samples_per_class):
            for c_id in range(num_classes):
                sample_data = generator.get_sample(c_id, s_id, rotate_mask=False)

                xyac_i = sample_data['xyac']
                xya[i] = xyac_i[:, :3]
                colors[i] = xyac_i[:, 3]
                labels[i] = sample_data['label']

                i += 1
        
    def save_npz(filename, _xya, _colors, _labels):
        print(f"Saving file: {filename} ")
        np.savez(filename,
                 xya=_xya,
                 colors=_colors,
                 labels=_labels,
                 symmetry=generator.symmetry,
                 side=generator.unit_side,
                 num_tiles=generator.num_tiles,
                 num_classes=num_classes,
                 class_lookup=generator.imageset.class_name_to_id | 
                              generator.imageset.class_id_to_name )
        print(f"Saved!")

    ffull = f"{prefix}_t{num_tiles:03d}_c{num_copies:02d}_u{round(100*generator.unit_side):02d}.npz"
    save_npz(ffull, xya, colors, labels)

    return ffull

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} symmetry num_tiles num_copies [unit_side]")
        exit(0)
    

    folder = "MPEG7/gifs"
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