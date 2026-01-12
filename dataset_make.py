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
    columnss = 4.5         # xysc (float16) + colors (uint8)
    total_base_images = num_classes * samples_per_class

    bytes_per_sample = num_tiles * columnss * bytes_per_val
    target_bytes = target_mb * (1024**2)

    num_copies_per_sample = int(target_bytes / (total_base_images * bytes_per_sample))
    return num_copies_per_sample

def generate_and_save(generator, num_classes, samples_per_class, num_copies, prefix):
    num_tiles = generator.num_tiles
    total_samples = num_classes * samples_per_class * num_copies
    print(f"\nTotal Samples = {num_classes} classes * {samples_per_class} samples * {num_copies} copies = {total_samples}.")

    print(f"xysc: ({total_samples}, {num_tiles}, 4) [float16]")
    print(f"colors: ({total_samples}, {num_tiles}) [uint8]")
    print(f"labels: ({total_samples},) [uint8]")
    xy = np.zeros((total_samples, num_tiles, 2), dtype=float)
    angles = np.zeros((total_samples, num_tiles), dtype=float)
    colors = np.zeros((total_samples, num_tiles), dtype=np.uint8)
    labels = np.zeros((total_samples,), dtype=np.uint8)

    print("\nFilling data...")
    i = 0
    for _ in tqdm(range(num_copies)):
        for s_id in range(samples_per_class):
            for c_id in range(num_classes):
                sample_data = generator.get_sample(c_id, s_id)

                xyac_i = sample_data['xyac']
                xy[i] = xyac_i[:, :2]
                angles[i] = xyac_i[:, 2]
                colors[i] = xyac_i[:, 3].astype(np.uint8)
                labels[i] = np.uint8(sample_data['label'])

                i += 1
        
    xysc = np.zeros((total_samples, num_tiles, 4), dtype=np.float16)
    xysc[..., :2] = xy
    xysc[..., 2]   = np.sin(angles)
    xysc[..., 3]   = np.cos(angles)

    def save_npz(filename, _xysc, _colors, _labels):
        print(f"Saving file: {filename} ")
        np.savez(filename,
                 xysc=_xysc,
                 colors=_colors,
                 labels=_labels,
                 symmetry=generator.symmetry,
                 side=generator.unit_side,
                 num_tiles=generator.num_tiles)
        print(f"Saved!")

    half_total_samples = total_samples // 2
    fhalf = f"{prefix}_t{num_tiles}_c{num_copies//2}.npz"
    save_npz(fhalf, xysc[:half_total_samples], colors[:half_total_samples], labels[:half_total_samples])
    
    ffull = f"{prefix}_t{num_tiles}_c{num_copies}.npz"
    save_npz(ffull, xysc, colors, labels)

    return fhalf, ffull

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} [symmetry] [<target_mb>=1024] [<num_tiles>=768]")
        exit(0)
    
    SYMMETRY = int(sys.argv[1])

    folder = "MPEG7/gifs"
    imageset = ImageSet(folder)
    NUM_CLASSES = imageset.num_classes
    SAMPLES_PER_CLASS = 20

    TARGET_SIZE_MB = 1024
    if len(sys.argv) > 2:
        TARGET_SIZE_MB = float(sys.argv[2])

    NUM_TILES = 768
    if len(sys.argv) > 3:
        NUM_TILES = int(sys.argv[3])

    num_random_copies = calculate_num_copies_per_sample(NUM_TILES, TARGET_SIZE_MB, NUM_CLASSES, SAMPLES_PER_CLASS)
    print(f"\nnum_random_copies: {num_random_copies}")

    if SYMMETRY == 6:
        gen6 = Generator6(imageset, num_tiles=NUM_TILES, target_halfside=5., unit_side=.05)
        files = generate_and_save(gen6, NUM_CLASSES, SAMPLES_PER_CLASS, num_random_copies, prefix="datasets/hex")
        # 768 with .05 is great
        # 364 (x, y) need to be bigger unit_side ≈ .1 ?
    else:
        gen5 = Generator5(imageset, num_tiles=500, target_halfside=5., unit_side=.1)
        files = generate_and_save(gen5, NUM_CLASSES, SAMPLES_PER_CLASS, num_random_copies, prefix="datasets/pen")

    for f in files:
        npz_stats(f)

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