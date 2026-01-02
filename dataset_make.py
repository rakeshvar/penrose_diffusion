import numpy as np
from tqdm import tqdm

from Generator import Generator6
from ImageSet import ImageSet


def calculate_num_copies(sample_size, target_mb=1024., num_classes=70, samples_per_class=20):
    """
    Calculates how many random augmentations per image are needed  to hit the target file size.
    """
    bytes_per_val = 2  # float16
    cols = 4.5           # 4 features + 1 class label column
    total_base_images = num_classes * samples_per_class
    
    bytes_per_sample = sample_size * cols * bytes_per_val
    target_bytes = target_mb * (1024**2)
    
    num_copies = int(target_bytes / (total_base_images * bytes_per_sample))
    return num_copies

def generate_and_save(generator, num_classes, samples_per_class, num_copies, prefix):
    sample_size = generator.sample_size
    total_samples = num_classes * samples_per_class * num_copies
    
    print(f"Targeting {total_samples} total samples.")
    print(f"Allocating X: ({total_samples}, {sample_size}, 4) [float16]")
    print(f"Allocating Y: ({total_samples},) [uint8]")
    
    # Pre-allocate separate arrays
    all_x = np.zeros((total_samples, sample_size, 4), dtype=np.float16)
    all_y = np.zeros((total_samples,), dtype=np.uint8)
    
    idx = 0
    # Nested loops as requested: copies -> samples -> classes
    for _ in tqdm(range(num_copies)):
        for s_id in range(samples_per_class):
            for c_id in range(num_classes):
                sample_data = generator.get_sample(c_id, s_id)
                
                all_x[idx] = sample_data['x'].astype(np.float16)
                all_y[idx] = np.uint8(sample_data['y'])
                
                idx += 1

    def save_npz(filename, x, y):
        print(f"Saving file: {filename} ")
        np.savez(filename, x=x, y=y)
        print(f"Saved!")

    half_num_copies = num_copies // 2
    save_npz(f"{prefix}_s{sample_size}_c{half_num_copies}.npz", all_x[:half_num_copies], all_y[:half_num_copies])
    save_npz(f"{prefix}_s{sample_size}_c{num_copies}.npz", all_x, all_y)

if __name__ == "__main__":
    import sys

    folder = "data/MPEG7"
    imageset = ImageSet(folder)    
    NUM_CLASSES = imageset.num_classes
    SAMPLES_PER_CLASS = 20

    TARGET_SIZE_MB = 1024
    if len(sys.argv) > 1:
        TARGET_SIZE_MB = float(sys.argv[1])
    SAMPLE_SIZE = 768
    if len(sys.argv) > 2:
        SAMPLE_SIZE = int(sys.argv[2])

    num_random_copies = calculate_num_copies(SAMPLE_SIZE, TARGET_SIZE_MB, NUM_CLASSES, SAMPLES_PER_CLASS)
    print(f"Calculated num_random_copies: {num_random_copies}")

    COMPRESSION_FACTOR = 2
    num_random_copies = int(num_random_copies * COMPRESSION_FACTOR)
    print(f"Final num_random_copies: {num_random_copies}")

    gen6 = Generator6(imageset, sample_size=SAMPLE_SIZE, target_halfside=5., unit_side=.05)
    generate_and_save(gen6, NUM_CLASSES, SAMPLES_PER_CLASS, num_random_copies, prefix="Data_hex")
