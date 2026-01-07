import numpy as np
from tqdm import tqdm

from Generator import Generator6
from ImageSet import ImageSet


def calculate_num_copies_per_sample(num_tokens, target_mb=1024., num_classes=70, samples_per_class=20):
    """
    Calculates how many random augmentations per image are needed  to hit the target file size.
    """
    bytes_per_val = 2    # float16
    cols = 4.5           # 4 features + 1 class label column
    total_base_images = num_classes * samples_per_class
    
    bytes_per_sample = num_tokens * cols * bytes_per_val
    target_bytes = target_mb * (1024**2)
    
    num_copies_per_sample = int(target_bytes / (total_base_images * bytes_per_sample))
    return num_copies_per_sample

def generate_and_save(generator, num_classes, samples_per_class, num_copies, prefix):
    num_tokens = generator.num_tokens
    total_samples = num_classes * samples_per_class * num_copies
    print(f"\nTotal Samples = {num_classes} classes * {samples_per_class} samples * {num_copies} copies = {total_samples}.")

    print(f"xyac: ({total_samples}, {num_tokens}, 4) [float16]")
    print(f"labels: ({total_samples},) [uint8]")    
    xyac = np.zeros((total_samples, num_tokens, 4), dtype=np.float16)
    labels = np.zeros((total_samples,), dtype=np.uint8)
    
    print("\nFilling data...")
    i = 0
    for _ in tqdm(range(num_copies)):
        for s_id in range(samples_per_class):
            for c_id in range(num_classes):
                sample_data = generator.get_sample(c_id, s_id)
                
                xyac[i] = sample_data['xyac'].astype(np.float16)
                labels[i] = np.uint8(sample_data['label'])
                
                i += 1

    def save_npz(filename, _xyac, _labels):
        print(f"Saving file: {filename} ")
        np.savez(filename, 
                 xyac=_xyac, 
                 labels=_labels, 
                 symmetry=generator.symmetry, 
                 side=generator.unit_side,
                 num_tokens=generator.num_tokens)
        print(f"Saved!")

    half_total_samples = total_samples // 2
    save_npz(f"{prefix}_s{num_tokens}_c{num_copies//2}.npz", xyac[:half_total_samples], labels[:half_total_samples])
    save_npz(f"{prefix}_s{num_tokens}_c{num_copies}.npz", xyac, labels)

if __name__ == "__main__":
    import sys

    folder = "data/MPEG7"
    imageset = ImageSet(folder)    
    NUM_CLASSES = imageset.num_classes
    SAMPLES_PER_CLASS = 20

    TARGET_SIZE_MB = 1024
    if len(sys.argv) > 1:
        TARGET_SIZE_MB = float(sys.argv[1])
    NUM_TOKENS = 768
    if len(sys.argv) > 2:
        NUM_TOKENS = int(sys.argv[2])

    num_random_copies = calculate_num_copies_per_sample(NUM_TOKENS, TARGET_SIZE_MB, NUM_CLASSES, SAMPLES_PER_CLASS)
    print(f"\nCalculated num_random_copies: {num_random_copies}")

    COMPRESSION_FACTOR = 2
    num_random_copies = int(num_random_copies * COMPRESSION_FACTOR)
    print(f"Final num_random_copies: {num_random_copies}\n")

    gen6 = Generator6(imageset, num_tokens=NUM_TOKENS, target_halfside=5., unit_side=.05)
    generate_and_save(gen6, NUM_CLASSES, SAMPLES_PER_CLASS, num_random_copies, prefix="Data_hex")
