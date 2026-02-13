import numpy as np
from tqdm import tqdm

def calculate_num_copies_per_sample(num_tiles, target_mb=1024., num_classes=70, samples_per_class=20):
    """
    Calculates how many random augmentations per image are needed  to hit the target file size.
    """
    bytes_per_val = 4    # float32
    columns = 3.25        # xya (float32) + colors (uint8)
    total_base_images = num_classes * samples_per_class

    bytes_per_sample = num_tiles * columns * bytes_per_val
    target_bytes = target_mb * (1024**2)

    num_copies_per_sample = int(target_bytes / (total_base_images * bytes_per_sample))
    return num_copies_per_sample

def generate_and_save(generator, samples_per_class, num_copies, prefix,
                      save_xyac=True, save_labels=True, save_indices=False):
    num_tiles = generator.num_tiles
    num_classes = generator.imageset.num_classes
    total_samples = num_classes * samples_per_class * num_copies
    print(f"\nTotal Samples = {num_classes} classes * {samples_per_class} samples * {num_copies} copies = {total_samples}.")

    if save_xyac:
        print(f"xya: ({total_samples}, {num_tiles}, 3) [float32]")
        xya = np.zeros((total_samples, num_tiles, 3), dtype=np.float32)
        print(f"colors: ({total_samples}, {num_tiles}) [uint8]")
        colors = np.zeros((total_samples, num_tiles), dtype=np.uint8)
    if save_labels:
        print(f"labels: ({total_samples},) [uint8]")
        labels = np.zeros((total_samples,), dtype=np.uint8)
    if save_indices:
        print(f"indices: ({total_samples},) [uint16]")
        indices = np.zeros((total_samples, num_tiles), dtype=np.uint16)

    print("\nFilling data...")
    i = 0
    with tqdm(total=total_samples) as pbar:
      for icopy in range(num_copies):
        for s_id in range(samples_per_class):
            for c_id in range(num_classes):
                sample_data = generator.get_sample(c_id, s_id, rotate_mask=False)

                if save_xyac:
                    xya[i] = sample_data['xya']             # type: ignore
                    colors[i] = sample_data['colors']       # type: ignore           
                if save_labels:
                    labels[i] = sample_data['label']        # type: ignore
                if save_indices:
                    indices[i] = sample_data['indices']     # type: ignore

                i += 1
                pbar.set_description(f"Saved {sample_data['name']:20s}")
                pbar.update(1)

    filename = f"{prefix}_t{num_tiles:03d}_c{num_copies:02d}_u{round(100*generator.unit_side):02d}.npz"
    print(f"Saving file: {filename} ")

    data = {}
    if save_xyac:
        data['xya'] = xya               # type: ignore
        data['colors'] = colors         # type: ignore
    if save_labels:
        data['labels'] = labels         # type: ignore
    if save_indices:
        data['indices'] = indices       # type: ignore
        data['canvas_xyac'] = generator.canvas_xyac
        data['vocab_size'] = indices.max() + 1 # type: ignore
    
    np.savez(filename,
            symmetry=generator.symmetry,
            side=generator.unit_side,
            num_tiles=generator.num_tiles,
            num_classes=num_classes,
            class_lookup=generator.imageset.class_name_to_id |
                        generator.imageset.class_id_to_name,
            **data)
    print(f"Saved!")

    return filename
