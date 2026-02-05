import numpy as np
from code.data.generator import Generator6
from code.data.imageset import ImageSet
from tqdm import tqdm


def get_qrs_sample(gen6, class_id=None, inclassid=None, rotate_mask=True):
    try:
        _ = gen6.canvas_qrs
    except AttributeError:
        gen6.canvas_qrs = np.array([(h.q, h.r, h.s) for h in gen6.canvas], dtype=np.int8)

    if class_id is None:
        sample = next(gen6.imagesetiter)
    else:
        sample = gen6.imageset.get_particular_sample(class_id, inclassid)

    H, W = sample.mask.shape

    scaling = np.sqrt(gen6.num_tiles / (sample.on * gen6.density))
    c2hw = lambda x: x / scaling
    hw2c = lambda u: u * scaling
    eqsqhfsd = c2hw(np.sqrt(gen6.area_of_one_polygon)) / 2.0  # Equivalent square half side


    # Rotate Canvas
    θ = np.random.uniform(-gen6.rot_range, gen6.rot_range)
    cosθ, sinθ = np.cos(θ), np.sin(θ)
    rot_mat = np.array([[cosθ, sinθ], [-sinθ, cosθ]])  # important minus goes here
    xy_rot = gen6.canvas_xy @ rot_mat

    # Translate Canvas
    x0 = np.random.uniform(-gen6.halfside, gen6.halfside - hw2c(H))
    y0 = np.random.uniform(-gen6.halfside, gen6.halfside - hw2c(W))
    new_xy = xy_rot - np.array([x0, y0])

    # Rotate Mask
    if rotate_mask:
        θmask = np.random.uniform(-gen6.rot_range/6, gen6.rot_range/6)
        cosθ, sinθ = np.cos(θmask), np.sin(θmask)
        rot_mask_mat = np.array([[cosθ, -sinθ], [sinθ, cosθ]])

    coverage = np.zeros(gen6.canvas_xy.shape[0], dtype=int)
    def update_coverage(uu, vv):
        uuvv = np.stack([uu, vv], axis=1) - np.array([H/2, W/2])
        if rotate_mask:  uuvv = uuvv @ rot_mask_mat # type: ignore
        uuvv = uuvv + np.array([H/2, W/2])
        uu = np.round(uuvv[:, 0]).astype(int)
        vv = np.round(uuvv[:, 1]).astype(int)
        is_in_bounds = (uu >= 0) & (uu < H) & (vv >= 0) & (vv < W)
        coverage[is_in_bounds] += sample.mask[uu[is_in_bounds], vv[is_in_bounds]]

    # half-square corners in float coords
    uv = c2hw(new_xy)
    u, v = uv[:, 0], uv[:, 1]
    update_coverage(u - eqsqhfsd, v - eqsqhfsd)
    update_coverage(u - eqsqhfsd, v + eqsqhfsd)
    update_coverage(u + eqsqhfsd, v - eqsqhfsd)
    update_coverage(u + eqsqhfsd, v + eqsqhfsd)

    sets_idx = {val: np.flatnonzero(coverage == val) for val in (1, 2, 3, 4)}
    qrs = np.zeros((gen6.num_tiles, 3), dtype=np.int8)
    taken = 0
    take_now = 5

    for val in (4, 3, 2, 1):
        if taken >= gen6.num_tiles:
            break
        take_now = val
        idxs = sets_idx[val]
        take = idxs[:gen6.num_tiles - taken]
        if len(take) > 0:
            qrs[taken:taken + len(take)] = gen6.canvas_qrs[take]
            taken += len(take)

    name = f"{sample.classname}-{sample.inclassid:02d}"
    # diagnostics printout
    if take_now < 2 or taken < gen6.num_tiles:
        sets_idx = [np.where(coverage == val)[0] for val in range(5)]  # 0..4
        print(f"{sample.classid:02d} {name:20s} ({H:3d}, {W:3d}) {sample.on/(H*W):.0%}"
            f"\t±{gen6.halfside:.1f}/{scaling:.3f} = ±{gen6.halfside/scaling:.0f} {gen6.unit_side}->{2*eqsqhfsd:.1f}"
            f"\tmapped_to: ({x0:+.2f}, {y0:+.2f}) to ({x0+hw2c(H):+.2f}, {y0+hw2c(W):+.2f}) rot={θ:+.2f}({θ*180/np.pi:+.0f}°)"
            f"\tsets: ({len(sets_idx[4]):3d}, {len(sets_idx[3]):3d}, {len(sets_idx[2]):3d}, {len(sets_idx[1]):3d}) ⇒ {taken:3d} {take_now}")

    # return the actual canvas objects in the same order as original code
    return {'qrs':qrs, 'label':sample.classid, 'name':name}


def generate_qrs_and_save(generator, samples_per_class, num_copies, prefix):
    num_tiles = generator.num_tiles
    num_classes = generator.imageset.num_classes
    total_samples = num_classes * samples_per_class * num_copies
    print(f"\nTotal Samples = {num_classes} classes * {samples_per_class} samples * {num_copies} copies = {total_samples}.")

    qrs = np.zeros((total_samples, num_tiles, 3), dtype=np.int8)
    print(f"qrs: ({total_samples}, {num_tiles}, 3) [{qrs.dtype}]")
    labels = np.zeros((total_samples,), dtype=np.uint8)
    print(f"labels: ({total_samples},) [{labels.dtype}]")

    print("\nFilling data...")
    i = 0
    for _ in tqdm(range(num_copies)):
        for s_id in range(samples_per_class):
            for c_id in range(num_classes):
                sample_data = get_qrs_sample(generator, c_id, s_id, rotate_mask=True)
                qrs_i = sample_data['qrs']
                qrs[i] = qrs_i[:, :3]
                labels[i] = sample_data['label']
                i += 1

    def save_npz(filename, _qrs, _labels):
        print(f"Saving file: {filename} ")
        np.savez(filename,
                 qrs=_qrs,
                 labels=_labels,
                 symmetry=generator.symmetry,
                 side=generator.unit_side,
                 num_tiles=generator.num_tiles,
                 num_classes=num_classes,
                 class_lookup=generator.imageset.class_name_to_id |
                              generator.imageset.class_id_to_name )
        print(f"Saved!")

    ffull = f"{prefix}_t{num_tiles:03d}_c{num_copies:02d}_u{round(100*generator.unit_side):02d}.npz"
    save_npz(ffull, qrs, labels)

    return ffull


def main():
    folder = "library/MPEG7/gifs"
    imageset = ImageSet(folder)
    SAMPLES_PER_CLASS = 20

    NUM_TILES = int(sys.argv[1])
    NUM_COPIES = int(sys.argv[2])

    if len(sys.argv) > 3:
        UNIT_SIDE = float(sys.argv[3])
    else:                               # Aiming for a std of 1. for x, y
        UNIT_SIDE = .18 * (96/NUM_TILES)**.5
    UNIT_SIDE = round(UNIT_SIDE, 2)

    print(f"NUM_TILES: {NUM_TILES} \nNUM_COPIES: {NUM_COPIES} \nUNIT_SIDE: {UNIT_SIDE}")

    gen6 = Generator6(imageset, num_tiles=NUM_TILES, target_halfside=5., unit_side=UNIT_SIDE)
    file = generate_qrs_and_save(gen6, SAMPLES_PER_CLASS, NUM_COPIES, prefix="datasets/hexqrs")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} num_tiles num_copies [unit_side]")
        exit(0)
    main()