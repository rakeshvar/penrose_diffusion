import numpy as np

class HexGenerator:
    def __init__(self, max_count, side):
        self.max_count = max_count
        self.side = side
        self.hex_data = get_hex_rings_count(max_count)
        self.xy_data = hex_centers_xy(self.hex_data, side)

    def select_points(self, mask, N):
        # Generate random translation and rotation
        alpha = np.random.uniform(-np.pi/6, np.pi/6)
        a, b = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)

        # Gather xy positions of hex
        points = np.array([[x/self.side, y/self.side] for x, y, _, _, _, _ in self.xy_data])

        # Step 1–2: transform
        c, s = np.cos(alpha), np.sin(alpha)
        R = np.array([[c, -s], [s, c]])
        pts = self.side * (points @ R.T) + np.array([a, b])

        # Step 3: count how many inside mask
        H, W = mask.shape
        xs = ((pts[:,0] + 0.5) * W).astype(int)
        ys = ((0.5 - pts[:,1]) * H).astype(int)
        inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        M = mask[ys[inside], xs[inside]].sum()

        # Step 4: scale correction
        s1 = s0 * np.sqrt(N / (M + 1e-9))
        pts = s1 * (points @ R.T) + np.array([a, b])

        # Step 5: trim using distance_transform
        from scipy.ndimage import distance_transform_edt, map_coordinates
        dist = distance_transform_edt(mask)
        xs = np.clip((pts[:,0] + 0.5) * W, 0, W-1)
        ys = np.clip((0.5 - pts[:,1]) * H, 0, H-1)
        samples = map_coordinates(dist, [ys, xs], order=1, mode='constant', cval=0)
        topN_idx = np.argsort(-samples)[:N]

        return pts[topN_idx]
