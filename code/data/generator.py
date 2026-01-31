import numpy as np
from abc import ABC

from code.pen.pregen import get_pen_mother_tiles
from code.hex.pregen import get_hex_mother_tiles
from code.utils.geom import inscribed_square_halfside

class Generator(ABC):
    area_with_unit_side:float = 1.0
    rot_range:float = np.pi

    def __init__(self, imageset, num_tiles, target_halfside, unit_side):
        """
        Build a grid covering square region ([-C, C] × [-C, C]). C = tothalfside
        """

        self.canvas = self._get_mother_tiles(target_halfside, unit_side)  # type: ignore
        self.canvas_xy = np.array([(h.x, h.y) for h in self.canvas], dtype=float)
        self.colors = np.array([h.color for h in self.canvas])
        self.angles = np.array([h.angle for h in self.canvas])
        self.sides = np.array([h.side for h in self.canvas])

        self.imageset = imageset
        self.halfside = inscribed_square_halfside(self.canvas)
        self.unit_side = unit_side
        self.num_tiles = num_tiles

        print(f"  UnitSide: {self.unit_side}")
        print(f"  CanvasHalfSide: {self.halfside:.2f} (vs. {target_halfside})")
        print(f"  Density: {self.density:.3f}")
        print(f"  Sampling Size: {self.num_tiles}")

        self.imagesetiter = iter(self.imageset)

    @property
    def area_of_one_polygon(self):
        return self.area_with_unit_side * self.unit_side ** 2

    @property
    def density(self):
        return 1./self.area_of_one_polygon

    def get_sample(self, class_id=None, inclassid=None, rotate_mask=True):
        if class_id is None:
            sample = next(self.imagesetiter)
        else:
            sample = self.imageset.get_particular_sample(class_id, inclassid)

        H, W = sample.mask.shape

        scaling = np.sqrt(self.num_tiles / (sample.on * self.density))
        c2hw = lambda x: x / scaling
        hw2c = lambda u: u * scaling
        eqsqhfsd = c2hw(np.sqrt(self.area_of_one_polygon)) / 2.0  # Equivalent square half side


        # Rotate Canvas
        θ = np.random.uniform(-self.rot_range, self.rot_range)
        cosθ, sinθ = np.cos(θ), np.sin(θ)
        rot_mat = np.array([[cosθ, sinθ], [-sinθ, cosθ]])  # important minus goes here
        xy_rot = self.canvas_xy @ rot_mat

        # Translate Canvas
        x0 = np.random.uniform(-self.halfside, self.halfside - hw2c(H))
        y0 = np.random.uniform(-self.halfside, self.halfside - hw2c(W))
        new_xy = xy_rot - np.array([x0, y0])

        # Rotate Mask
        if rotate_mask:
            θmask = np.random.uniform(-self.rot_range/6, self.rot_range/6)
            cosθ, sinθ = np.cos(θmask), np.sin(θmask)
            rot_mask_mat = np.array([[cosθ, -sinθ], [sinθ, cosθ]])

        coverage = np.zeros(self.canvas_xy.shape[0], dtype=int)
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
        xyac = np.zeros((self.num_tiles, 4), dtype=float)
        taken = 0
        take_now = 5
        offset = np.array([hw2c(H) / 2., hw2c(W) / 2.])

        for val in (4, 3, 2, 1):
            if taken >= self.num_tiles:
                break
            take_now = val
            idxs = sets_idx[val]
            take = idxs[:self.num_tiles - taken]
            if len(take) > 0:
                xyac[taken:taken + len(take), :2] = new_xy[take] - offset
                xyac[taken:taken + len(take), 2] = self.angles[take] + θ
                xyac[taken:taken + len(take), 3] = self.colors[take]
                taken += len(take)

        name = f"{sample.classname}-{sample.inclassid:02d}"
        # diagnostics printout
        if take_now < 2 or taken < self.num_tiles:
            sets_idx = [np.where(coverage == val)[0] for val in range(5)]  # 0..4
            print(f"{sample.classid:02d} {name:20s} ({H:3d}, {W:3d}) {sample.on/(H*W):.0%}"
              f"\t±{self.halfside:.1f}/{scaling:.3f} = ±{self.halfside/scaling:.0f} {self.unit_side}->{2*eqsqhfsd:.1f}"
              f"\tmapped_to: ({x0:+.2f}, {y0:+.2f}) to ({x0+hw2c(H):+.2f}, {y0+hw2c(W):+.2f}) rot={θ:+.2f}({θ*180/np.pi:+.0f}°)"
              f"\tsets: ({len(sets_idx[4]):3d}, {len(sets_idx[3]):3d}, {len(sets_idx[2]):3d}, {len(sets_idx[1]):3d}) ⇒ {taken:3d} {take_now}")

        # return the actual canvas objects in the same order as original code
        return {'xyac':xyac, 'label':sample.classid, 'name':name}


class Generator6(Generator):
    area_with_unit_side = 3. * np.sqrt(3.) / 2.  # Area of a hexagon with side 1
    rot_range = np.pi
    symmetry = 6

    def _get_mother_tiles(self, tothalfside, unit_side):
        return get_hex_mother_tiles(tothalfside, unit_side)


from code.pen.base import psi, psi2
class Generator5(Generator):
    area_with_unit_side = np.sin(np.pi/5) * psi2 + np.sin(2*np.pi/5) * psi # Weighted average of areas of rhombuses with side 1
    rot_range = np.pi
    symmetry = 5

    def _get_mother_tiles(self, tothalfside, unit_side):
        return get_pen_mother_tiles(tothalfside, unit_side)