import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from typing import Optional

from pen_base import PenGrid
from hex_base import HexGrid
from pen_pregen import get_pen_mother_tiles
from hex_pregen import get_hex_mother_tiles
from utils import inscribed_square_halfside

from hex_svg import save_svg as hex_save_svg
from pen_svg import save_svg as pen_save_svg

from ImageSet import ImageSet

class Generator(ABC):
    unit_area:float = 1.0
    rot_range:float = np.pi

    def __init__(self, imageset, sample_size, target_halfside, unit_side):
        """
        Build a grid covering square region ([-C, C] × [-C, C]). C = tothalfside
        """

        self.canvas = self._get_mother_tiles(target_halfside, unit_side)
        self.canvas_xy = np.array([(h.x, h.y) for h in self.canvas], dtype=float)
        self.colors = np.array([h.color for h in self.canvas])
        self.angles = np.array([h.angle for h in self.canvas])
        self.sides = np.array([h.side for h in self.canvas]) 

        self.imageset = imageset
        self.halfside = inscribed_square_halfside(self.canvas)
        self.unit_side = unit_side
        self.sample_size = sample_size

        print(f"  UnitSide: {self.unit_side}")
        print(f"  CanvasHalfSide: {self.halfside:.2f} (vs. {target_halfside})")
        print(f"  Density: {self.density:.3f}")
        print(f"  Sampling Size: {self.sample_size}")

        self.imagesetiter = iter(self.imageset)

    @abstractmethod
    def _get_mother_tiles(self, tothalfside, unit_side):
        raise NotImplementedError

    @property
    def area_of_one_unit(self):
        return self.unit_area * self.unit_side ** 2
        
    @property
    def density(self):
        return 1./self.area_of_one_unit
    
    def get_sample(self):
        sample = next(self.imagesetiter)
        H, W = sample.mask.shape

        scaling = np.sqrt(self.sample_size / (sample.on * self.density))
        c2hw = lambda x: x / scaling
        hw2c = lambda u: u * scaling
        eqsqhfsd = c2hw(np.sqrt(self.area_of_one_unit)) / 2.0  # Equivalent square half side

        coverage = np.zeros(self.canvas_xy.shape[0], dtype=int)
        def update_coverage(uu, vv):
            is_in_bounds = (uu >= 0) & (uu < H) & (vv >= 0) & (vv < W)
            coverage[is_in_bounds] += sample.mask[uu[is_in_bounds], vv[is_in_bounds]]            

        # Rotate
        theta = np.random.uniform(-self.rot_range, self.rot_range)
        rot_mat = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]  # minus sin goes here because we are post multiplying @ rot_mat
        ])
        xy_rot = self.canvas_xy @ rot_mat

        # Translate
        x0 = np.random.uniform(-self.halfside, self.halfside - hw2c(H))
        y0 = np.random.uniform(-self.halfside, self.halfside - hw2c(W))
        new_xy = xy_rot - np.array([x0, y0])


        uv = c2hw(new_xy)
        u = uv[:, 0]
        v = uv[:, 1]

        # compute corner indices
        u1 = np.round(u - eqsqhfsd).astype(int)
        v1 = np.round(v - eqsqhfsd).astype(int)
        u2 = np.round(u + eqsqhfsd).astype(int)
        v2 = np.round(v + eqsqhfsd).astype(int)
        update_coverage(u1, v1)
        update_coverage(u1, v2)
        update_coverage(u2, v1)
        update_coverage(u2, v2)
    
        sets_idx = {val: np.flatnonzero(coverage == val) for val in (1, 2, 3, 4)}
        ret = np.zeros((self.sample_size, 5), dtype=float)
        taken = 0
        take_now = 5

        for val in (4, 3, 2, 1):
            if taken >= self.sample_size:
                break
            take_now = val
            idxs = sets_idx[val]
            take = idxs[:self.sample_size - taken]
            if len(take) > 0:
                ret[taken:taken + len(take), :2] = new_xy[take]
                ret[taken:taken + len(take), 2] = self.colors[take]
                ret[taken:taken + len(take), 3] = self.angles[take] + theta
                ret[taken:taken + len(take), 4] = self.sides[take]
                taken += len(take)

        name = f"{sample.classname}-{sample.inclassid:02d}"
        # diagnostics printout
        if take_now < 2 or taken < self.sample_size:
            sets_idx = [np.where(coverage == val)[0] for val in range(5)]  # 0..4
            print(f"{sample.classid:02d} {name:20s} ({H:3d}, {W:3d}) {sample.on/(H*W):.0%}"
              f"\t±{self.halfside:.1f}/{scaling:.3f} = ±{self.halfside/scaling:.0f} {self.unit_side}->{2*eqsqhfsd:.1f}"
              f"\tmapped_to: ({x0:+.2f}, {y0:+.2f}) to ({x0+hw2c(H):+.2f}, {y0+hw2c(W):+.2f}) rot={theta:+.2f}({theta*180/np.pi:+.0f}°)"
              f"\tsets: ({len(sets_idx[4]):3d}, {len(sets_idx[3]):3d}, {len(sets_idx[2]):3d}, {len(sets_idx[1]):3d}) ⇒ {taken:3d} {take_now}")

        # return the actual canvas objects in the same order as original code
        return ret, name


class Generator6(Generator):
    unit_area = 3. * np.sqrt(3.) / 2.  
    rot_range = np.pi/6

    def _get_mother_tiles(self, tothalfside, unit_side):
        canvas = get_hex_mother_tiles(tothalfside, unit_side)
        hex_save_svg(canvas, "hex_canvas.svg")
        return canvas
    
    
from pen_base import psi, psi2
class Generator5(Generator):
    unit_area = np.sin(np.pi/5) * psi2 + np.sin(2*np.pi/5) * psi
    rot_range = np.pi/2

    def _get_mother_tiles(self, tothalfside, unit_side):
        canvas = get_pen_mother_tiles(tothalfside, unit_side)
        pen_save_svg(canvas, "pen_canvas.svg")
        return canvas
    

if __name__ == "__main__":
    from tqdm import tqdm
    folder = "data/MPEG7"
    imageset = ImageSet(folder)

    generator6 = Generator6(imageset, sample_size=500, target_halfside=5., unit_side=.05)
    for i in tqdm(range(len(imageset))):
        sample_matrix, name = generator6.get_sample()
        grid = HexGrid(sample_matrix)
        hex_save_svg(grid, f"data/svgs_hex/{i:04d}_{name}.svg")

    generator5 = Generator5(imageset, sample_size=500, target_halfside=5., unit_side=.1)
    for i in tqdm(range(len(imageset))):
        sample_matrix, name = generator5.get_sample()
        grid = PenGrid(sample_matrix, from_np=True)
        pen_save_svg(grid, f"data/svgs_pen/{i:04d}_{name}.svg")