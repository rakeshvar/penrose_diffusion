import numpy as np
from PIL import Image
from pathlib import Path

from pen_base import PenGrid
from hex_base import HexGrid
from pen_pregen import get_pen_mother_tiles
from hex_pregen import get_hex_mother_tiles
from utils import inscribed_square_halfside, zealous_crop

from hex_svg import save_svg as hex_save_svg
from pen_svg import save_svg as pen_save_svg

class ImageSet:
    def __init__(self, folder):
        self.folder = folder
        self.images = []
        self.number_in_class = []
        self.class_name_to_id = dict()
        self.class_id_to_name = dict()

        self.class_ids = []
        num_classes = 0

        self.num_on = []

        for f in sorted(Path(folder).glob("*.gif")):
            img = Image.open(f)
            arr = np.array(img, dtype=np.uint8)

            if False and arr[0,0] + arr[0,-1] + arr[-1,0] + arr[-1,-1] != 0:
                print(f"{f.name}\t{arr.size:7d}{arr.shape}", end="\t")
                unique, counts = np.unique(arr, return_counts=True)
                for val, count in zip(unique, counts):
                    print(f" {val:4d}: {count:7d} ({count/arr.size*100:.2f}%)", end="\t")
                print("CHECK" if np.sum(arr==0) + np.sum(arr==1)  + np.sum(arr==255) != arr.size else "", end="\t")
                print(f"Corner pixels: {arr[0,0]} {arr[0,-1]} {arr[-1,0]} {arr[-1,-1]}")

            arr[arr > 0] = 1                   # Some images have values 255 for ON
            arr = zealous_crop(arr, margin=5)
            self.images.append(arr)
            self.num_on.append(np.sum(arr))

            # Zealous clip the image to remove the sides and keep only the central on part

            class_name, num = f.stem.split("-")
            if class_name not in self.class_name_to_id:
                self.class_name_to_id[class_name] = num_classes
                self.class_id_to_name[num_classes] = class_name
                num_classes += 1

            self.number_in_class.append(int(num))
            self.class_ids.append(self.class_name_to_id[class_name])

        self.num_on = np.array(self.num_on)
        self.num_classes = num_classes
        self.number_in_class = np.array(self.number_in_class)
        self.class_ids = np.array(self.class_ids)

        # print(f"Found {len(self.images)} images")
        # print(f"  Num Classes: {self.num_classes}")
        # for class_num in range(self.num_classes):
        #     print(f"    Class {class_num} {self.class_id_to_name[class_num]}: {np.sum(self.class_ids == class_num)}")
        # input("Press Enter to continue...")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        id = self.class_ids[idx]
        name = self.class_id_to_name[id]
        number_in_class = self.number_in_class[idx]
        return self.images[idx], id, name, self.num_on[idx], number_in_class

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_random_sample()

    def get_random_sample(self):
        idx = np.random.randint(0, len(self))
        return self[idx]

class Generator:
    def __init__(self, imageset, symmetry, sample_size, tothalfside, unit_side):
        """
        Build a grid covering square region ([-C, C] × [-C, C]). C = tothalfside
        """
        if symmetry == 5:
            self.canvas = get_pen_mother_tiles(tothalfside, unit_side)
            pen_save_svg(self.canvas, "pen_canvas.svg")
        elif symmetry == 6:
            self.canvas = get_hex_mother_tiles(tothalfside, unit_side)
            hex_save_svg(self.canvas, "hex_canvas.svg")
        else:
            raise ValueError(f"Invalid symmetry: {symmetry}")

        self.imageset = imageset
        self.symmetry = symmetry
        self.halfside = inscribed_square_halfside(self.canvas)
        self.unit_side = unit_side
        self.sample_size = sample_size

        """* Take the sub-square region ([-C, C]^2).
        Count how many hex centers fall within it (D*4C**2).
        The number of hexes per unit area = D
        """
        count = 0
        for h in self.canvas:
            if abs(h.x) <= self.halfside and abs(h.y) <= self.halfside:
                count += 1
        self.density = count / (4 * self.halfside**2)
        count1 = 0
        for h in self.canvas:
            if abs(h.x) <= .5 and abs(h.y) <= .5:
                count1 += 1


        print(f"  UnitSide: {self.unit_side}")
        print(f"  CanvasHalfSide: {self.halfside:.2f} (vs. {tothalfside})")
        print(f"  Density: {self.density:.3f} (vs. {count1})")
        print(f"  Sampling Size: {self.sample_size}")

        self.imagesetiter = iter(self.imageset)

    def sample(self):
        mask, id, name, M, number_in_class = next(self.imagesetiter)
        H, W = mask.shape

        # Scale the mask so that, when placed on the grid, roughly sample_size units lie inside it.
        scaling = np.sqrt(self.sample_size / (M * self.density))

        """ Coverage classification
        * For each unit center:
            * Convert its ((x,y)) to mask coordinates ((u,v)) using the scaled mask mapping.
            * Approximate vertices with four corners ((u1,v1), (u2,v2), (u3,v3), (u4,v4)).
            * Count how many are “on” → assign **coverage class 0–4** (inside→outside gradient).
            * Classify the point into **class k = {0..4}** depending on this count.
                * 4 → deep inside
                * 3 → near inside edge
                * 2 → straddling
                * 1 → just outside
                * 0 → outside
        """
        def maskuv(uu, vv):
            if uu < 0 or vv < 0 or uu >= H or vv >= W:
                return 0
            else:
                return mask[uu, vv]
        def scale2hw(x):
            return x/scaling
        def scale2C(u):
            return u*scaling

        x0 = np.random.uniform(-self.halfside, self.halfside-scale2C(H))
        y0 = np.random.uniform(-self.halfside, self.halfside-scale2C(W))
        scaled_unitside = scale2hw(self.unit_side)
        sets = [[], [], [], [], []]
        for i, h in enumerate(self.canvas):
            u = scale2hw(h.x-x0)
            v = scale2hw(h.y-y0)
            u1, v1 = round(u-scaled_unitside), round(v-scaled_unitside)
            u2, v2 = round(u+scaled_unitside), round(v-scaled_unitside)
            k = maskuv(u1, v1) + maskuv(u2, v1) + maskuv(u1, v2) + maskuv(u2, v2)
            sets[k].append(h)

        """Point selection
            * Gather all centers and sort by class (4 → most inside).
            * Collect units until total count = sample size
        """
        ret = []
        for k in (4, 3, 2, 1):
            if len(ret) < self.sample_size:
                ret.extend(sets[k][:self.sample_size - len(ret)])

        if True:
            print(f"{name:20s}{id:02d} ({H:3d}, {W:3d}) {M/(H*W):.0%}"
                  f"\t±{self.halfside:.1f}/{scaling:.3f} = ±{self.halfside/scaling:.0f}"
                  f"\tmapped_to: ({x0:+.2f}->, {y0:+.2f}) to ({x0+scale2C(H):+.2f}, {y0+scale2C(W):+.2f})"
                  f"\tsets: ({len(sets[4]):3d}, {len(sets[3]):3d}, {len(sets[2]):3d}, {len(sets[1]):3d}) ⇒ {len(ret):3d}")

        return ret, f"{name}-{number_in_class:02d}"

if __name__ == "__main__":
    folder = "data/MPEG7"
    imageset = ImageSet(folder)

    generator6 = Generator(imageset, symmetry=6, sample_size=500, tothalfside=5., unit_side=.05)
    for i in range(len(imageset)):
        sample, name = generator6.sample()
        sample = HexGrid.from_list(sample)
        hex_save_svg(sample, f"data/svgs_hex/{name}.svg")

    generator5 = Generator(imageset, symmetry=5, sample_size=500, tothalfside=5., unit_side=.1)
    for i in range(len(imageset)):
        sample, name = generator5.sample()
        sample = PenGrid(sample, from_rhombuses=True)
        pen_save_svg(sample, f"data/svgs_pen/{name}.svg")
