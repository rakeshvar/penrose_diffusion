import numpy as np
from PIL import Image
from pathlib import Path

from pen_base import PenGrid
from hex_base import HexGrid
from pen_pregen import get_pen_mother_tiles
from hex_pregen import get_hex_mother_tiles
from utils import inscribed_square_halfside

from hex_svg import save_svg as hex_save_svg

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

            # if arr[0,0] + arr[0,-1] + arr[-1,0] + arr[-1,-1] != 0:
            #     print(f"{f.name}\t{arr.size:7d}{arr.shape}", end="\t")
            #     unique, counts = np.unique(arr, return_counts=True)
            #     for val, count in zip(unique, counts):
            #         print(f" {val:4d}: {count:7d} ({count/arr.size*100:.2f}%)", end="\t")
            #     print("CHECK" if np.sum(arr==0) + np.sum(arr==1)  + np.sum(arr==255) != arr.size else "", end="\t")
            #     print(f"Corner pixels: {arr[0,0]} {arr[0,-1]} {arr[-1,0]} {arr[-1,-1]}")

            self.images.append(arr)
            arr[arr > 0] = 1
            self.num_on.append(np.sum(arr))

            class_name, num = f.stem.split("-")
            if class_name not in self.class_name_to_id:
                self.class_name_to_id[class_name] = num_classes
                self.class_id_to_name[num_classes] = class_name
                num_classes += 1

            self.number_in_class.append(int(num))
            self.class_ids.append(self.class_name_to_id[class_name])

            if len(self.images) % 10000 == 0:
                print(f"Processed {len(self.images)} images")
                break

        self.num_on = np.array(self.num_on)
        self.num_classes = num_classes
        self.number_in_class = np.array(self.number_in_class)
        self.class_ids = np.array(self.class_ids)

        # print(f"Found {len(self.images)} images")
        # print(f"  Num Classes: {self.num_classes}")
        # print(f"  Samples: {len(self.images)}")
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
    def __init__(self, imageset, symmetry, N, C, S):
        """
        Build a grid covering square region ([-C, C] \times [-C, C]).
        """
        if symmetry == 5:
            self.canvas = get_pen_mother_tiles(C, S)
        elif symmetry == 6:
            self.canvas = get_hex_mother_tiles(C, S)
            hex_save_svg(self.canvas, "hex_canvas.svg")
        else:
            raise ValueError(f"Invalid symmetry: {symmetry}")

        self.imageset = imageset
        self.symmetry = symmetry
        self.C = inscribed_square_halfside(self.canvas)
        self.S = S
        self.N = N

        """* Take the sub-square region ([-C, C]^2).
        Count how many hex centers fall within it (D*4C**2).
        The number of hexes per unit area = D
        """
        count = 0
        for h in self.canvas:
            if abs(h.x) <= self.C and abs(h.y) <= self.C:
                count += 1
        self.D = count / (4 * self.C**2)
        count1 = 0
        for h in self.canvas:
            if abs(h.x) <= .5 and abs(h.y) <= .5:
                count1 += 1


        print(f"  S: {self.S}")
        print(f"  C: {self.C:.2f}")
        print(f"  D: {self.D:.3f} (vs. {count1})")
        print(f"  N: {self.N}")

        self.imagesetiter = iter(self.imageset)

    def sample(self):
        """1. **Pick a mask**
        Select one of your masks (🐈, 🐋, 🐟, etc.) with its associated (s).
        """
        mask, id, name, M, number_in_class = next(self.imagesetiter)
        H, W = mask.shape

        """* Fix your desired number of output tiles (N).
        * For each binary mask (of size (H \times W)):

        * Count its “on” pixels → (M).
        * Compute the scale factor:
            [
            s = \\sqrt{ \frac{N}{M , D} }
            ]
        * (The scale (s) enlarges or shrinks the mask so that, when projected on the grid, roughly (N) hex centers lie inside it.)
        """
        s = np.sqrt(self.N / (M * self.D))

        """2. **Scale the mask**
        Conceptually, apply scale (s) about the mask center when mapping coordinates (no need to resample image).
        """

        """3. **Choose transform parameters**
        * Pick random rotation (a) (angle in radians, e.g. ([-π, π))).
        * Rotate by (a_1) about that shifted origin.
        """
        # a = np.random.uniform(-np.pi, np.pi)
        # for h in self.canvas:
        #     h.rotate(a)

        """4. **Apply transform to canvas**
                * Pick random translation ((x_1, y_1)).
            Each uniformly in range:
            [
            x_1, y_1 \\sim \text{Unif}(-C, C - sW)
            ]
        * Translate the hex grid by ((x_1, y_1)).
        """
        # x0 = -self.C/2 #np.random.uniform(-self.C, self.C - s * W)
        # y0 = -self.C/2 #np.random.uniform(-self.C, self.C - s * H)
        # for h in self.canvas:
        #     h.translate(x0, y0)

        """5. **Coverage classification**
        * For each hex center:

            * Convert its ((x,y)) to mask coordinates ((u,v)) using the scaled mask mapping.
            * Look at four neighboring pixels around ((u,v)).
            * Count how many are “on” → assign **coverage class 0–4** (inside→outside gradient).
        
            2. **Normalize coordinate mapping**

            * Define the world bounds (e.g. ([-3, 3]) × ([-3, 3])).
            * Map each center to mask indices:
                [
                u_i = (x_i - x_{\\min}) / (x_{\\max}-x_{\\min}) * (W - 1)
                ]
                [
                v_i = (y_{\\max} - y_i) / (y_{\\max}-y_{\\min}) * (H - 1)
                ]

            3. **Compute coverage classes (fixed mask scale = 1)**
            For each point:

            * Take the four neighboring pixels
                [
                (⌊u⌋, ⌊v⌋), (⌈u⌉, ⌊v⌋), (⌊u⌋, ⌈v⌉), (⌈u⌉, ⌈v⌉)
                ]
            * Count how many of these are “on” in the mask.
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

        x0 = y0 = -self.C/2
        sets = [[], [], [], [], []]
        for i, h in enumerate(self.canvas):
            u = (h.x-x0) / s
            v = (h.y-y0) / s
            u1, v1 = int(np.floor(u)), int(np.floor(v))
            u2, v2 = int(np.ceil(u)), int(np.floor(v))
            k = maskuv(u1, v1) + maskuv(u2, v1) + maskuv(u1, v2) + maskuv(u2, v2)
            # if k > 0:
            #     print(f"{i}) ({h.x:.2f}, {h.y:.2f}) -> ({u:.2f}, {v:.2f}) -> ({u1}, {v1}) + ({u1}, {v2}) + ({u2}, {v1}) + ({u2}, {v2}) = {k}")
            sets[k].append(h)

        """6. **Point selection**

        * Gather all centers and sort by class (4 → most inside).
        * Collect hexes until total count = (N).

            * If you have fewer than (N) in higher classes, include lower classes as needed.
            * If more, randomly subsample from the topmost filled classes.

        """
        ret = []
        for k in (4, 3, 2, 1):
            if len(ret) < self.N:
                ret.extend(sets[k][:self.N - len(ret)])

        if abs(len(sets[4])-self.N)/self.N > .1:
            print(f"Chose a {name}({id}) shape:{(H, W)} on: {M} pixels ({M/(H*W):.0%}) scale = {s:.3f}")
            print(f"\tUp   Scaled shape of grid: ±{self.C:.2f}->±{self.C/s:.2f}")
            print(f"\tDown Scaled shape of mask: {H}->{H*s:.2f} {W}->{W*s:.2f}")
            for k in range(5):
                print(f"{k}: {len(sets[k])}")
            print(f"{len(ret)} selected")

        return ret, f"{name}-{number_in_class:02d}"

if __name__ == "__main__":
    folder = "data/MPEG7"
    imageset = ImageSet(folder)

    generator6 = Generator(imageset, symmetry=6, N=500, C=5., S=.05)
    for i in range(10000):
        sample, name = generator6.sample()
        sample = HexGrid.from_list(sample)
        if len(sample) > 0:
            hex_save_svg(sample, f"data/svgs/{name}.svg")

    # from pen_svg import save_svg
    # generator5 = Generator(imageset, symmetry=5, N=100, C=2., S=.1)
    # for i in range(10):
    #     sample = generator5.sample()
    #     save_svg(sample, f"{i}.svg")
    #     print(sample)
