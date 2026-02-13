import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict, namedtuple
from tqdm import tqdm

from code.utils.geom import zealous_crop

Sample = namedtuple("Sample", ["mask", "classid", "on", "classname", "inclassid"])

class ImageSet:
    def __init__(self, folder):
        self.folder = folder
        self.class_name_to_id = dict()
        self.class_id_to_name = dict()
        num_classes = 0
        self.samples = []

        print("Loading images...")
        for f in tqdm(sorted(Path(folder).glob("*.gif"))):
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

            class_name, inclassid = f.stem.split("-")
            if class_name not in self.class_name_to_id:
                self.class_name_to_id[class_name] = num_classes
                self.class_id_to_name[num_classes] = class_name
                num_classes += 1

            class_id = self.class_name_to_id[class_name]
            sample = Sample(mask=arr, classid=class_id, on=np.sum(arr), classname=class_name, inclassid=int(inclassid))
            self.samples.append(sample)

        self.num_classes = num_classes

        self.samples_lookup = defaultdict(dict)
        for s in self.samples:
            self.samples_lookup[s.classid][s.inclassid] = s

        print(f"Found {len(self.samples)} images")
        print(f"  Num Classes: {self.num_classes}")
        # for class_num in range(self.num_classes):
        #     print(f"    Class {class_num} {self.class_id_to_name[class_num]}: {len([s.inclassid for s in self.samples if s.classid == class_num])} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_random_sample()

    def get_random_sample(self):
        idx = np.random.randint(0, len(self))
        return self[idx]

    def get_particular_sample(self, class_id, inclassid):   # inclassid is indexed from 1 in the data folder
        try:
            return self.samples_lookup[class_id][inclassid+1]
        except KeyError:
            print(f"Could not find sample {class_id}-{inclassid+1}")
            raise KeyError

    def print_scaled_hw(self, num_tiles, density):
        def chw(sample):
            H, W = sample.mask.shape
            scaling = np.sqrt(num_tiles / (sample.on * density))
            return H * scaling, W * scaling
        
        chw = np.array([chw(sample) for sample in self.samples], dtype=float) # type: ignore
        minh, minw = np.min(chw, axis=0) # type: ignore
        maxh, maxw = np.max(chw, axis=0) # type: ignore
        avgh, avgw = np.mean(chw, axis=0) # type: ignore

        print(f"  H: Min: {minh:.2f} Avg: {avgh:.2f} Max: {maxh:.2f} (±{maxh/2:.2f})")
        print(f"  W: Min: {minw:.2f} Avg: {avgw:.2f} Max: {maxw:.2f} (±{maxw/2:.2f})")

        min_canvas_radius = max(maxh/2, maxw/2)
        print(f"  Min Need Canvas Radius: {min_canvas_radius:.2f}")