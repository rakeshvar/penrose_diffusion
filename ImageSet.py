import numpy as np
from PIL import Image
from pathlib import Path
from collections import namedtuple

from utils import inscribed_square_halfside, zealous_crop

Sample = namedtuple("Sample", ["mask", "classid", "on", "classname", "inclassid"])

class ImageSet:
    def __init__(self, folder):
        self.folder = folder
        self.class_name_to_id = dict()
        self.class_id_to_name = dict()
        num_classes = 0
        self.samples = []

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

            class_name, inclassid = f.stem.split("-")
            if class_name not in self.class_name_to_id:
                self.class_name_to_id[class_name] = num_classes
                self.class_id_to_name[num_classes] = class_name
                num_classes += 1

            class_id = self.class_name_to_id[class_name]
            sample = Sample(mask=arr, classid=class_id, on=np.sum(arr), classname=class_name, inclassid=int(inclassid))
            self.samples.append(sample)

        self.num_classes = num_classes

        print(f"Found {len(self.samples)} images")
        print(f"  Num Classes: {self.num_classes}")
        for class_num in range(self.num_classes):
            print(f"    Class {class_num} {self.class_id_to_name[class_num]}: {len([s for s in self.samples if s.classid == class_num])} samples")

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

