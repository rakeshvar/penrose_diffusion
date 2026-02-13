
from pathlib import Path
import numpy as np
from tqdm import tqdm
from code.data.generator import Generator5, Generator6
from code.data.imageset import ImageSet
from code.polygons.hex.svg import save_svg as hex_save_svg
from code.polygons.pen.svg import save_svg as pen_save_svg
from code.polygons.hex.xya import HexGrid
from code.polygons.pen.xya import PenGrid

out_folder = Path("library/tiles/")

folder = "library/MPEG7/gifs"
imageset = ImageSet(folder)
#-------
# Scalings
#-------

num_classes = 70
samples_per_class = 20
num_tiles = 1024
target_halfside = 5.
hex_unit_side = .05
pen_unit_side = .1

hex_density = 1/(Generator6.area_with_unit_side * hex_unit_side ** 2)
pen_density = 1/(Generator5.area_with_unit_side * pen_unit_side ** 2)

print(f"For {num_tiles} hexagons with unit side {hex_unit_side}"
      f"\n\tdensity is {hex_density:.2f}"
      f"\n\tscale H and W")
imageset.print_scaled_hw(num_tiles, hex_density)

print(f"For {num_tiles} pentagons with unit side {pen_unit_side}"
      f"\n\tdensity is {pen_density:.2f}"
      f"\n\tscale H and W")
imageset.print_scaled_hw(num_tiles, pen_density)

input("Press Enter to generate samples ...")

#-------
# Save
#-------

for symmetry in [6, 5]:
    if symmetry == 6:
        GEN = Generator6 
        side = hex_unit_side 
        save_svg = hex_save_svg 
        sym = "hex"
    else:
        GEN = Generator5
        side = pen_unit_side
        save_svg = pen_save_svg
        sym = "pen"

    gen = GEN(imageset, num_tiles=num_tiles, target_halfside=5., unit_side=side)
    save_svg(gen.canvas, out_folder / f"canvas_{sym}.svg")


    with tqdm(total=num_classes * samples_per_class) as pbar:
        for c_id in range(num_classes):
            for s_id in range(samples_per_class):

                sample = gen.get_sample(c_id, s_id, rotate_mask=False)
                xya = sample['xya']
                colors = sample['colors']
                xyac = np.concatenate([xya, colors[:, None]], axis=1)
                grid = HexGrid(xyac, gen.unit_side)
                hex_save_svg(grid, out_folder / f"classes_hex/{sample['name']}.svg", print_ok=False)
                
                pbar.set_description(f"Saved {sample['name']:20s}")
                pbar.update(1)
