from pathlib import Path
from tqdm import tqdm
from code.data.generator import Generator5, Generator6
from code.data.imageset import ImageSet
from code.hex.svg import save_svg as hex_save_svg
from code.pen.svg import save_svg as pen_save_svg
from code.hex.base import HexGrid
from code.pen.base import PenGrid

folder = "library/MPEG7/gifs"
imageset = ImageSet(folder)

out_folder = Path("library/tiles/")

#-------
# Hex
#-------
generator6 = Generator6(imageset, num_tiles=500, target_halfside=5., unit_side=.05)
hex_save_svg(generator6.canvas, out_folder / "canvas_hex.svg")
for i in tqdm(range(len(imageset))):
    sample = generator6.get_sample()
    grid = HexGrid(sample['xyac'], generator6.unit_side)
    hex_save_svg(grid, out_folder / f"classes_hex/{sample['name']}.svg", print_ok=False)


#-------
# Pen
#-------
generator5 = Generator5(imageset, num_tiles=500, target_halfside=5., unit_side=.1)
pen_save_svg(generator5.canvas, out_folder / "canvas_pen.svg")
for i in tqdm(range(len(imageset))):
    sample = generator5.get_sample()
    grid = PenGrid(sample['xyac'], from_np=True, side=generator5.unit_side)
    pen_save_svg(grid, out_folder / f"classes_pen/{sample['name']}.svg", print_ok=False)