from pathlib import Path
from code.polygons.pen.shapes import triangle_tiling, star_tiling, circle_tiling
from code.polygons.pen.svg import save_svg


#----------------------------------------
# Configurations
#----------------------------------------

configs = [
    {'draw-arcs': False},

    {'tile-opacity': 1.0,
     'Aarc-colour': '#f04040',
     'Carc-colour': '#4040f0'
    },

    {
     'Aarc-colour': '#ff5e25',
     'Carc-colour': 'none',
     'Stile-colour': '#009000',
     'Ltile-colour': '#90f030',
    }
]

out = Path("library/tiles/mother")
out.mkdir(parents=True, exist_ok=True)

for j in range(5):
    triangle_tiling.inflate(1)
    circle_tiling.inflate(1)
    star_tiling.inflate(1)

    for i, config in enumerate(configs):
        save_svg(triangle_tiling, out / f'triangle_{i}_{j}.svg', add_config=config)
        save_svg(star_tiling, out / f'star_{i}_{j}.svg', add_config=config)
        save_svg(circle_tiling, out / f'circle_{i}_{j}.svg', add_config=config)
