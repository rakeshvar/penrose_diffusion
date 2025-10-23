import copy
from utils import svg_path

def save_svg(hexgird, filename, target_side=20):
    # Color palette
    config = {
    "colors": {
        0: "#D8D388",
        1: "#4ee055",
        2: "#A34444",
        },
    "margin": 0.05,
    "stroke-colour": "#ffffff",
    "base-stroke-width": 1,
    }

    # Scale to target side
    orig_side = hexgird.side
    if not (.5 < orig_side/target_side < 1.5):
        scale_factor = target_side / orig_side
        hexgird = copy.deepcopy(hexgird)
        hexgird.scale(scale_factor)

    # Determine viewbox size
    xmin = ymin = float('inf')
    xmax = ymax = float('-inf')
    for t in hexgird:
        for x, y in t.vertices:
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)

    wd, ht = xmax-xmin, ymax-ymin
    m = config['margin']
    viewbox = f'{xmin-wd*m} {ymin-ht*m} {wd+2*wd*m} {ht+2*ht*m}'

    # Build SVG
    svg = ['<?xml version="1.0" encoding="utf-8"?>',
        f'<svg viewBox="{viewbox}" preserveAspectRatio="xMidYMid meet" version="1.1" baseProfile="full" xmlns="http://www.w3.org/2000/svg" style="background-color: black;">',
        f'<g style="stroke:{config["stroke-colour"]}; stroke-width: {config["base-stroke-width"]}; stroke-linejoin: round;">'
    ]

    # Draw hexagons
    for h in hexgird:
        fill_color = config["colors"].get(h.color, '#94a3b8')

        # Draw hexagon
        path = svg_path(h)
        svg.append(f'<path fill="{fill_color}" d="{path}" />')

    svg.append('</g>\n</svg>')

    svg = '\n'.join(svg)
    with open(filename, 'w') as f:
        f.write(svg)

    print(f"Saved SVG to {filename}.")
    return svg

