import copy
from utils import svg_path

def save_svg(hexgrid, filename, ndigits=3):
    # Color palette
    config = {
    "colors": {
        0: "#D8D388",
        1: "#4ee055",
        },
    "margin": 0.05,
    "stroke-colour": "#ffffff",
    }

    # Determine viewbox size
    xmin = ymin = float('inf')
    xmax = ymax = float('-inf')
    for t in hexgrid:
        for y, x in t.vertices:
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)

    wd, ht = xmax-xmin, ymax-ymin
    stats =  f"# {len(hexgrid)} hexagons \n"
    stats += f"# x {xmin:.2f} to {xmax:.2f} ({wd:.2f})\n"
    stats += f"# y {ymin:.2f} to {ymax:.2f} ({ht:.2f})\n"

    m = config['margin']
    xmin -= m*wd
    ymin -= m*ht
    wd += 2*m*wd
    ht += 2*m*ht
    viewbox = f'{xmin:.2f} {ymin:.2f} {wd:.2f} {ht:.2f}'

    # Build SVG
    svg = ['<?xml version="1.0" encoding="utf-8"?>']
    svg.append( '<svg preserveAspectRatio="xMidYMid meet" version="1.1" baseProfile="full" xmlns="http://www.w3.org/2000/svg"')
    svg.append(f'     width="{int(1080*wd/ht)}" height="1080"')
    svg.append(f'     viewBox="{viewbox}">')
    svg.append(f'<rect x="{xmin:.3f}" y="{ymin:.3f}" width="{wd:.3f}" height="{ht:.3f}" fill="black"/>\n')
    svg.append(f'<g style="stroke:{config["stroke-colour"]}; stroke-width: {hexgrid.side/20:.4f};')
    svg.append( '   stroke-linejoin: round; vector-effect: non-scaling-stroke;">\n')
    svg.append(stats)

    # Draw hexagons
    for h in hexgrid:
        fill_color = config["colors"][h.color]

        # Draw hexagon
        path = svg_path(h, ndigits=ndigits)
        svg.append(f'<path fill="{fill_color}" d="{path}" />')

    svg.append('</g>\n</svg>')

    svg = '\n'.join(svg)
    with open(filename, 'w') as f:
        f.write(svg)

    return svg

