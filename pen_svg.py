import math
import copy
from utils import cross, svg_path


def svg_arc(U, V, W):
    """
    SVG "d" path for the circular arc between sides UV and UW, joined at half-distance along these sides.
    """
    start = (U + V) / 2
    r = abs((V - U) / 2)    # arc radius
    end = (U + W) / 2

    # Ensure we draw the arc for the angular component < 180 deg
    US, UE = start - U, end - U
    if cross(US, UE) > 0:
        start, end = end, start

    return 'M {} {} A {} {} 0 0 0 {} {}'.format(start.real, start.imag, r, r, end.real, end.imag)


def svg_arcs(rhombus):
    """
    SVG "d" path for the two circular arcs about vertices A and C.
    """
    A, B, C, D = rhombus.vertices
    arc_a = svg_arc(A, B, D)
    arc_c = svg_arc(C, B, D)
    return arc_a, arc_c


def save_svg(tiling, filename, additional_config={}, target_side=20):
    # Default configuration
    config = {
            'stroke-colour': '#ffffff',
            'base-stroke-width': 1,
            'margin': .05,
            'Stile-colour': '#0080f0',
            'Ltile-colour': '#0035f3',
            'Aarc-colour': '#ff8000',
            'Carc-colour': '#f0c030',
            'draw-arcs': True,
            'tile-opacity': 0.5,
            }
    config.update(additional_config)

    if hasattr(tiling, 'remove_mirror_images'):
        tiling = copy.deepcopy(tiling)
        tiling.remove_mirror_images()
        tiling = tiling.elements

    def tile_colour(e):
        if e.__class__.__name__ == 'Fatt' or (hasattr(e, 'topangle') and abs(e.topangle - (3*math.pi/5)) < 1e-6):
            return config['Ltile-colour']
        else:
            return config['Stile-colour']

    # Scale to target side
    orig_side = tiling[0].side_length
    if not (.5 < orig_side/target_side < 1.5):
        scale_factor = target_side / orig_side
        tiling = copy.deepcopy(tiling)
        for t in tiling:
            t.scale(scale_factor)

    # Determine viewbox size
    xmin = ymin = float('inf')
    xmax = ymax = float('-inf')
    for t in tiling:
        for v in t.vertices:
            xmin = min(xmin, v.real)
            xmax = max(xmax, v.real)
            ymin = min(ymin, v.imag)
            ymax = max(ymax, v.imag)

    wd, ht = xmax-xmin, ymax-ymin
    m = config['margin']
    viewbox = f'{xmin-wd*m} {ymin-ht*m} {wd+2*wd*m} {ht+2*ht*m}'

    # Build SVG
    svg = ['<?xml version="1.0" encoding="utf-8"?>',
        f'<svg viewBox="{viewbox}" preserveAspectRatio="xMidYMid meet" version="1.1" baseProfile="full" xmlns="http://www.w3.org/2000/svg" style="background-color: black;">',
        f'<g style="stroke:{config["stroke-colour"]}; stroke-width: {config["base-stroke-width"]}; stroke-linejoin: round; opacity: {config["tile-opacity"]};">'
    ]

    for t in tiling:
        dpath = svg_path(t)
        svg.append(f'<path fill="{tile_colour(t)}" d="{dpath}"/>')

        if config['draw-arcs']:
            arc1_d, arc2_d = svg_arcs(t)
            svg.append(f'<path fill="none" stroke="{config["Aarc-colour"]}" d="{arc1_d}"/>')
            svg.append(f'<path fill="none" stroke="{config["Carc-colour"]}" d="{arc2_d}"/>')

    svg.append('</g>\n</svg>')
    svg = '\n'.join(svg)

    with open(filename, 'w') as fo:
        fo.write(svg)

    print(f'Wrote SVG to {filename}')
    return svg