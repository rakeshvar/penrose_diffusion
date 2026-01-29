import math
from code.utils_geom import cross, svg_path, vertexy
from code.pen.base import PenGrid, TriangleGrid

def svg_arc(U, V, W, ndigits):
    """
    SVG "d" path for the circular arc between sides UV and UW, joined at half-distance along these sides.
    """
    start = (U + V) / 2
    r = abs((V - U) / 2)    # arc radius
    end = (U + W) / 2

    # Ensure we draw the arc for the angular component < 180 deg
    US, UE = start - U, end - U
    if cross(US, UE) < 0:
        start, end = end, start

    sr, si = vertexy(start)
    er, ei = vertexy(end)
    return f'M {si:.{ndigits}f} {sr:.{ndigits}f} A {r:.{ndigits}f} {r:.{ndigits}f} 0 0 0 {ei:.{ndigits}f} {er:.{ndigits}f}'


def svg_arcs(rhombus, ndigits):
    """
    SVG "d" path for the two circular arcs about vertices A and C.
    """
    A, B, C, D = rhombus.vertices
    arc_a = svg_arc(A, B, D, ndigits)
    arc_c = svg_arc(C, B, D, ndigits)
    return arc_a, arc_c


def save_svg(pengrid: PenGrid|TriangleGrid, filename, ndigits=3, print_ok=True, add_config=None):
    config = {
    'stroke-colour': '#ffffff',
    'margin': .05,
    'Stile-colour': '#0080f0',
    'Ltile-colour': '#0035f3',
    'Aarc-colour': '#ff8000',
    'Carc-colour': '#f0c030',
    'draw-arcs': True,
    'tile-opacity': 0.5,
    }
    if add_config is not None:
        config.update(add_config)

    if isinstance(pengrid, TriangleGrid):
        pengrid = PenGrid(pengrid)

    def tile_colour(e):
        if e.__class__.__name__ == 'Fatt' or (hasattr(e, 'topangle') and abs(e.topangle - (3*math.pi/5)) < 1e-6):
            return config['Ltile-colour']
        elif e.__class__.__name__ == 'Thin' or (hasattr(e, 'topangle') and abs(e.topangle - (math.pi/5)) < 1e-6):
            return config['Stile-colour']
        else:
            raise ValueError(f"Unknown Penrose Tile {e} of type {type(e)}")


    # Determine viewbox size
    xmin = ymin = float('inf')
    xmax = ymax = float('-inf')
    for t in pengrid:
        for v in t.vertices:
            y, x = vertexy(v)
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)

    wd, ht = xmax-xmin, ymax-ymin
    stats =  f"# {len(pengrid)} hexagons \n"
    stats += f"# x {xmin:.2f} to {xmax:.2f} ({wd:.2f})\n"
    stats += f"# y {ymin:.2f} to {ymax:.2f} ({ht:.2f})\n"

    m = config['margin']
    xmin -= m*wd
    ymin -= m*ht
    wd += 2*m*wd
    ht += 2*m*ht
    viewbox = f'{xmin:.3f} {ymin:.3f} {wd:.3f} {ht:.3f}'

    # Build SVG
    svg = ['<?xml version="1.0" encoding="utf-8"?>']
    svg.append( '<svg preserveAspectRatio="xMidYMid meet" version="1.1" baseProfile="full" xmlns="http://www.w3.org/2000/svg"')
    svg.append(f'     width="{1000*wd/ht:.3f}" height="{1000:.3f}"')
    svg.append(f'     viewBox="{viewbox}">')
    svg.append(f'<rect x="{xmin:.3f}" y="{ymin:.3f}" width="{wd:.3f}" height="{ht:.3f}" fill="black"/>\n')
    svg.append(f'<g style="stroke:{config["stroke-colour"]}; stroke-width: {pengrid.side/20:.4f};')
    svg.append( '   stroke-linejoin: round; vector-effect: non-scaling-stroke;">\n')
    svg.append(stats)

    # Draw pentagons (rhombuses)
    for t in pengrid:
        dpath = svg_path(t, ndigits=ndigits)
        svg.append(f'<path fill="{tile_colour(t)}" d="{dpath}"/>')

        if config['draw-arcs']:
            arc1_d, arc2_d = svg_arcs(t, ndigits)
            svg.append(f'<path fill="none" stroke="{config["Aarc-colour"]}" d="{arc1_d}"/>')
            svg.append(f'<path fill="none" stroke="{config["Carc-colour"]}" d="{arc2_d}"/>')

    svg.append('</g>\n</svg>')
    svg = '\n'.join(svg)

    with open(filename, 'w') as fo:
        fo.write(svg)

    if print_ok:
        print("Saved svg file  :", filename)

    return svg
