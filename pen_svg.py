import cmath


def reim(v):
    return v.real, v.imag


def cross(u, v):
    return u.real*v.imag - u.imag*v.real


def path(rhombus):
    """ SVG "d" path for the rhombus. """
    AB, BC = rhombus.B - rhombus.A, rhombus.C - rhombus.B 
    return 'm{},{} l{},{} l{},{} l{},{}z'.format(*(reim(rhombus.A) + reim(AB) + reim(BC) + reim(-AB)))


def svg_arc(self, U, V, W):
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
    D = rhombus.A - rhombus.B + rhombus.C
    arc_a = rhombus.svg_arc(rhombus.A, rhombus.B, D)      
    arc_c = rhombus.svg_arc(rhombus.C, rhombus.B, D)      
    return arc_a, arc_c



def save_svg(tiling, additional_config, filename):
    # Default configuration
    config = {
            'stroke-colour': '#ffffff',
            'base-stroke-width': 1,
            'margin': .05,
            'Stile-colour': '#0080f0',
            'Ltile-colour': '#0035f3',
            'Aarc-colour': '#ff8000',
            'Carc-colour': '#f0c030',
            'draw-arcs': False,
            }
    config.update(additional_config)
    
    tiling.remove_mirror_images() # Just in case

    def tile_colour(e):
        if isinstance(e, Fatt):
            col = config['Ltile-colour']
        else:
            col = config['Stile-colour']

        if hasattr(col, '__call__'): # Can be a function or a string
            return col(e)
        else:   
            return col

    # Determine viewbox size
    xmin = ymin = float('inf')
    xmax = ymax = float('-inf')
    for t in tiling.elements:
        for v in [t.A, t.B, t.C]:
            xmin = min(xmin, v.real)
            xmax = max(xmax, v.real)
            ymin = min(ymin, v.imag)
            ymax = max(ymax, v.imag)

    wd, ht = xmax-xmin, ymax-ymin
    m = config['margin']
    viewbox = f'{xmin-wd*m} {ymin-ht*m} {wd+2*wd*m} {ht+2*ht*m}'

    # Build SVG
    svg = ['<?xml version="1.0" encoding="utf-8"?>',
        f'<svg viewBox="{viewbox}" preserveAspectRatio="xMidYMid meet" version="1.1" baseProfile="full" xmlns="http://www.w3.org/2000/svg">',
        f'<g style="stroke:{config['stroke-colour']}; stroke-width: {config['base-stroke-width']}; stroke-linejoin: round;">'
    ]
    
    for t in tiling.elements:
        dpath = t.path(rhombus=True)
        svg.append(f'<path fill="{tile_colour(t)}" d="{dpath}"/>')

        if config['draw-arcs']:
            arc1_d, arc2_d = t.arcs(rhombus=True)
            svg.append(f'<path fill="none" stroke="{config['Aarc-colour']}" d="{arc1_d}"/>')
            svg.append(f'<path fill="none" stroke="{config['Carc-colour']}" d="{arc2_d}"/>')

    svg.append('</g>\n</svg>')
    svg = '\n'.join(svg)

    with open(filename, 'w') as fo:
        fo.write(svg)

    print(f'Wrote SVG to {filename}')
    return svg