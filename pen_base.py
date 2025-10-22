import math
import random

TOL = 1.e-5                       # A small tolerance for comparing floats for equality
psi = (math.sqrt(5) - 1) / 2      # psi = 1/phi where phi is the Golden ratio, (sqrt(5)+1)/2 = 0.618033988749895
psi2 = 1 - psi                    # psi**2 = 1 - psi = 0.381966011250105

class Triangle:
    """
    A Robinson triangle (or the rhombus formed by union with the mirror image about the base).
    """
    def __init__(self, A, B, C):
        """ A and C are on the base; B is at the 'top'. """
        self.A, self.B, self.C = A, B, C

    def centre(self):
        """ Center of the base """
        return (self.A + self.C) / 2

    def path(self, rhombus=True):
        """ SVG "d" path for the triangle or rhombus. """
        AB, BC = self.B - self.A, self.C - self.B 
        xy = lambda v: (v.real, v.imag)

        if rhombus:
            return 'm{},{} l{},{} l{},{} l{},{}z'.format(*(xy(self.A) + xy(AB) + xy(BC) + xy(-AB)))
        else:
            return        'm{},{} l{},{} l{},{}z'.format(*(xy(self.A) + xy(AB) + xy(BC)))


    def get_arc_d(self, U, V, W, rhombus):
        """
        SVG "d" path for the circular arc between sides UV and UW, joined at half-distance along these sides. 
        """
        start = (U + V) / 2
        r = abs((V - U) / 2)    # arc radius

        if rhombus:
            end = (U + W) / 2
        else:
            UN = V + W - 2*U            # Direction from U to the opposite vertex of the rhombus (base of triangle)
            end = U + r * UN / abs(UN)  # Ends on the base

        # Ensure we draw the arc for the angular component < 180 deg
        cross = lambda u, v: u.real*v.imag - u.imag*v.real
        US, UE = start - U, end - U
        if cross(US, UE) > 0:
            start, end = end, start
 
        return 'M {} {} A {} {} 0 0 0 {} {}'.format(start.real, start.imag, r, r, end.real, end.imag)

    def arcs(self, rhombus=True):
        """
        SVG "d" path for the two circular arcs about vertices A and C. 
        """
        D = self.A - self.B + self.C
        arc_a = self.get_arc_d(self.A, self.B, D, rhombus)      
        arc_c = self.get_arc_d(self.C, self.B, D, rhombus)      
        return arc_a, arc_c

    def conjugate(self):
        """
        The reflection of this triangle about the x-axis. 
        """
        return self.__class__(self.A.conjugate(), self.B.conjugate(), self.C.conjugate())

class Fatt(Triangle):
    """
    "B_L" Penrose tile in the P3 tiling scheme:
        A "large/fat" Robinson triangle (sides in ratio 1:1:phi).
    """

    def inflate(self):
        """
        Fatt Triangle breaks into three triangles: one Thin and two Fatt.
        """
        D = psi2 * self.A + psi * self.C
        E = psi2 * self.A + psi * self.B

        # Take care to order the vertices here so as to get the right orientation for the resulting triangles.
        return [Fatt(D, E, self.A),
                Thin(E, D, self.B),
                Fatt(self.C, D, self.B)]

class Thin(Triangle):
    """
    "B_S" Penrose tile in the P3 tiling scheme:
        A "small/thin" Robinson triangle (sides in ratio 1:1:psi).
    """

    def inflate(self):
        """
        Thin Triangle breaks into one Thin and one Fatt.
        """
        D = psi * self.A + psi2 * self.B
        return [Thin(D, self.C, self.A),
                Fatt(self.C, D, self.B)]

class PenroseP3:
    """ P3 Penrose tiling. """
    def __init__(self, initial_tiles):
        self.elements = initial_tiles

    def inflate(self, times=1):
        """ "Inflate" each triangle in the tiling ensemble."""
        for _ in range(times):
            new_elements = []
            for element in self.elements:
                new_elements.extend(element.inflate())
            self.elements = new_elements

    def remove_mirror_images(self):
        """
        Keep only one of each pair of tiles that are mirror images of each other.
        """
        selements = sorted(self.elements, key=lambda e: (e.centre().real, e.centre().imag))
        self.elements = [selements[0]]

        for i in range(len(selements)-1):
            if abs(selements[i+1].centre() - selements[i].centre()) > TOL:
                self.elements.append(selements[i+1])

    def add_conjugate_elements(self):
        """ Extend the tiling by reflection about the x-axis. """
        self.elements.extend([e.conjugate() for e in self.elements])

    def rotate(self, theta):
        rot = math.cos(theta) + 1j * math.sin(theta)
        for e in self.elements:
            e.A *= rot
            e.B *= rot
            e.C *= rot

    def flip_y(self):
        """ Flip the figure about the y-axis. """
        for e in self.elements:
            e.A = complex(-e.A.real, e.A.imag)
            e.B = complex(-e.B.real, e.B.imag)
            e.C = complex(-e.C.real, e.C.imag)

    def flip_x(self):
        """ Flip the figure about the x-axis. """
        for e in self.elements:
            e.A = e.A.conjugate()
            e.B = e.B.conjugate()
            e.C = e.C.conjugate()

#------------------------------------------
# SVG generation methods for PenroseP3
#------------------------------------------
def save_svg(tiling, additional_config, scale, filename):
    """
    scale determines the size of the final image.
    add_config updates the default configuration.
    """      

    # Default configuration
    config = {'width': '100%', 'height': '100%',
            'stroke-colour': '#fff',
            'base-stroke-width': 0.05,
            'margin': 1.05,
            'tile-opacity': 0.6,
            'random-tile-colours': False,
            'Stile-colour': '#08f',
            'Ltile-colour': '#0035f3',
            'Aarc-colour': '#f00',
            'Carc-colour': '#00f',
            'draw-tiles': True,
            'draw-arcs': False,
            'reflect-x': True,
            'draw-rhombuses': True,
            'rotate': 0,
            'flip-y': False, 'flip-x': False,
            }
    config.update(additional_config)

    if config['draw-rhombuses']:
        tiling.remove_mirror_images()

    if config['reflect-x']:
        tiling.add_conjugate_elements()
        tiling.remove_mirror_images()

    if config['rotate'] != 0:
        tiling.rotate(config['rotate'])

    if config['flip-y']: # After rotation
        tiling.flip_y()

    if config['flip-x']: # After rotation and flip_y
        tiling.flip_x()

    def get_tile_colour(e):
        if config['random-tile-colours']:
            return '#' + hex(random.randint(0,0xfff))[2:]
        
        else:        
            col = config['Ltile-colour' if isinstance(e, Fatt) else 'Stile-colour']
            if hasattr(col, '__call__'): # Can be a function or a string
                return col(e)
            else:   
                return col

    #------------ SVG generation --------------

    xmin = ymin = -scale * config['margin']
    width =  height = 2*scale * config['margin']
    viewbox ='{} {} {} {}'.format(xmin, ymin, width, height)

    svg = ['<?xml version="1.0" encoding="utf-8"?>',
            '<svg width="{}" height="{}" viewBox="{}"'
            ' preserveAspectRatio="xMidYMid meet" version="1.1"'
            ' baseProfile="full" xmlns="http://www.w3.org/2000/svg">'
            .format(config['width'], config['height'], viewbox)]
    
    # The tiles' stroke widths ideally scales with ngen as psi**ngen * 
    stroke_width = str(int(scale**.5) * config['base-stroke-width'])

    svg.append('<g style="stroke:{}; stroke-width: {}; stroke-linejoin: round;">'
            .format(config['stroke-colour'], stroke_width))
    draw_rhombuses = config['draw-rhombuses']

    for T in tiling.elements:
        if config['draw-tiles']:
            svg.append('<path fill="{}" fill-opacity="{}" d="{}"/>'.format(get_tile_colour(T), config['tile-opacity'], T.path(rhombus=draw_rhombuses)))

        if config['draw-arcs']:
            arc1_d, arc2_d = T.arcs(draw_rhombuses)
            svg.append('<path fill="none" stroke="{}" d="{}"/>'.format(config['Aarc-colour'], arc1_d))
            svg.append('<path fill="none" stroke="{}" d="{}"/>'.format(config['Carc-colour'], arc2_d))

    svg.append('</g>\n</svg>')
    svg = '\n'.join(svg)

    with open(filename, 'w') as fo:
        fo.write(svg)

    print(f'Wrote SVG to {filename}')
    return svg