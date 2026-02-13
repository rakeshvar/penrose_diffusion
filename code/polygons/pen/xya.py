import math
import cmath
import copy
from collections import namedtuple

from ..polygon import Polygon, Grid
from .triangle import Triangle, Thin, Fatt

class Rhombus(Polygon):
    """
    M: complex number representing middle of base
    angle: angle of MB relative to horizontal (in radians)
    side: length of side AB
    """
    def __init__(self, tri):
        if isinstance(tri, Triangle):
            m, t, s = tri.reparametrize()
            self.center = m
            self.tilt = t
            self.side = s
            self.type = type(tri)
            self.color = self.type == Thin

        else:
            self.center = tri.center
            self.tilt = tri.tilt
            self.side = tri.side
            self.color = tri.color
            self.type = Thin if tri.color else Fatt


    @property
    def topangle(self):
            if self.type == Thin:
                return math.pi / 5
            else:
                return 3 * math.pi / 5

    def triangle(self):
        half_base = self.side * math.sin(self.topangle / 2)
        height = self.side * math.cos(self.topangle / 2)
        uMB = cmath.exp(1j * self.tilt)          # Direction from M to B
        uAC = -1j * uMB                           # Perpendicular direction (base direction)

        B = self.center - height * uMB
        A = self.center + half_base * uAC
        C = self.center - half_base * uAC
        return self.type(A, B, C)

    def scale(self, factor):
        self.center *= factor
        self.side *= factor

    def rotate(self, alpha):
        self.center *= cmath.exp(1j * alpha)
        self.tilt += alpha

    def translate(self, dx, dy):
        self.center += dx + 1j * dy

    @property
    def vertices(self):
        return self.triangle().vertices

    @property
    def side_length(self):
        return abs(self.side)

    @property
    def angle(self):
        return self.tilt

    @property
    def x(self):
        return self.center.real

    @property
    def y(self):
        return self.center.imag


class PenGrid(Grid):
    symmetry = 5

    def __init__(self, triangles, from_rhombuses=False, from_np=False, side=None):
        if from_rhombuses:
            self.rhombuses = triangles
        elif from_np:
            assert side is not None, "Side must be specified."
            Rhom = namedtuple('Rhom', ['center', 'tilt', 'color', 'side'])
            self.rhombuses = [Rhombus(Rhom(complex(t[0], t[1]), t[2], t[3], side)) for t in triangles]
        else:
            triangles = copy.deepcopy(triangles)
            triangles.remove_mirror_images()
            self.rhombuses = [Rhombus(t) for t in triangles]

    def rotate(self, alpha):
        for h in self.rhombuses:
            h.rotate(alpha)

    def translate(self, dx, dy):
        for h in self.rhombuses:
            h.translate(dx, dy)

    def scale(self, factor):
        for h in self.rhombuses:
            h.scale(factor)

    def __iter__(self):
        return iter(self.rhombuses)

    def __len__(self):
        return len(self.rhombuses)

    @property
    def side(self):
        return abs(self.rhombuses[0].side)
