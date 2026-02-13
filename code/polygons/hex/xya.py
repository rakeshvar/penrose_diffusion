import math
import numpy as np
from collections import namedtuple

from ..polygon import Grid, Polygon
from .qrs import HexQRS, HexQRSGrid

#------------------------------------------
# Hexagons as (x, y, angle)
#------------------------------------------
class Hexagon(Polygon):
    symmetry = 6

    def __init__(self, hexagon):
        self.x, self.y = hexagon.center
        self.color = hexagon.color
        self.angle = hexagon.angle
        self.side = hexagon.side
        try:
            self.q, self.r, self.s = hexagon.q, hexagon.r, hexagon.s
            self.index = hexagon.index
        except AttributeError:
            self.q = self.r = self.s = self.index = None

    def rotate(self, alpha):
        self.x, self.y = (
            self.x * math.cos(alpha) - self.y * math.sin(alpha),
            self.x * math.sin(alpha) + self.y * math.cos(alpha)
        )
        self.angle += alpha

    def translate(self, dx, dy):
        self.x += dx
        self.y += dy

    def scale(self, factor):
        self.x *= factor
        self.y *= factor
        self.side *= factor

    @property
    def center(self):
        return self.x, self.y

    @property
    def vertices(self):
        vertices0 = HexQRS(0, 0, 0, self.side, self.angle).vertices
        return [(vx + self.x, vy + self.y) for vx, vy in vertices0]

    def __str__(self) -> str:
        return f"HexXYA {self.x:7.2f} {self.y:7.2f} {self.angle:7.2f} ({math.degrees(self.angle):+3.0f}) {self.color} {self.side:.1f}"

#------------------------------------------
# HexGrid as list of HexXYA
#------------------------------------------
class HexGrid(Grid):
    symmetry = 6

    def __init__(self, hexagons, side=None):
        if isinstance(hexagons, HexQRSGrid):
            self.hexxyas = [Hexagon(h) for h in hexagons]
        elif isinstance(hexagons, list):
            if isinstance(hexagons[0], HexQRS):
                self.hexxyas = [Hexagon(h) for h in hexagons]
            elif isinstance(hexagons[0], Hexagon):
                self.hexxyas = hexagons
            else:
                raise ValueError(f"Type of list elements not supported: {type(hexagons[0])}")
        elif isinstance(hexagons, np.ndarray):
            assert side is not None, "Side must be specified for numpy array"
            hextuple = namedtuple('hextuple', ['center', 'angle', 'color', 'side'])
            self.hexxyas = [Hexagon(hextuple((h[0], h[1]), h[2], h[3], side)) for h in hexagons]
        else:
            raise ValueError(f"Type of hexagons not supported: {type(hexagons)}")

    def rotate(self, alpha):
        for h in self.hexxyas:
            h.rotate(alpha)

    def translate(self, dx, dy):
        for h in self.hexxyas:
            h.translate(dx, dy)

    def scale(self, factor):
        for h in self.hexxyas:
            h.scale(factor)

    def __iter__(self):
        return iter(self.hexxyas)

    def __len__(self):
        return len(self.hexxyas)

    @property
    def side(self):
        return self.hexxyas[0].side