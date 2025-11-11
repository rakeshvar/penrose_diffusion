import math
import cmath
import copy
from utils import cross

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

    @property
    def D(self):
        """ The fourth vertex of the rhombus formed by this triangle and its mirror image about the base. """
        return self.A - self.B + self.C

    @property
    def vertices(self):
        """ Return the triangle vertices as a tuple. """
        return self.A, self.B, self.C, self.D

    @property
    def center(self):
        """ Center of the base """
        return (self.A + self.C) / 2

    @property
    def side_length(self):
        """ Length of side AB (= BC = CD = DA). """
        return abs(self.B - self.A)

    def rotate(self, theta):
        rot = cmath.exp(1j * theta)
        self.A *= rot
        self.B *= rot
        self.C *= rot

    def rotated(self, theta):
        rot = cmath.exp(1j * theta)
        return self.__class__(self.A * rot, self.B * rot, self.C * rot)

    def scale(self, factor):
        self.A *= factor
        self.B *= factor
        self.C *= factor

    def flip_x(self):
        """
        The reflection of this triangle about the x-axis.
        """
        return self.__class__(self.A.conjugate(), self.B.conjugate(), self.C.conjugate())

    def flip_y(self):
        """
        The reflection of this triangle about the y-axis.
        """
        return self.__class__(complex(-self.A.real, self.A.imag),
                              complex(-self.B.real, self.B.imag),
                              complex(-self.C.real, self.C.imag))

    def reparametrize(self):
        """
        Reparametrize the triangle to (center, angle, side length) form.
        """
        M = self.center
        MB = self.B - M
        angle = cmath.phase(MB)
        side = abs(self.B - self.A)
        if cross(MB, self.C - self.A) < 0:
            angle += math.pi
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        return M, angle, side

    @property
    def side(self):
        return abs(self.B - self.A)


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


class TriangleGrid:
    """ P3 Penrose tiling made of two types of triangles. """
    def __init__(self, initial_tiles):
        self.elements = initial_tiles

    def __iter__(self):
        return iter(self.elements)

    def inflate(self, times=1):
        """ "Inflate" each triangle in the tiling ensemble."""
        for _ in range(times):
            new_elements = []
            for element in self.elements:
                new_elements.extend(element.inflate())
            self.elements = new_elements

    def rotate(self, theta):
        for e in self.elements:
            e.rotate(theta)

    def flip_x(self):
        """ Flip the figure about the x-axis. """
        self.elements = [e.flip_x() for e in self.elements]

    def add_x_flipped(self):
        """ Extend the tiling by reflection about the x-axis. """
        self.elements.extend([e.flip_x() for e in self.elements])

    def flip_y(self):
        self.elements = [e.flip_y() for e in self.elements]

    def add_y_flipped(self):
        self.elements.extend([e.flip_y() for e in self.elements])

    def remove_mirror_images(self):
        """
        Keep only one of each pair of tiles that are mirror images of each other.
        """
        seen_centers = set()
        new_elements = []
        for t in self.elements:
            c = t.center
            c_key = (round(c.real / TOL) , round(c.imag / TOL))  # Use rounded coordinates as key
            if c_key not in seen_centers:
                seen_centers.add(c_key)
                new_elements.append(t)
        self.elements = new_elements

    @property
    def side(self):
        return self.elements[0].side

    def __len__(self):
        return len(self.elements)

class Rhombus:
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
            self.color = self.type == Fatt

        else:
            self.center = tri.center
            self.tilt = tri.tilt
            self.side = tri.side
            self.color = tri.color
            self.type = Fatt if tri.color else Thin


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

from collections import namedtuple

class PenGrid:
    def __init__(self, triangles, from_rhombuses=False, from_np=False):
        if from_rhombuses:
            self.rhombuses = triangles
        elif from_np:
            Rhom = namedtuple('Rhom', ['center', 'color', 'tilt', 'side'])
            self.rhombuses = [Rhombus(Rhom(complex(t[0], t[1]), t[2], t[3], t[4])) for t in triangles]
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
