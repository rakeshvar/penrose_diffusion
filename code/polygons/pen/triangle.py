import math
import cmath
from code.utils.geom import cross

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