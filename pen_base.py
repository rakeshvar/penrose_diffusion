import math
import cmath

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

    def conjugate(self):
        """
        The reflection of this triangle about the x-axis. 
        """
        return self.__class__(self.A.conjugate(), self.B.conjugate(), self.C.conjugate())
    
    def reparametrize(self):
        """
        Reparametrize the triangle to (center, angle, side length) form.
        """
        M = self.centre()
        MB = self.B - M
        AC = self.C - self.A
        angle = cmath.phase(MB) - cmath.phase(AC)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        side = abs(self.B - self.A)
        
        return M, angle, side
    
    @classmethod
    def from_reparametrized(cls, M, angle, side, phi):
        """
        Create a triangle from (M, angle, side) parameters
        
        Args:
            M: complex number representing middle of base
            angle: angle of MB relative to horizontal (in radians)
            side: length of side AB
            
        Returns:
            Triangle: New triangle object
        """

        half_base = side * math.sin(phi / 2) 
        height = side * math.cos(phi / 2)
        uMB = cmath.exp(1j * angle)          # Direction from M to B
        uAC = 1j * uMB                       # Perpendicular direction (base direction)
        
        B = M + height * uMB
        A = M - half_base * uAC
        C = M + half_base * uAC
        
        return cls(A, B, C)
    

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
    
    @classmethod
    def from_reparametrized(cls, M, angle, side):
        return super().from_reparametrized(M, angle, side, 3*math.pi/5)


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

    @classmethod
    def from_reparametrized(cls, M, angle, side):
        return super().from_reparametrized(M, angle, side, math.pi/5)


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

    def remove_mirror_images(self):
        """
        Keep only one of each pair of tiles that are mirror images of each other.
        """
        selements = sorted(self.elements, key=lambda e: (e.centre().real, e.centre().imag))
        self.elements = [selements[0]]

        for i in range(len(selements)-1):
            if abs(selements[i+1].centre() - selements[i].centre()) > TOL:
                self.elements.append(selements[i+1])
