import math
import cmath
import copy
from collections import Counter
from utils import cross

def deg(u):
    return cmath.phase(u) * 180 / math.pi

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
        signa = 1 if cross(MB, self.C - self.A) > 0 else -1
        side *= signa
        return M, angle, side
    

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
        

    def get_rhombus_tiles(self):
        """
        Get the rhombus representations of all tiles in the tiling.
        """
        tiles = copy.deepcopy(self)
        tiles.remove_mirror_images()
        return [Rhombus(t) for t in tiles.elements]


class Rhombus:
    """
    M: complex number representing middle of base
    angle: angle of MB relative to horizontal (in radians)
    side: length of side AB
    """
    def __init__(self, tri:Triangle):
        m, t, s = tri.reparametrize()
        self.center = m
        self.tilt = t  
        self.side = s

        if isinstance(tri, Thin):
            self.topangle = math.pi / 5  
        else:
            self.topangle = 3 * math.pi / 5

        if False:   # Consistency checks (disabled for performance)        
            half_base = self.side * math.sin(self.topangle / 2)
            orig_half_base = abs(tri.C - tri.A) / 2
            if abs(half_base - orig_half_base) > TOL:
                raise ValueError("Inconsistent side length and top angle in Rhombus initialization.")
            
            height = self.side * math.cos(self.topangle / 2)
            orig_height = abs(tri.B - tri.center)
            if abs(height - orig_height) > TOL:
                raise ValueError("Inconsistent side length and top angle in Rhombus initialization.")

            uMB = cmath.exp(1j * self.tilt)          # Direction from M to B
            uAC = 1j * uMB                           # Perpendicular direction (base direction)
            orig_uAC = (tri.C - tri.A) / abs(tri.C - tri.A)
            dot_product = (uAC.real * orig_uAC.real + uAC.imag * orig_uAC.imag)
            if abs(dot_product) - 1 > TOL:
                raise ValueError(f"Inconsistent base direction in Rhombus initialization. dot_product = {dot_product}")
    
    def scale(self, factor):
        self.center *= factor
        self.side *= factor

    def triangle(self):
        half_base = self.side * math.sin(self.topangle / 2) 
        height = self.side * math.cos(self.topangle / 2)
        uMB = cmath.exp(1j * self.tilt)          # Direction from M to B
        uAC = -1j * uMB                           # Perpendicular direction (base direction)
        
        B = self.center - height * uMB
        A = self.center + half_base * uAC
        C = self.center - half_base * uAC

        if self.topangle == math.pi / 5:
            return Thin(A, B, C)
        elif self.topangle == 3 * math.pi / 5:
            return Fatt(A, B, C)
        else: 
            raise ValueError("Invalid top angle for Rhombus to Triangle conversion.") 

    @property
    def vertices(self):
        return self.triangle().vertices
    
    @property
    def side_length(self):
        return abs(self.side)

def distances_to_the_closest_neighbor(elements):
    """
    For each tile, compute the distance to the closest neighboring tile.
    """
    dists = Counter()
    for i, t1 in enumerate(elements):
        min_dist = float('inf')
        c1 = t1.center
        for j, t2 in enumerate(elements):
            if i != j:
                c2 = t2.center
                dist = abs(c2 - c1)
                if dist < min_dist:
                    min_dist = dist
        dists[TOL * round(min_dist/TOL)] += 1

    print("Distances to closest neighbor:")
    for dist in sorted(dists):
        print(f"  {dist:6.3f}: {dists[dist]}")        
    
    return dists

