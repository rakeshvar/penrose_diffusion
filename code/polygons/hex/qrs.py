import math
from code.utils.geom import inscribed_square_halfside
from .index import cube_to_spiral, spiral_to_cube, directions

def get_color(q, r, s):
    """
    A coloring scheme such that no 'dark' hexagons touch each other.
    Returns 1 for 'dark' color, 0 for 'light' color.
    """
    return int((max(abs(q), abs(r), abs(s)) + min(abs(q), abs(r), abs(s))) % 3 == 0)

#--------------------------------------------------
#     (q, r, s) represented hexagons
#--------------------------------------------------
class HexQRS:
    def __init__(self, q, r, s, side=1, angle=0):
        assert q + r + s == 0, "Cube coordinates must satisfy q + r + s = 0"
        self.q = q
        self.r = r
        self.s = s
        self.color = get_color(q, r, s)
        self.side = side
        self.angle = angle
        self.index = cube_to_spiral(q, r, s)

    @property
    def center(self):
        x = self.side * (math.sqrt(3) * self.q + math.sqrt(3) / 2 * self.r)
        y = self.side * (3 / 2 * self.r)
        return x, y

    def __vertexi_xy(self, i):
        """Get corner position of hexagon"""
        anglei = math.pi / 3 * i - math.pi / 6
        angle = anglei + self.angle
        center_x, center_y = self.center
        x = center_x + self.side * math.cos(angle)
        y = center_y + self.side * math.sin(angle)
        return x, y

    @property
    def vertices(self):
        return [self.__vertexi_xy(i) for i in range(6)]

    def scale(self, factor):
        self.side *= factor

    def __add__(self, direction):
        dq, dr, ds = direction
        return HexQRS(self.q + dq, self.r + dr, self.s + ds, self.side, self.angle)

#--------------------------------------------------
#     Hex ring generators
#--------------------------------------------------
def get_hex_ring(degree):
    """
    Generate hexes at exactly distance 'degree' from origin with 2-coloring.
    Returns list of tuples (q, r, s, color) in circular order.
    """
    hexes = [HexQRS(0, -degree, degree)]

    for direction in directions:
        for step in range(degree):
            hexes.append(hexes[-1] + direction)

    if degree:       # You end up at the beginning of the ring
        hexes.pop()

    return hexes

#--------------------------------------------------
#     Hex grid generators in (q, r, s)  
#--------------------------------------------------
class HexQRSGrid:
    @classmethod
    def from_degree(cls, max_degree):
        all_hexes = []
        for degree in range(max_degree):
            all_hexes.extend(get_hex_ring(degree))
        return cls(all_hexes)

    @classmethod
    def from_count(cls, target_count):
        all_hexes = []
        degree = 0
        while len(all_hexes) < target_count:
            all_hexes.extend(get_hex_ring(degree))
            degree += 1
        return cls(all_hexes)

    @classmethod
    def from_halfside(cls, target_hexside, target_halfside):
        """
        Generate hexagons that cover a square of half size 'total_halfside'.
        With hexagons with side 'hex_side'.
        """
        degree = 0
        all_hexes = get_hex_ring(degree)
        unscaled_halfside = target_halfside * all_hexes[0].side / target_hexside

        while inscribed_square_halfside(all_hexes) < unscaled_halfside:
            degree += 1
            all_hexes.extend(get_hex_ring(degree))
        return cls(all_hexes)

    def __init__(self, hexes):
        self.hexes = hexes

    def __iter__(self):
        return iter(self.hexes)

    def __len__(self):
        return len(self.hexes)