import math

def get_color(q, r, s):
    """
    A coloring scheme such that no 'dark' hexagons touch each other.
    Returns 1 for 'dark' color, 0 for 'light' color.
    """
    return int((max(abs(q), abs(r), abs(s)) + min(abs(q), abs(r), abs(s))) % 3 == 0)

class Hexagon:
    def __init__(self, q, r, s, side=1, angle=0):
        assert q + r + s == 0, "Cube coordinates must satisfy q + r + s = 0"
        self.q = q
        self.r = r
        self.s = s
        self.color = get_color(q, r, s)
        self.side = side
        self.angle = angle

    @property
    def center(self):
        x = self.side * (math.sqrt(3) * self.q + math.sqrt(3) / 2 * self.r)
        y = self.side * (3 / 2 * self.r)
        return x, y

    def __hex_corner_xy(self, i):
        """Get corner position of hexagon"""
        anglei = math.pi / 3 * i - math.pi / 6
        angle = anglei + self.angle
        center_x, center_y = self.center
        x = center_x + self.side * math.cos(angle)
        y = center_y + self.side * math.sin(angle)
        return x, y

    @property
    def vertices(self):
        return [self.__hex_corner_xy(i) for i in range(6)]

    def scale(self, factor):
        self.side *= factor
    
    def __add__(self, direction):
        dq, dr, ds = direction
        return Hexagon(self.q + dq, self.r + dr, self.s + ds, self.side, self.angle)

directions = [
    (1, 0, -1),   # East
    (0, 1, -1),   # Southeast
    (-1, 1, 0),   # Southwest
    (-1, 0, 1),   # West
    (0, -1, 1),   # Northwest
    (1, -1, 0),   # Northeast
]

def get_hex_ring(degree):
    """
    Generate hexes at exactly distance 'degree' from origin with 2-coloring.
    Returns list of tuples (q, r, s, color) in circular order.
    """
    hexes = [Hexagon(0, -degree, degree)]

    for direction in directions:
        for step in range(degree):
            hexes.append(hexes[-1] + direction)
    
    return hexes

class HexagonGrid:
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

    def __init__(self, hexes):
        self.hexes = hexes

    def __iter__(self):
        return iter(self.hexes)

class HexXYA:
    def __init__(self, hexagon):
        self.x, self.y = hexagon.center
        self.color = hexagon.color
        self.angle = hexagon.angle
        self.side = hexagon.side
    
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
    def vertices(self):
        vertices0 = Hexagon(0, 0, 0, self.side, self.angle).vertices
        return [(vx + self.x, vy + self.y) for vx, vy in vertices0]

class HexGrid:
    def __init__(self, hexagons):
        self.hexxyas = [HexXYA(h) for h in hexagons]
    
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