def get_color(q, r, s):
    """
    A coloring scheme such that no 'dark' hexagons touch each other.
    Returns 1 for 'dark' color, 0 for 'light' color.
    """
    return int((max(abs(q), abs(r), abs(s)) + min(abs(q), abs(r), abs(s))) % 3 == 0)

def get_hex_ring(degree):
    """
    Generate hexes at exactly distance 'degree' from origin with 2-coloring.
    Returns list of tuples (q, r, s, color) in circular order.
    """
    if degree == 0:
        return [(0, 0, 0, get_color(0, 0, 0))]
    
    hexes = []
    q, r, s = 0, -degree, degree
    
    directions = [
        (1, 0, -1),   # East
        (0, 1, -1),   # Southeast
        (-1, 1, 0),   # Southwest
        (-1, 0, 1),   # West
        (0, -1, 1),   # Northwest
        (1, -1, 0),   # Northeast
    ]
    
    for direction_idx in range(6):
        dq, dr, ds = directions[direction_idx]
        for step in range(degree):
            hexes.append((q, r, s, get_color(q, r, s)))
            q += dq
            r += dr
            s += ds
    
    return hexes

def get_hex_rings_degree(max_degree):
    """
    Generate hexes in rings from degree 0 to max_degree.
    Returns list of tuples (q, r, s, color).
    """
    all_hexes = []
    for degree in range(max_degree ):
        ring_hexes = get_hex_ring(degree)
        all_hexes.extend(ring_hexes)
    return all_hexes

def get_hex_rings_count(hex_count):
    """
    Generate hexes in rings until reaching at least hex_count hexagons.
    Returns list of tuples (q, r, s, color).
    """
    all_hexes = []
    degree = 0
    while len(all_hexes) < hex_count:
        ring_hexes = get_hex_ring(degree)
        all_hexes.extend(ring_hexes)
        degree += 1
    return all_hexes