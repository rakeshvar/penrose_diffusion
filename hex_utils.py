import math

def cube_to_xy(q, r, side):
    """Convert cube coordinates to pixel position"""
    x = side * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
    y = side * (3 / 2 * r)
    return x, y
    
def hex_centers_xy(hexes, side):
    """Convert list of hex cube coordinates to pixel positions"""
    xy_data = []

    for hex_data in hexes:
        if len(hex_data) == 4:
            q, r, s, color = hex_data
        else:
            q, r, s = hex_data
            color = 0  # Default color
        
        x, y = cube_to_xy(q, r, side)
        xy_data.append((x, y, q, r, s, color))
    
    return xy_data

def hex_corner_xy(center_x, center_y, i, side):
    """Get corner position of hexagon"""
    angle = math.pi / 3 * i - math.pi / 6
    x = center_x + side * math.cos(angle)
    y = center_y + side * math.sin(angle)
    return x, y

def hex_corners_xy(center_x, center_y, side):
    return [hex_corner_xy(center_x, center_y, i, side) for i in range(6)]

