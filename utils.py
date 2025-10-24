import math
import cmath
from collections import Counter

def deg(rad_or_dir):
    if isinstance(rad_or_dir, complex):
        return cmath.phase(rad_or_dir) * 180 / math.pi
    else:
        return rad_or_dir * 180 / math.pi

def reim(v):
    return v.real, v.imag

def vertexy(v):
    if isinstance(v, complex):
        return int(round(v.real)), int(round(v.imag))
    else:
        return int(round(v[0])), int(round(v[1]))

def cross(u, v):
    return u.real*v.imag - u.imag*v.real

def svg_path(polygon):
    vertices = polygon.vertices
    ax, ay = vertexy(vertices[0])
    path = f"M{ax},{ay} "
    for v in vertices[1:]:
        vx, vy = vertexy(v)
        path += f"L{vx},{vy} "
    path += "Z"
    return path


def display_svg(filename):
    from IPython.display import SVG, display
    with open(filename, 'r') as f:
        svg = f.read()
        display(SVG(svg))

import numpy as np

def inscribed_square_halfside(grid):
    """
    Given N points where the first two columns are (x, y),
    rotate by 45°, find the limiting extent, and return diag/sqrt(2).
    """
    
    xy = np.array([p.center for p in grid], dtype=float)

    theta = np.deg2rad(45)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    rot_xy = xy @ R.T

    xmin, xmax = rot_xy[:, 0].min(), rot_xy[:, 0].max()
    ymin, ymax = rot_xy[:, 1].min(), rot_xy[:, 1].max()
    diag = min(xmax, -xmin, ymax, -ymin)

    s = diag / np.sqrt(2)
    print(f"Grid size: {len(grid):4d} Inscribed_square_halfside: {s:6.1f}")
    return s

def print_tile_stats(grid):
    # Calculate and print some tile statistics
    xs = [h.x for h in grid]
    ys = [h.y for h in grid]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    colors = Counter(h.color for h in grid)

    print(f"""Tile 
          Count: {len(grid)}
          Colors: {dict(colors)}  
          xmin: {xmin:.4f} xmax: {xmax:.4f}   
          ymin: {ymin:.4f} ymax: {ymax:.4f}
    """)
