import math
import cmath
from collections import Counter

TOL = 1e-6

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
    ay, ax = vertexy(vertices[0])
    path = f"M{ax},{ay} "
    for v in vertices[1:]:
        vy, vx = vertexy(v)          # Flip x, y to match image convention
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
    Given a set of points, it will give the half-side of the smallest inscribed square.
    The simple appoarch works because the grid is such that the in-circle touches (x_max, 0) or (0, y_max).
    """
    try:
        l1, l2 = [h.center[0] for h in grid], [h.center[1] for h in grid]
    except:
        l1, l2 = [h.center.real for h in grid], [h.center.imag for h in grid]

    return min(max(map(abs, l1)), max(map(abs, l2))) / math.sqrt(2)

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

def zealous_crop(arr, margin=0):
    """
    Remove blank space around the border while maintaining a minimum margin.
    
    Args:
        arr: numpy array of the image (binary after thresholding)
        margin: minimum number of pixels to keep as border around the content
    
    Returns:
        Cropped numpy array with specified margin
    """
    # Find non-empty rows and columns
    non_empty_rows = np.where(arr.any(axis=1))[0]
    non_empty_cols = np.where(arr.any(axis=0))[0]
    
    if non_empty_rows.size == 0 or non_empty_cols.size == 0:
        return arr  # Return original if entirely blank
    
    # Get the content boundaries
    top_content = non_empty_rows[0]
    bottom_content = non_empty_rows[-1]
    left_content = non_empty_cols[0]
    right_content = non_empty_cols[-1]
    
    # Calculate crop boundaries with margin
    top = max(0, top_content - margin)
    bottom = min(arr.shape[0] - 1, bottom_content + margin)
    left = max(0, left_content - margin)
    right = min(arr.shape[1] - 1, right_content + margin)
    
    # Crop the array
    return arr[top:bottom+1, left:right+1]
