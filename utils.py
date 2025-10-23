import math
import cmath

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