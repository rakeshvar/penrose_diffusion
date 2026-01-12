import math
import cmath
from collections import Counter

TOL = 1e-6

def cross(u, v):
    return u.real*v.imag - u.imag*v.real

def deg(rad_or_dir):
    if isinstance(rad_or_dir, complex):
        return cmath.phase(rad_or_dir) * 180 / math.pi
    else:
        return rad_or_dir * 180 / math.pi

def reim(v):
    return v.real, v.imag


def vertexy(v):
    if isinstance(v, complex):
        return v.real, v.imag
    else:
        return v[0], v[1]

def prettxy(v, ndigits=None):
    x, y = vertexy(v)

    if ndigits is None:
        return x, y

    if ndigits == 0:
        return int(round(x)), int(round(y))

    if ndigits is not None:
        return format(x, f".{ndigits}f"), format(y, f".{ndigits}f")

    raise ValueError(f"Unknown value for ndigits: {ndigits}")

def svg_path(polygon, ndigits):
    # Flip x, y to match image convention
    vertices = polygon.vertices
    ay, ax = prettxy(vertices[0], ndigits=ndigits)
    path = f"M{ax},{ay} "
    for v in vertices[1:]:
        vy, vx = prettxy(v, ndigits=ndigits)
        path += f"L{vx},{vy} "
    path += "Z"
    return path


def display_svg(filename):
    from IPython.display import SVG, display
    with open(filename, 'r') as fp:
        svg = fp.read()
        display(SVG(svg))

import numpy as np

def inscribed_square_halfside(grid):
    """
    Given a set of points, it will give the half-side of the smallest inscribed square.
    The simple appoarch works because the grid is such that the in-circle touches (x_max, 0) or (0, y_max).
    """
    try:
        l1, l2 = [h.center[0] for h in grid], [h.center[1] for h in grid]
    except TypeError:
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

class TablePrinter:
    def __init__(self, ncolumns, column_wd):
        self.n_columns = ncolumns
        self.max_name_len = column_wd

    def border(self, l, m, r):
        print(l, end="")
        for i in range(self.n_columns):
            print("─" * self.max_name_len, end="")
            if i < self.n_columns - 1:
                print(m, end="")
            else:
                print(r, end="")
        print()

    def top_line(self):
        self.border("┌", "┬", "┐")

    def mid_line(self):
        self.border("├", "┼", "┤")

    def bot_line(self):
        self.border("└", "┴", "┘")

    def line(self, *args):
        print("│", end="")
        for arg in args:
            # check if arg is a string and has no digits in it
            if isinstance(arg, str) and not any(c.isdigit() for c in arg):
                print(f"{arg:<{self.max_name_len}}│", end="")
            elif isinstance(arg, float):
                print(f"{arg:{self.max_name_len}.4f}│", end="")
            else:
                print(f"{arg:>{self.max_name_len}}│", end="")
        print()



def pairwise_compare(vals, names, title, diag="both"):
    """
    Compares given values to each other and prints a table.
    diag: "both" prints both above abd below the diagonal,
          "down" prints only below the diagonal,
          "up" prints only above the diagonal.
    """
    nvals = len(vals)
    ncolumns = nvals + 1
    max_name_len = max(len(str(name)) for name in names + [title]) + 2

    table = TablePrinter(ncolumns, max_name_len)

    table.top_line()
    table.line(title, *names)
    table.mid_line()

    for i in range(nvals):
        row_vals = []

        for j in range(nvals):
            if j == i:
                val = '(' + f(vals[i]) + ')'
            else:
                if (diag == "up" and j > i) or (diag == "down" and j < i) or (diag == "both"):
                    val = f(vals[i] / vals[j])
                else:
                    val = ""
            row_vals.append(val)

        table.line(names[i], *row_vals)

    table.bot_line()


def linear_compare(vals, names, title, diag=None):
    """
    Simple utility to print comparison table.
    """
    table = TablePrinter(3, 15)
    table.top_line()
    table.line(title, "Value", "Best x Factor")
    table.mid_line()

    best_val = min(vals)

    for val, name in zip(vals, names):
        if val == best_val:
            diff_str = "(Best)"
        else:
            diff = val / best_val
            diff_str = "x " + f(diff)

        table.line(name, f(val), diff_str)

    table.bot_line()

def f(v, N:int|str = ""):
    """
    Format a value to show not too many decimals.
    Parameters:
        v (int or float): The value to format
        N (int)         : Field width for formatting
    Returns:
        str: Formatted string
    """
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer()):
        return f"{v:{N}d}"

    if 0 < v < 2:
        return f"{v:{N}.2f}"
    elif 2 <= v < 10:
        return f"{v:{N}.1f}"
    else:  # val >= 10
        return f"{v:{N}.0f}"


def xysc_to_xyac(xysc, colors=None):
    """
    Convert (x, y, sinθ, cosθ) to (x, y, θ) or (x, y, θ, color).
    """
    xysc = xysc.cpu().numpy()
    s = xysc[..., 2]
    c = xysc[..., 3]
    angle = np.arctan2(s, c)

    if colors is None:
        out = np.stack([xysc[..., 0], xysc[..., 1], angle], axis=-1)
    else:
        colors = colors.cpu().numpy()[:, :, 0]
        out = np.stack([xysc[..., 0], xysc[..., 1], angle, colors], axis=-1)

    return out


def npz_stats(npz_name):
    with np.load(npz_name) as data:
        xya = data['xya']
        x = xya[..., 0].astype(np.float64)
        y = xya[..., 1].astype(np.float64)
        angle = xya[..., 2].astype(np.float64)
        sin = np.sin(angle).astype(np.float64)
        cos = np.cos(angle).astype(np.float64)
        colors = data['colors'].astype(np.float64)

    tp = TablePrinter(6, 15)
    tp.top_line()
    tp.line("Var", "Global/Ind.", "min", "mean", "max", "std")

    def stats(v, name):
        tp.mid_line()
        tp.line(name, "global", np.min(v), np.mean(v), np.max(v), np.std(v))
        tp.line(name, "individual avg",
                np.min(v, axis=-1).mean(), np.mean(v, axis=-1).mean(),
                np.max(v, axis=-1).mean(), np.std(v, axis=-1).mean())


    stats(x, "x")
    stats(y, "y")
    stats(sin, "sin")
    stats(cos, "cos")
    stats(angle, "angle")
    stats(colors, "color")

    tp.bot_line()


def print_config(config):
    print("Config:")
    for k, v in config.items():
        print(k)
        for kk, vv in v.items():
            print(f"\t{kk}: {vv}")
