from collections import Counter

import numpy as np

#-------------------------------------------------------------------------------
# Print Utilities
#-------------------------------------------------------------------------------
def print_config(config):
    print("Config:")
    for k, v in config.items():
        print(k)
        for kk, vv in v.items():
            print(f"\t{kk}: {vv}")

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

class TablePrinter:
    def __init__(self, ncolumns, column_wd, ndigits=4):
        self.n_columns = ncolumns
        self.max_name_len = column_wd
        self.ndigits = ndigits

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
                print(f"{arg:{self.max_name_len}.{self.ndigits}f}│", end="")
            else:
                print(f"{arg:>{self.max_name_len}}│", end="")
        print()


def npz_stats(npz_name):
    print("\nFile: ", npz_name)

    with np.load(npz_name) as data:
        xya = data['xya']
        x = xya[..., 0].astype(np.float64)
        y = xya[..., 1].astype(np.float64)
        angle = xya[..., 2].astype(np.float64)
        sin = np.sin(angle).astype(np.float64)
        cos = np.cos(angle).astype(np.float64)
        colors = data['colors'].astype(np.float64)

    tp = TablePrinter(7, 11)
    tp.top_line()
    tp.line("Variable", "Global/Ind.", "min", "mean", "max", "std", "range")

    def stats(v, name):
        tp.mid_line()
        tp.line(name, "global", np.min(v), np.mean(v), np.max(v), np.std(v), np.max(v) - np.min(v))
        tp.line(name, "indiv. avg",
                np.min(v, axis=-1).mean(), np.mean(v, axis=-1).mean(),
                np.max(v, axis=-1).mean(), np.std(v, axis=-1).mean(),
                (np.max(v, axis=-1) - np.min(v, axis=-1)).mean())
        tp.line(name, "indiv. max",
                np.min(v, axis=-1).max(), np.mean(v, axis=-1).max(),
                np.max(v, axis=-1).max(), np.std(v, axis=-1).max(),
                (np.max(v, axis=-1) - np.min(v, axis=-1)).max())
        tp.line(name, "indiv. min",
                np.min(v, axis=-1).min(), np.mean(v, axis=-1).min(),
                np.max(v, axis=-1).min(), np.std(v, axis=-1).min(),
                (np.max(v, axis=-1) - np.min(v, axis=-1)).min())


    stats(x, "x")
    stats(y, "y")
    stats(sin, "sin")
    stats(cos, "cos")
    stats(angle, "angle")
    stats(colors, "color")
    tp.bot_line()

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
    if isinstance(v, int):
        return f"{v:{N}d}"

    if 0 < v < 2:
        return f"{v:{N}.2f}"
    elif 2 <= v < 10:
        return f"{v:{N}.1f}"
    else:  # val >= 10
        return f"{v:{N}.0f}"

#------------------------------------------------------------------------------
# IO
#------------------------------------------------------------------------------
def safe_path(folder, fname):
    if isinstance(folder, (str,)):
        return f"{folder.rstrip('/')}/{fname}"
    else:
        return folder / fname

def infer_type(val_str):
    vls = val_str.lower()
    if vls == 'none':    return None
    if vls == 'true':    return True
    if vls == 'false':   return False
    try:                 return int(val_str)
    except ValueError:   pass    
    try:                 return float(val_str)
    except ValueError:   return val_str
