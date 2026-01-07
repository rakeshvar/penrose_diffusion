import math
import cmath
import copy
from collections import Counter
from utils import deg, TOL

from pen_base import Triangle, TriangleGrid, Rhombus
from pen_svg import save_svg
from pen_shapes import circle_tiling

def get_copy():
    return copy.deepcopy(circle_tiling)

#---------------------------------------------------------------------------------
# Reparametrize Test
#---------------------------------------------------------------------------------
def print_reparametrize():
    tiling = get_copy()
    tiling.inflate(4)
    for t in tiling:
        m, a, s = t.reparametrize()
        print(f"center: ({m.real:6.0f}, {m.imag:6.0f}) angle: {a:6.2f}({a*180/math.pi:5.0f}) side: {s:6.1f}")


def print_reparametrize_roundtrip(t):
        rh = Rhombus(t)
        print(f"center: {rh.center:6.0f} tilt: {rh.tilt:6.2f}({deg(rh.tilt):.0f})" 
            f"side: {rh.side:6.2f} topangle: {rh.topangle:6.2f}({rh.topangle*180/math.pi:.0f}) ")

        recreated = rh.triangle()
        print(f"A: {t.A:6.0f} -> {recreated.A:6.0f}", "✓" if abs(t.A - recreated.A) < 1e-6 else "X")
        print(f"B: {t.B:6.0f} -> {recreated.B:6.0f}", "✓" if abs(t.B - recreated.B) < 1e-6 else "X")
        print(f"C: {t.C:6.0f} -> {recreated.C:6.0f}", "✓" if abs(t.C - recreated.C) < 1e-6 else "X")
        print(f"D: {t.D:6.0f} -> {recreated.D:6.0f}", "✓" if abs(t.D - recreated.D) < 1e-6 else "X")

        b = t.A - t.C
        rb = recreated.A - recreated.C
        print(f"Base: {b:6.0f} -> {rb:6.0f}", "✓" if abs(b - rb) < 1e-6 else "X")

        print()


def test_reparametrize():
    for i in range(5):
        tiling = get_copy()
        print(f"--- Inflation level: {i} ---")
        tiling.inflate(i)

        recreated_tiles = TriangleGrid([Rhombus(t).triangle() for t in tiling])
        save_svg(pengrid=tiling, filename=f'pics/{i}_original.svg')
        save_svg(pengrid=recreated_tiles, filename=f'pics/{i}_recreated.svg')

        for t in tiling:
            print_reparametrize_roundtrip(t)

        input("Press Enter to continue to next inflation level...")


#---------------------------------------------------------------------------------
# Duplicate Rhombus Test (Detecting two Triangles that are mirror images of each other)
# --------------------------------------------------------------------------------
def distances_to_the_closest_neighbor(tiling:TriangleGrid):
    """
    For each tile, compute the distance to the closest neighboring tile.
    """
    dists = Counter()
    for i, t1 in enumerate(tiling):
        min_dist = float('inf')
        c1 = t1.center
        for j, t2 in enumerate(tiling):
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


def consistency_checks(self:Rhombus, tri:Triangle):
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
