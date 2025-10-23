import math
import unittest
import copy

from pen_base import PenroseP3, Rhombus
from pen_examples.circle import circle_tiling 
from pen_examples.triangle import triangle_tiling
from pen_svg import save_svg

def get_copy():
	return copy.deepcopy(circle_tiling)

tiling = get_copy()
tiling.inflate(4)
for t in tiling.elements:
	m, a, s = t.reparametrize()
	print(f"center: ({m.real:6.0f}, {m.imag:6.0f}) angle: {a:6.2f}({a*180/math.pi:5.0f}) side: {s:6.1f}")

def test_reparametrize_roundtrip(t):
		rh = Rhombus(t)
		tilt_in_degrees = rh.tilt * 180 / math.pi
		print(f"center: {rh.center:6.0f} tilt: {rh.tilt:6.2f}({tilt_in_degrees:.0f})" 
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

for i in range(5):
	tiling = get_copy()
	print(f"--- Inflation level: {i} ---")
	tiling.inflate(i)

	recreated_tiles = PenroseP3([Rhombus(t).triangle() for t in tiling.elements])
	save_svg(tiling=tiling, filename=f'{i}_original.svg')
	save_svg(tiling=recreated_tiles, filename=f'{i}_recreated.svg')

	for t in tiling.elements:
		test_reparametrize_roundtrip(t)
	
	input("Press Enter to continue to next inflation level...")