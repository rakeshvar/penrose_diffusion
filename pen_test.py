import math
import unittest

from pen_base import PenroseP3, Thin


# Recreate the initial configuration from example4.py (but keep inflations small so tests run fast)
scale = 1000
theta = math.pi / 5
rot = math.cos(theta) + 1j * math.sin(theta)
B = 0 + 0j

A1 = scale + 0.j
C1 = C2 = A1 * rot
A2 = A3 = C1 * rot
C3 = C4 = A3 * rot
A4 = A5 = C4 * rot
C5 = -A1

tiling = PenroseP3([
	Thin(A1, B, C1), 
	Thin(A2, B, C2),
	Thin(A3, B, C3), 
	Thin(A4, B, C4),
	Thin(A5, B, C5)])

tiling.add_conjugate_elements()
# Use a modest number of inflations for the tests so they remain fast but include both Thin and Fatt tiles
tiling.inflate(times=3)


class TestReparametrizeRoundtrip(unittest.TestCase):
	def test_roundtrip_for_sample_tiles(self):
		"""For a sample of tiles, reparametrize and recreate via from_reparametrized and compare vertices."""
		max_tests = min(60, len(tiling.elements))
		for t in tiling.elements[:max_tests]:
			M, angle, side = t.reparametrize()
			recreated = t.__class__.from_reparametrized(M, angle, side)

			# Vertices should match up to a small numerical tolerance
			tol = 1e-6
			self.assertTrue(abs(t.A - recreated.A) < tol, f"A differs: {t.A} vs {recreated.A}")
			self.assertTrue(abs(t.B - recreated.B) < tol, f"B differs: {t.B} vs {recreated.B}")
			self.assertTrue(abs(t.C - recreated.C) < tol, f"C differs: {t.C} vs {recreated.C}")

	def test_reparametrize_center_and_angle_range(self):
		"""Sanity-check the output of reparametrize for one tile: centre equals triangle.centre() and angle in [-pi, pi]."""
		t = tiling.elements[0]
		M, angle, side = t.reparametrize()

		# centre is midpoint of base
		self.assertAlmostEqual(M.real, t.centre().real, places=8)
		self.assertAlmostEqual(M.imag, t.centre().imag, places=8)

		# angle should be in principal branch [-pi, pi]
		self.assertTrue(-math.pi <= angle <= math.pi)


if __name__ == '__main__':
	unittest.main()

