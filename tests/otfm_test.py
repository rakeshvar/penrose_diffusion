import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from code.augment import GeometryAugment
from code.models.diffuser import OTFlowMatcher
from code.models.directdiff.direct_model import DirectDiffusionModel
from code.models.directdiff.ot_prefetch import OTBatchPrefetcher
from code.utils.advanced import (
    ANGLE_SCALE,
    sample_ot_noise,
    scaled_to_xyac,
    wrap_angle,
    xya_to_scaled,
)
from code.utils.lossy import ScipyBatchedLSA, gather_by_permutation, ot_cost_matrix


class ScaledAngleTest(unittest.TestCase):
    def test_angle_round_trip(self):
        angles = torch.tensor(
            [-3. * math.pi, -math.pi, -1.2, 0., 2.4, math.pi, 3. * math.pi]
        )
        xya = torch.stack((torch.arange(len(angles)), -torch.arange(len(angles)), angles), dim=-1)

        scaled, colors = xya_to_scaled(xya)
        recovered = scaled_to_xyac(scaled)

        self.assertIsNone(colors)
        torch.testing.assert_close(recovered[..., :2], xya[..., :2])
        torch.testing.assert_close(recovered[..., 2], wrap_angle(angles))
        self.assertTrue((scaled[..., 2] >= -math.sqrt(3.) - 1e-6).all())
        self.assertTrue((scaled[..., 2] <= math.sqrt(3.) + 1e-6).all())

    def test_color_is_preserved_when_requested(self):
        xya = torch.tensor([[[1., 2., math.pi / 2]]])
        colors = torch.tensor([[1.]])
        scaled, _ = xya_to_scaled(xya)
        xyac = scaled_to_xyac(scaled, colors)

        self.assertEqual(xyac.shape, (1, 1, 4))
        torch.testing.assert_close(xyac[..., 3], colors)
        torch.testing.assert_close(xyac[..., 2], torch.tensor([[math.pi / 2]]))

    def test_scale_matches_uniform_unit_variance(self):
        self.assertAlmostEqual(ANGLE_SCALE, math.sqrt(3.) / math.pi)
        generator = torch.Generator().manual_seed(7)
        angles = (torch.rand(200_000, generator=generator) * 2. - 1.) * math.pi
        scaled = angles * ANGLE_SCALE

        self.assertAlmostEqual(float(scaled.mean()), 0., places=2)
        self.assertAlmostEqual(float(scaled.var(unbiased=False)), 1., places=2)


class ScipyBatchedLSATest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(11)
        self.x0 = torch.randn((4, 12, 3), generator=generator)
        self.noise = torch.randn((4, 12, 3), generator=generator)
        self.cost = ot_cost_matrix(self.x0, self.noise)
        # Alternating colors so unconstrained OT often wants to cross classes.
        self.colors = torch.tensor([[i % 2 for i in range(12)]] * 4)

    def test_threaded_solver_returns_optimal_permutations(self):
        with ScipyBatchedLSA(max_workers=1) as serial:
            expected = serial.solve(self.cost)
        with ScipyBatchedLSA(max_workers=4) as threaded:
            actual = threaded.solve(self.cost)

        expected_cost = self.cost.gather(2, expected.unsqueeze(-1)).sum()
        actual_cost = self.cost.gather(2, actual.unsqueeze(-1)).sum()
        torch.testing.assert_close(actual_cost, expected_cost)
        for permutation in actual:
            np.testing.assert_array_equal(
                np.sort(permutation.numpy()), np.arange(self.noise.shape[1])
            )

    def test_color_constrained_assignment_stays_within_color(self):
        with ScipyBatchedLSA(max_workers=4) as solver:
            unconstrained = solver.solve(self.cost)
            constrained = solver.solve(self.cost, self.colors)

        # Unconstrained OT on this fixture crosses colors; color LSA must not.
        crossed = False
        for batch in range(len(self.colors)):
            src = self.colors[batch]
            dst = self.colors[batch, unconstrained[batch]]
            crossed = crossed or bool((src != dst).any())
            torch.testing.assert_close(
                self.colors[batch],
                self.colors[batch, constrained[batch]],
            )
            for color in (0, 1):
                select = (self.colors[batch] == color).nonzero(as_tuple=False).squeeze(-1)
                sub = self.cost[batch].index_select(0, select).index_select(1, select)
                local = constrained[batch, select]
                # Map global column indices back into the color block.
                inverse = torch.empty_like(constrained[batch])
                inverse[select] = torch.arange(len(select))
                local_cols = inverse[local]
                _, expected_cols = linear_sum_assignment(sub.numpy())
                np.testing.assert_array_equal(local_cols.numpy(), expected_cols)
        self.assertTrue(crossed, "fixture should exercise cross-color unconstrained OT")

    def test_gather_preserves_each_noise_multiset(self):
        with ScipyBatchedLSA(max_workers=2) as solver:
            permutation = solver.solve(self.cost, self.colors)
        matched = gather_by_permutation(self.noise, permutation)

        for batch in range(len(self.noise)):
            original_rows = sorted(map(tuple, self.noise[batch].tolist()))
            matched_rows = sorted(map(tuple, matched[batch].tolist()))
            self.assertEqual(matched_rows, original_rows)

    def test_closed_solver_rejects_work(self):
        solver = ScipyBatchedLSA(max_workers=2)
        solver.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            solver.solve_numpy(self.cost.numpy())


class OTBatchPrefetcherTest(unittest.TestCase):
    @staticmethod
    def make_batches():
        generator = torch.Generator().manual_seed(19)
        batches = []
        for _ in range(3):
            xya = torch.randn((4, 10, 3), generator=generator)
            xya[..., 2] = (torch.rand((4, 10), generator=generator) * 2. - 1.) * math.pi
            colors = torch.randint(0, 2, (4, 10), generator=generator)
            labels = torch.randint(0, 5, (4,), generator=generator)
            batches.append((xya, colors, labels))
        return batches

    def test_cpu_fallback_drains_batches_and_matches_optimally(self):
        with OTBatchPrefetcher(
            "cpu",
            GeometryAugment(),
            max_workers=2,
            seed=23,
            async_enabled=True,
        ) as prefetcher:
            prepared = list(prefetcher.iter_prepared(self.make_batches()))

        self.assertEqual(len(prepared), 3)
        self.assertEqual(prefetcher.mean_wait_ms, 0.)
        for batch in prepared:
            self.assertEqual(batch.x0.shape, (4, 10, 3))
            self.assertEqual(batch.noise.shape, (4, 10, 3))
            with ScipyBatchedLSA(max_workers=1) as solver:
                rematch = solver.solve(
                    ot_cost_matrix(batch.x0, batch.noise),
                    batch.colors,
                )
            identity = torch.arange(10).expand(4, -1)
            cost = ot_cost_matrix(batch.x0, batch.noise)
            identity_cost = cost.gather(2, identity.unsqueeze(-1)).sum()
            optimal_cost = cost.gather(2, rematch.unsqueeze(-1)).sum()
            # Color-constrained rematch: identity must already be optimal within colors.
            torch.testing.assert_close(identity_cost, optimal_cost)

    def test_cpu_fallback_is_deterministic_with_dedicated_seed(self):
        batches = self.make_batches()
        with OTBatchPrefetcher(
            "cpu", GeometryAugment(), max_workers=2, seed=31
        ) as first:
            first_result = list(first.iter_prepared(batches))
        with OTBatchPrefetcher(
            "cpu", GeometryAugment(), max_workers=2, seed=31
        ) as second:
            second_result = list(second.iter_prepared(batches))

        for left, right in zip(first_result, second_result):
            torch.testing.assert_close(left.x0, right.x0)
            torch.testing.assert_close(left.noise, right.noise)
            torch.testing.assert_close(left.colors, right.colors)
            torch.testing.assert_close(left.labels, right.labels)

    def test_closed_prefetcher_rejects_new_batches(self):
        prefetcher = OTBatchPrefetcher("cpu", GeometryAugment(), max_workers=1)
        prefetcher.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            prefetcher.prepare(self.make_batches()[0])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_prefetch_matches_synchronous_preparation(self):
        batches = [
            tuple(tensor.pin_memory() for tensor in batch)
            for batch in self.make_batches()
        ]
        with OTBatchPrefetcher(
            "cuda",
            GeometryAugment(),
            max_workers=4,
            seed=37,
            async_enabled=True,
        ) as asynchronous:
            async_result = list(asynchronous.iter_prepared(batches))
            self.assertEqual(len(asynchronous.wait_times_ms), len(batches))

        with OTBatchPrefetcher(
            "cuda",
            GeometryAugment(),
            max_workers=4,
            seed=37,
            async_enabled=False,
        ) as synchronous:
            sync_result = list(synchronous.iter_prepared(batches))

        for async_batch, sync_batch in zip(async_result, sync_result):
            torch.testing.assert_close(async_batch.x0, sync_batch.x0)
            torch.testing.assert_close(async_batch.noise, sync_batch.noise)
            torch.testing.assert_close(async_batch.colors, sync_batch.colors)
            torch.testing.assert_close(async_batch.labels, sync_batch.labels)


class OTFlowMatcherTest(unittest.TestCase):
    def test_structured_base_statistics(self):
        generator = torch.Generator().manual_seed(41)
        noise = sample_ot_noise(
            (2_000, 96, 3),
            device="cpu",
            generator=generator,
        )

        torch.testing.assert_close(
            noise[..., :2].mean(dim=(0, 1)),
            torch.zeros(2),
            atol=0.01,
            rtol=0.,
        )
        torch.testing.assert_close(
            noise.var(dim=(0, 1), unbiased=False),
            torch.ones(3),
            atol=0.02,
            rtol=0.,
        )
        self.assertGreaterEqual(float(noise[..., 2].min()), -math.sqrt(3.))
        self.assertLessEqual(float(noise[..., 2].max()), math.sqrt(3.))

    def test_flow_path_and_velocity_target(self):
        matcher = OTFlowMatcher(ndims=2, num_timesteps=11)
        x0 = torch.randn(3, 8, 3)
        noise = torch.randn(3, 8, 3)
        times = torch.tensor([0, 5, 10])

        xt, velocity = matcher.q_sample(x0, times, ϵ=noise)

        torch.testing.assert_close(xt[0], x0[0])
        torch.testing.assert_close(xt[1], (x0[1] + noise[1]) / 2.)
        torch.testing.assert_close(xt[2], noise[2])
        torch.testing.assert_close(velocity, noise - x0)

    def test_unmatched_forward_endpoint_is_rejected(self):
        matcher = OTFlowMatcher(ndims=2)
        with self.assertRaisesRegex(ValueError, "matched noise"):
            matcher.q_sample(torch.randn(2, 4, 3), torch.ones(2, dtype=torch.long))

    def test_sampling_uses_structured_base(self):
        class ZeroDenoiser(nn.Module):
            io_dim = 3

            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.tensor(0.))

            def forward(self, x, colors, times, labels):
                return torch.zeros_like(x)

        matcher = OTFlowMatcher(ndims=2)
        colors = torch.zeros((1_000, 32), dtype=torch.long)
        labels = torch.zeros(1_000, dtype=torch.long)
        samples = matcher.sample(ZeroDenoiser(), colors, labels, num_steps=1)

        self.assertEqual(samples.shape, (1_000, 32, 3))
        self.assertAlmostEqual(float(samples[..., 0].var(unbiased=False)), 1., places=1)
        self.assertAlmostEqual(float(samples[..., 1].var(unbiased=False)), 1., places=1)
        self.assertAlmostEqual(float(samples[..., 2].var(unbiased=False)), 1., places=1)
        self.assertGreaterEqual(float(samples[..., 2].min()), -math.sqrt(3.))
        self.assertLessEqual(float(samples[..., 2].max()), math.sqrt(3.))


class DirectOTFMModelTest(unittest.TestCase):
    @staticmethod
    def config():
        return {
            "model": "direct",
            "diffuser": "otfm",
            "loss": "npl",
            "representation": "scaled_xya",
            "io_dim": 3,
            "ot_async_prefetch": False,
            "ot_workers": 2,
            "ot_seed": 47,
            "class_embed_dim": 4,
            "time_embed_dim": 8,
            "d_model": 16,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.,
        }

    @staticmethod
    def batch():
        generator = torch.Generator().manual_seed(53)
        xya = torch.randn((2, 8, 3), generator=generator)
        xya[..., 2] = (torch.rand((2, 8), generator=generator) * 2. - 1.) * math.pi
        colors = torch.randint(0, 2, (2, 8), generator=generator)
        labels = torch.randint(0, 3, (2,), generator=generator)
        return xya, colors, labels

    def test_training_and_sampling_use_three_dimensional_flow(self):
        model = DirectDiffusionModel(
            self.config(),
            SimpleNamespace(num_classes=3),
        )
        model.runtime_setup()
        try:
            loss, auxiliary = model.train_step(*self.batch())
            self.assertEqual(loss.ndim, 0)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(auxiliary.numel(), 0)
            loss.backward()
            self.assertEqual(model.denoiser.io_dim, 3)
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

            _, colors, labels = self.batch()
            samples = model.sample(colors, labels, num_steps=2)
            self.assertEqual(samples.shape, (2, 8, 4))
            self.assertTrue(torch.isfinite(samples).all())
            self.assertTrue((samples[..., 2] >= -math.pi).all())
            self.assertTrue((samples[..., 2] <= math.pi).all())
        finally:
            model.runtime_teardown()

    def test_invalid_otfm_representation_is_rejected(self):
        config = self.config()
        config["io_dim"] = 4
        with self.assertRaisesRegex(ValueError, "io_dim=3"):
            DirectDiffusionModel(config, SimpleNamespace(num_classes=3))

    def test_existing_direct_diffusion_path_remains_four_dimensional(self):
        config = self.config()
        config.update(
            {
                "diffuser": "ddpm",
                "loss": "npl",
                "representation": "xysc",
                "io_dim": 4,
            }
        )
        model = DirectDiffusionModel(config, SimpleNamespace(num_classes=3))
        loss, _ = model.train_step(*self.batch())
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.denoiser.io_dim, 4)


if __name__ == "__main__":
    unittest.main()
