import unittest

import torch

from train import build_lr_scheduler


class LearningRateSchedulerTest(unittest.TestCase):
    PEAK_LR = 1e-3
    NUM_EPOCHS = 101
    WARMUP_EPOCHS = 5
    DECAY_EPOCHS = 96

    @staticmethod
    def make_optimizer():
        parameter = torch.nn.Parameter(torch.tensor(0.))
        return torch.optim.SGD([parameter], lr=LearningRateSchedulerTest.PEAK_LR)

    @staticmethod
    def advance(optimizer, scheduler, steps):
        for _ in range(steps):
            optimizer.step()
            scheduler.step()

    def test_warmup_cosine_and_permanent_floor(self):
        optimizer = self.make_optimizer()
        scheduler = build_lr_scheduler(
            optimizer,
            self.PEAK_LR,
            self.NUM_EPOCHS,
        )

        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.01 * self.PEAK_LR)
        self.advance(optimizer, scheduler, self.WARMUP_EPOCHS)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], self.PEAK_LR)

        self.advance(optimizer, scheduler, self.DECAY_EPOCHS // 2)
        halfway_lr = optimizer.param_groups[0]['lr']
        self.assertGreater(halfway_lr, 0.1 * self.PEAK_LR)
        self.assertLess(halfway_lr, self.PEAK_LR)

        self.advance(
            optimizer,
            scheduler,
            self.DECAY_EPOCHS - self.DECAY_EPOCHS // 2,
        )
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.1 * self.PEAK_LR)

        self.advance(optimizer, scheduler, 2 * self.DECAY_EPOCHS)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.1 * self.PEAK_LR)

    def test_new_scheduler_state_round_trip_preserves_floor(self):
        optimizer = self.make_optimizer()
        scheduler = build_lr_scheduler(
            optimizer,
            self.PEAK_LR,
            self.NUM_EPOCHS,
        )
        self.advance(optimizer, scheduler, self.NUM_EPOCHS + 10)

        resumed_optimizer = self.make_optimizer()
        resumed_scheduler = build_lr_scheduler(
            resumed_optimizer,
            self.PEAK_LR,
            self.NUM_EPOCHS,
        )
        resumed_optimizer.load_state_dict(optimizer.state_dict())
        resumed_scheduler.load_state_dict(scheduler.state_dict())

        self.assertAlmostEqual(
            resumed_optimizer.param_groups[0]['lr'],
            0.1 * self.PEAK_LR,
        )
        self.advance(resumed_optimizer, resumed_scheduler, 10)
        self.assertAlmostEqual(
            resumed_optimizer.param_groups[0]['lr'],
            0.1 * self.PEAK_LR,
        )


if __name__ == "__main__":
    unittest.main()
