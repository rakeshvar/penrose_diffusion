import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from code.config import Config
from code.filesystem import CheckPointer, safe_torch_load


class CheckPointerRetentionTest(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda _: 1.,
        )

    def save_epoch(self, checkpointer, epoch, loss, *, is_final=False, save_svg=True):
        checkpointer.save_checkpoint(
            epoch,
            self.model,
            self.optimizer,
            self.scheduler,
            loss,
            is_final=is_final,
        )
        if save_svg:
            checkpointer.save_svg(
                f"<svg>epoch {epoch}</svg>",
                f"sample_e{epoch:03d}.svg",
                loss,
                is_final=is_final,
            )

    def test_keeps_best_five_and_protects_worse_final_with_matching_svgs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpointer = CheckPointer(tmpdir, "unit")
            losses = [0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.7]
            for epoch, loss in enumerate(losses):
                self.save_epoch(
                    checkpointer,
                    epoch,
                    loss,
                    is_final=epoch == len(losses) - 1,
                )

            checkpoint_epochs = {
                safe_torch_load(path)["epoch"]
                for path in Path(tmpdir, "checkpoints").glob("*.pt")
            }
            svg_epochs = {
                int(path.stem.rsplit("e", 1)[1])
                for path in Path(tmpdir, "svg").glob("*.svg")
            }

            self.assertEqual(checkpointer.keep_best_n, 5)
            self.assertEqual(checkpoint_epochs, {0, 1, 2, 3, 4, 6})
            self.assertEqual(svg_epochs, checkpoint_epochs)

    def test_final_inside_best_five_does_not_add_a_sixth_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpointer = CheckPointer(tmpdir, "unit")
            losses = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            for epoch, loss in enumerate(losses):
                self.save_epoch(
                    checkpointer,
                    epoch,
                    loss,
                    is_final=epoch == len(losses) - 1,
                )

            checkpoints = list(Path(tmpdir, "checkpoints").glob("*.pt"))
            svgs = list(Path(tmpdir, "svg").glob("*.svg"))

            self.assertEqual(len(checkpoints), 5)
            self.assertEqual(len(svgs), 5)
            self.assertFalse(Path(tmpdir, "checkpoints", "cpunit_e000.pt").exists())
            self.assertFalse(Path(tmpdir, "svg", "sample_e000.svg").exists())

    def test_checkpoint_retention_does_not_create_svgs_when_sampling_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpointer = CheckPointer(tmpdir, "unit")
            for epoch, loss in enumerate([0.5, 0.4, 0.3, 0.2, 0.1, 0.6]):
                self.save_epoch(
                    checkpointer,
                    epoch,
                    loss,
                    is_final=epoch == 5,
                    save_svg=False,
                )

            self.assertEqual(
                len(list(Path(tmpdir, "checkpoints").glob("*.pt"))),
                6,
            )
            self.assertEqual(list(Path(tmpdir, "svg").glob("*.svg")), [])


class CheckPointerConfigTest(unittest.TestCase):
    def test_keep_best_n_can_be_overridden_from_training_arguments(self):
        argv = [
            "train.py",
            "fixture.npz",
            "dd32",
            "otfm",
            "-t",
            "keep_best_n=3",
        ]
        with patch("sys.argv", argv):
            config = Config()

        self.assertEqual(config.train["keep_best_n"], 3)


if __name__ == "__main__":
    unittest.main()
