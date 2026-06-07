from __future__ import annotations

import unittest
from pathlib import Path

from decoder_only.full_matrix import REQUIRED_TOP_LEVEL_KEYS, build_initial_report, plan_rows


class Args:
    input_checkpoint_path = "/tmp/decoder-base"
    training_dataset = "data/scenic/SCENIC_full_training_dataset.json"
    contrastive_training_dataset = "data/scenic/SCENIC_full_anchor_positive_negative.json"
    benchmark = "data/benchmarks/iot_instruction_benchmark_200.json"
    nproc_per_node = "8"
    cuda_visible_devices = "0,1,2,3,4,5,6,7"
    sparsity_gpu_ids = "0,1,2,3,4,5,6,7"
    regular_sft_epochs = 5
    contrastive_sft_epochs = 5
    recovery_epochs_per_stage = 1
    final_recovery_epochs = 1
    contrastive_loss_weight = 0.1
    contrastive_margin = 0.5
    negative_field = "negative"


class FullMatrixPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = plan_rows(Path("/tmp/out"), Args.training_dataset, Args.sparsity_gpu_ids)

    def test_expected_row_count(self) -> None:
        self.assertEqual(len(self.rows), 20)

    def test_nvidia24_only_appears_at_50_percent(self) -> None:
        nvidia_rows = [row for row in self.rows if row["method"] == "nvidia24"]
        self.assertEqual(len(nvidia_rows), 2)
        self.assertTrue(all(row["pruning_target_sparsity"] == 0.50 for row in nvidia_rows))

    def test_progressive_magnitude_rows(self) -> None:
        progressive_rows = [row for row in self.rows if row["method"] == "progressive_magnitude"]
        self.assertEqual(len(progressive_rows), 4)
        self.assertEqual(
            {(row["family"], row["pruning_target_sparsity"]) for row in progressive_rows},
            {
                ("regular_sft", 0.30),
                ("regular_sft", 0.50),
                ("contrastive_sft", 0.30),
                ("contrastive_sft", 0.50),
            },
        )
        for row in progressive_rows:
            self.assertEqual(row["recovery_epochs_per_stage"], 1)
            self.assertEqual(row["final_recovery_epochs"], 1)
            self.assertEqual(row["pruning_schedule"], "progressive")
            self.assertEqual(row["pruning_base_method"], "magnitude")

    def test_dense_baseline_rows(self) -> None:
        dense_rows = [row for row in self.rows if row["dense_baseline"]]
        self.assertEqual(len(dense_rows), 2)
        for row in dense_rows:
            self.assertEqual(row["method"], "dense")
            self.assertIsNone(row["pruning_target_sparsity"])
            self.assertEqual(row["pruning_schedule"], "none")

    def test_report_top_level_keys(self) -> None:
        report = build_initial_report(Args, Path("/tmp/out"))
        self.assertEqual(tuple(report.keys()), REQUIRED_TOP_LEVEL_KEYS)
        self.assertEqual(report["actual_rows_total"], 20)


if __name__ == "__main__":
    unittest.main()
