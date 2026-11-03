from __future__ import annotations

import unittest

import torch

from model import TargetConditionalObservedSetEncoder


MATCHED_DEPTH_MODES = (
    "full",
    "static_column_attention",
    "shared_target_query",
    "uniform_source_aggregation",
)


def build_encoder(mode: str) -> TargetConditionalObservedSetEncoder:
    return TargetConditionalObservedSetEncoder(
        input_dim=8,
        num_dim=4,
        embed_dim=16,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        mechanism_hidden_dim=16,
        cross_attn_layers=3,
        ablation_mode=mode,
    )


class MatchedDepthStructuralAblationTests(unittest.TestCase):
    def test_all_variants_keep_full_tose_depth(self) -> None:
        for mode in MATCHED_DEPTH_MODES:
            with self.subTest(mode=mode):
                encoder = build_encoder(mode)
                self.assertEqual(len(encoder.observed_blocks), 2)
                self.assertEqual(len(encoder.target_blocks), 6)
                self.assertEqual(len(encoder.shared_row_blocks), 0)
                self.assertTrue(
                    all(block.cross_attn.num_heads == 4 for block in encoder.target_blocks)
                )

    def test_all_variants_produce_the_same_output_shapes(self) -> None:
        values = torch.randn(3, 8)
        missing_mask = torch.tensor(
            [
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        )
        for mode in MATCHED_DEPTH_MODES:
            with self.subTest(mode=mode):
                encoder = build_encoder(mode).eval()
                with torch.no_grad():
                    output = encoder(values, missing_mask)
                self.assertEqual(output["field_tokens"].shape, (3, 8, 16))
                self.assertEqual(output["pooled_repr"].shape, (3, 16))

    def test_shared_target_query_passes_one_query_per_row_to_every_block(self) -> None:
        encoder = build_encoder("shared_target_query").eval()
        captured_queries: list[torch.Tensor] = []

        def capture_query(module, args, kwargs):  # type: ignore[no-untyped-def]
            del module, args
            captured_queries.append(kwargs["query"].detach().clone())

        handles = [
            block.cross_attn.register_forward_pre_hook(capture_query, with_kwargs=True)
            for block in encoder.target_blocks
        ]
        try:
            values = torch.randn(2, 8)
            missing_mask = torch.tensor(
                [
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0, 0],
                ],
                dtype=torch.float32,
            )
            with torch.no_grad():
                encoder(values, missing_mask)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(len(captured_queries), 6)
        for query in captured_queries:
            self.assertEqual(query.shape, (2, 1, 16))
            self.assertTrue(torch.allclose(query[0], query[1]))

    def test_all_variants_keep_the_same_parameter_count(self) -> None:
        parameter_counts = {
            mode: sum(parameter.numel() for parameter in build_encoder(mode).parameters())
            for mode in MATCHED_DEPTH_MODES
        }
        self.assertEqual(len(set(parameter_counts.values())), 1, parameter_counts)


if __name__ == "__main__":
    unittest.main()
