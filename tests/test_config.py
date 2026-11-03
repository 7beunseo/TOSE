from __future__ import annotations

import unittest

from config import get_arg_parser


class LossWeightDefaultsTest(unittest.TestCase):
    def test_normalized_alpha_defaults(self):
        args = get_arg_parser().parse_args([])

        self.assertEqual(args.forward_keep_loss_weight, 0.625)
        self.assertEqual(args.forward_aux_loss_weight, 0.375)
        self.assertEqual(
            args.forward_keep_loss_weight + args.forward_aux_loss_weight,
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
