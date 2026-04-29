"""Compatibility entrypoint for the LightGBM baseline.

Run with explicit data paths, for example:

python lgb_baseline.py ^
  --train-feature data/train/mengxi_boundary_anon_filtered.csv ^
  --train-label data/train/mengxi_node_price_selected.csv

The full train -> predict -> submission workflow is documented in README.md.
"""

from src.train_lgb import main


if __name__ == "__main__":
    main()

