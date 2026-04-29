# Data Layout

Place the official competition files under this directory:

```text
data/
  train/
    mengxi_boundary_anon_filtered.csv
    mengxi_node_price_selected.csv
  test/
    test_in_feature_ori.csv
  all_nc/
    *.nc
```

The first-round pipeline only uses the 15-minute boundary-condition CSV files.
Weather `.nc` files are intentionally left for a later model iteration.

