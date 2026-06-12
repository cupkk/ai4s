# Data Layout

This directory is the expected local mount point for competition-provided input
files. The repository can redistribute source code, configs, tests, and small
documentation examples, but it must not redistribute the official competition
CSVs, NWP `.nc` files, leaderboard credentials, generated model artifacts, or
private submission attempts.

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

The top-level README also documents the legacy competition extraction path used
by the original scripts:

```text
to_sais_new/to_sais_new/train/mengxi_boundary_anon_filtered.csv
to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv
to_sais_new/to_sais_new/test/test_in_feature_ori.csv
to_sais_new/to_sais_new/all_nc/*.nc
```

Both `data/` and `to_sais_new/` are ignored by Git so contributors can run the
pipeline locally without accidentally committing user-provided data. If you add
a new script, keep defaults under one of those ignored roots or require the path
through a CLI flag.

## Redistributable vs Local-Only Files

Redistributable files that belong in Git:

- source code under `src/`
- pipeline scripts under `scripts/`
- config templates under `configs/`
- tests with synthetic or minimal inline fixtures
- documentation that describes expected paths and output shapes

Local-only files that must stay out of Git:

- official competition CSVs and `.nc` weather archives
- private `output.csv` submissions or leaderboard exports
- trained model files, large prediction tables, and generated diagnostics
- credentials, cookies, notebook checkpoints, or local absolute paths

## Diagnostic Output Examples

NWP layout and time-alignment diagnostics:

```powershell
python -m src.nwp_diagnostics `
  --nwp-dir data/all_nc `
  --max-files 1 `
  --output outputs/nwp_diagnostics.csv
```

Expected console shape:

```text
source_file date       data_shape channel_axis_after_time hour_axis_after_time hour_count timezone_label ghi_max_hour ghi_peak_time_ok_10_15 missing_ratio
20250101.nc 2025-01-01 24x11x96   1                       2                    96         BJT            12           True                    0.0
saved_nwp_diagnostics=outputs/nwp_diagnostics.csv
```

Submission validation:

```powershell
python -m src.check_submission --submission output.csv
```

Expected success shape:

```text
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0
```

If a diagnostic prints private paths or file names from your machine, replace
them with placeholders such as `data/all_nc/20250101.nc` before sharing logs in
issues or pull requests.
