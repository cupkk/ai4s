from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from .features import TIME_COL


DEFAULT_NWP_VARIABLES = ["u100", "v100", "ghi", "t2m", "tcc", "tp", "sp"]
DEFAULT_NWP_STATS = ["mean", "std", "min", "max"]


def _decode_channels(raw_channels: Iterable[object]) -> List[str]:
    channels: List[str] = []
    for item in raw_channels:
        if isinstance(item, bytes):
            channels.append(item.decode("utf-8"))
        else:
            channels.append(str(item))
    return channels


def extract_one_nc(
    nc_path: Path,
    variables: Sequence[str] = DEFAULT_NWP_VARIABLES,
    stats: Sequence[str] = DEFAULT_NWP_STATS,
) -> pd.DataFrame:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read official .nc files; run: pip install h5py") from exc

    publish_date = pd.Timestamp(nc_path.stem)
    target_date = publish_date + timedelta(days=1)

    rows = []
    with h5py.File(nc_path, "r") as file:
        channels = _decode_channels(file["channel"][()])
        data = file["data"][()]
        # Official shape: time x hour x channel x lat x lon.
        if data.ndim != 5:
            raise ValueError(f"unexpected NWP data shape in {nc_path}: {data.shape}")
        data = data[0]

        for hour in range(data.shape[0]):
            row = {TIME_COL: target_date + pd.Timedelta(hours=hour)}
            for var in variables:
                if var not in channels:
                    continue
                arr = data[hour, channels.index(var), :, :].astype(float)
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    continue
                if "mean" in stats:
                    row[f"nwp_{var}_mean"] = float(np.mean(finite))
                if "std" in stats:
                    row[f"nwp_{var}_std"] = float(np.std(finite))
                if "min" in stats:
                    row[f"nwp_{var}_min"] = float(np.min(finite))
                if "max" in stats:
                    row[f"nwp_{var}_max"] = float(np.max(finite))
            if "u100" in channels and "v100" in channels:
                u = data[hour, channels.index("u100"), :, :].astype(float)
                v = data[hour, channels.index("v100"), :, :].astype(float)
                wind = np.sqrt(u * u + v * v)
                finite_wind = wind[np.isfinite(wind)]
                if finite_wind.size:
                    row["nwp_wind_speed_mean"] = float(np.mean(finite_wind))
                    row["nwp_wind_speed_std"] = float(np.std(finite_wind))
                    row["nwp_wind_speed_max"] = float(np.max(finite_wind))
            rows.append(row)

    hourly = pd.DataFrame(rows)
    quarter_rows = []
    for _, row in hourly.iterrows():
        for minutes in [0, 15, 30, 45]:
            item = row.copy()
            item[TIME_COL] = pd.Timestamp(row[TIME_COL]) + pd.Timedelta(minutes=minutes)
            quarter_rows.append(item)
    return pd.DataFrame(quarter_rows)


def build_nwp_feature_cache(
    nwp_dir: str,
    cache_path: str,
    start_time: str = "",
    end_time: str = "",
    variables: Sequence[str] = DEFAULT_NWP_VARIABLES,
) -> pd.DataFrame:
    nwp_path = Path(nwp_dir)
    if not nwp_path.exists():
        raise FileNotFoundError(f"NWP directory not found: {nwp_dir}")

    start = pd.Timestamp(start_time) if start_time else None
    end = pd.Timestamp(end_time) if end_time else None

    parts = []
    files = sorted(nwp_path.glob("*.nc"))
    for file in files:
        target_date = pd.Timestamp(file.stem) + timedelta(days=1)
        if start is not None and target_date + pd.Timedelta(hours=23, minutes=45) < start:
            continue
        if end is not None and target_date > end:
            continue
        parts.append(extract_one_nc(file, variables=variables))

    if not parts:
        raise ValueError(f"no NWP files matched requested range in {nwp_dir}")

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


def load_or_build_nwp_features(
    nwp_dir: str,
    cache_path: str,
    start_time: str = "",
    end_time: str = "",
) -> pd.DataFrame:
    cache = Path(cache_path)
    start = pd.Timestamp(start_time) if start_time else None
    end = pd.Timestamp(end_time) if end_time else None
    if cache.exists():
        nwp = pd.read_csv(cache)
        nwp[TIME_COL] = pd.to_datetime(nwp[TIME_COL])
        if not nwp.empty:
            covers_start = start is None or nwp[TIME_COL].min() <= start
            covers_end = end is None or nwp[TIME_COL].max() >= end
            if covers_start and covers_end:
                return nwp
    return build_nwp_feature_cache(nwp_dir, cache_path, start_time=start_time, end_time=end_time)


def merge_nwp_features(df: pd.DataFrame, nwp_features: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TIME_COL] = pd.to_datetime(out[TIME_COL])
    nwp = nwp_features.copy()
    nwp[TIME_COL] = pd.to_datetime(nwp[TIME_COL])
    out = out.merge(nwp, on=TIME_COL, how="left")
    nwp_cols = [col for col in out.columns if col.startswith("nwp_")]
    if nwp_cols:
        out[nwp_cols] = out[nwp_cols].ffill().bfill()
    return out
