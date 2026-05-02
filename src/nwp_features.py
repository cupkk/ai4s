from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from .features import TIME_COL


DEFAULT_NWP_VARIABLES = ["u100", "v100", "ghi", "t2m", "tcc", "tp", "sp"]
DEFAULT_NWP_STATS = ["mean", "std", "min", "max", "p10", "p50", "p90"]


def _decode_channels(raw_channels: Iterable[object]) -> List[str]:
    channels: List[str] = []
    for item in raw_channels:
        if isinstance(item, bytes):
            channels.append(item.decode("utf-8"))
        else:
            channels.append(str(item))
    return channels


def infer_channel_hour_axes(data0_shape: Sequence[int], channel_count: int) -> tuple[int, int]:
    """Infer channel/hour axes after the leading forecast-time axis is removed."""
    if len(data0_shape) != 4:
        raise ValueError(f"expected 4-D data after first axis, got shape={tuple(data0_shape)}")
    first_is_channel = data0_shape[0] == channel_count
    second_is_channel = data0_shape[1] == channel_count
    if first_is_channel and not second_is_channel:
        return 0, 1
    if second_is_channel and not first_is_channel:
        return 1, 0
    raise ValueError(
        "cannot infer channel/hour axes: "
        f"data0_shape={tuple(data0_shape)}, channel_count={channel_count}"
    )


def _lead_hours(file, hour_count: int) -> List[int]:
    if "lead_time" not in file:
        return list(range(hour_count))
    raw = np.asarray(file["lead_time"][()]).reshape(-1)
    if len(raw) != hour_count:
        return list(range(hour_count))
    return [int(value) for value in raw]


def _slice_channel_hour(
    data0: np.ndarray,
    channel_axis: int,
    hour_axis: int,
    channel_idx: int,
    hour_idx: int,
) -> np.ndarray:
    if channel_axis == 0 and hour_axis == 1:
        return data0[channel_idx, hour_idx, :, :]
    if channel_axis == 1 and hour_axis == 0:
        return data0[hour_idx, channel_idx, :, :]
    raise ValueError(f"unsupported channel/hour axes: channel={channel_axis}, hour={hour_axis}")


def inspect_nc_layout(nc_path: Path | str) -> dict:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to inspect official .nc files; run: pip install h5py") from exc

    path = Path(nc_path)
    with h5py.File(path, "r") as file:
        channels = _decode_channels(file["channel"][()])
        data_shape = tuple(int(v) for v in file["data"].shape)
        if len(data_shape) != 5:
            raise ValueError(f"unexpected NWP data shape in {path}: {data_shape}")
        channel_axis, hour_axis = infer_channel_hour_axes(data_shape[1:], len(channels))
        hour_count = int(data_shape[1 + hour_axis])
        lead_hours = _lead_hours(file, hour_count)
        return {
            "file": str(path),
            "data_shape": data_shape,
            "channels": channels,
            "channel_axis_after_time": channel_axis,
            "hour_axis_after_time": hour_axis,
            "hour_count": hour_count,
            "lead_hours": lead_hours,
        }


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
        if data.ndim != 5:
            raise ValueError(f"unexpected NWP data shape in {nc_path}: {data.shape}")
        data = data[0]
        channel_axis, hour_axis = infer_channel_hour_axes(data.shape, len(channels))
        hour_count = data.shape[hour_axis]
        lead_hours = _lead_hours(file, hour_count)

        for hour_idx, lead_hour in enumerate(lead_hours):
            row = {TIME_COL: target_date + pd.Timedelta(hours=lead_hour)}
            for var in variables:
                if var not in channels:
                    continue
                channel_idx = channels.index(var)
                arr = _slice_channel_hour(
                    data,
                    channel_axis=channel_axis,
                    hour_axis=hour_axis,
                    channel_idx=channel_idx,
                    hour_idx=hour_idx,
                ).astype(float)
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
                if "p10" in stats:
                    row[f"nwp_{var}_p10"] = float(np.percentile(finite, 10))
                if "p50" in stats:
                    row[f"nwp_{var}_p50"] = float(np.percentile(finite, 50))
                if "p90" in stats:
                    row[f"nwp_{var}_p90"] = float(np.percentile(finite, 90))
                if var == "ghi":
                    row["nwp_ghi_positive_ratio"] = float(np.mean(finite > 0))
            if "u100" in channels and "v100" in channels:
                u = _slice_channel_hour(
                    data,
                    channel_axis=channel_axis,
                    hour_axis=hour_axis,
                    channel_idx=channels.index("u100"),
                    hour_idx=hour_idx,
                ).astype(float)
                v = _slice_channel_hour(
                    data,
                    channel_axis=channel_axis,
                    hour_axis=hour_axis,
                    channel_idx=channels.index("v100"),
                    hour_idx=hour_idx,
                ).astype(float)
                wind = np.sqrt(u * u + v * v)
                finite_wind = wind[np.isfinite(wind)]
                if finite_wind.size:
                    row["nwp_wind_speed_mean"] = float(np.mean(finite_wind))
                    row["nwp_wind_speed_std"] = float(np.std(finite_wind))
                    row["nwp_wind_speed_max"] = float(np.max(finite_wind))
                    row["nwp_wind_speed_p10"] = float(np.percentile(finite_wind, 10))
                    row["nwp_wind_speed_p50"] = float(np.percentile(finite_wind, 50))
                    row["nwp_wind_speed_p90"] = float(np.percentile(finite_wind, 90))
                    row["nwp_wind_speed_cube_mean"] = float(np.mean(finite_wind**3))
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
    stats: Sequence[str] = DEFAULT_NWP_STATS,
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
        parts.append(extract_one_nc(file, variables=variables, stats=stats))

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
    variables: Sequence[str] = DEFAULT_NWP_VARIABLES,
    stats: Sequence[str] = DEFAULT_NWP_STATS,
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
            required_cols = {f"nwp_{var}_{stat}" for var in variables for stat in stats}
            required_cols.update(
                {
                    "nwp_wind_speed_p90",
                    "nwp_wind_speed_cube_mean",
                    "nwp_ghi_positive_ratio",
                }
            )
            has_required_columns = required_cols.issubset(set(nwp.columns))
            if covers_start and covers_end and has_required_columns:
                return nwp
    return build_nwp_feature_cache(
        nwp_dir,
        cache_path,
        start_time=start_time,
        end_time=end_time,
        variables=variables,
        stats=stats,
    )


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
