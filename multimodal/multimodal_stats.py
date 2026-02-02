import numpy as np
from collections import defaultdict

def _normalize_container_metrics(container_events):
    """
    Ensure all time-related fields are float or None.
    """
    normalized = {}

    for cid, info in container_events.items():
        new_info = info.copy()

        for key in ["arrival_actual", "arrival_expected", "departure"]:
            val = new_info.get(key)
            if val is None:
                continue
            try:
                new_info[key] = float(val)
            except Exception:
                new_info[key] = None

        normalized[cid] = new_info

    return normalized


def compute_all_metrics(container_events, analyze_start, analyze_end):
    analyze_start = float(analyze_start)
    analyze_end = float(analyze_end)
    container_events = _normalize_container_metrics(container_events)
    events = _filter_events(container_events, analyze_start, analyze_end)
    mode_stats = _compute_mode_combo_level(events)

    return mode_stats


def _safe_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _filter_events(container_events, analyze_start, analyze_end):
    """
    Keep containers with valid arrival_actual and within window.
    Departure is checked later depending on IC / OC usage.
    """
    result = {}

    for cid, info in container_events.items():
        arr = info.get("arrival_actual")

        if arr is None or not np.isfinite(arr):
            continue
        if not (analyze_start <= arr <= analyze_end):
            continue

        result[cid] = info

    return result


def _compute_mode_combo_level(events):
    """
    Aggregate statistics at (origin_mode, destination_mode) level.

    Valid combinations:
        origin_mode ∈ {train, truck, vessel}
        destination_mode ∈ {train, truck, vessel}
        origin_mode != destination_mode

    For each (origin, destination):
        - Average Container Processing Time = departure - arrival_actual
        - Delay Time = arrival_actual - arrival_expected
          (only for train / vessel arrivals)
    """

    buckets = defaultdict(lambda: {
        "processing": [],
        "delay": []
    })

    for info in events.values():
        origin_mode = info.get("origin_mode")
        dest_mode = info.get("destination_mode")

        # must be valid O-D combos
        if (
            origin_mode is None or
            dest_mode is None or
            origin_mode == dest_mode
        ):
            continue

        arrival_actual = info.get("arrival_actual")
        departure = info.get("departure")
        arrival_expected = info.get("arrival_expected")

        key = (origin_mode, dest_mode)

        # 1) Processing time
        if (
            arrival_actual is not None and
            departure is not None and
            np.isfinite(arrival_actual) and
            np.isfinite(departure)
        ):
            processing_time = departure - arrival_actual
            if np.isfinite(processing_time):
                buckets[key]["processing"].append(processing_time)

        # 2) Delay time (arrival-side, only train / vessel)
        if (
            origin_mode in {"train", "vessel"} and
            arrival_actual is not None and
            arrival_expected is not None and
            np.isfinite(arrival_actual) and
            np.isfinite(arrival_expected)
        ):
            delay = arrival_actual - arrival_expected
            if np.isfinite(delay):
                buckets[key]["delay"].append(delay)

    # summarize
    combo_stats = {}
    for (origin, dest), b in buckets.items():
        combo_stats[(origin, dest)] = {
            "avg_container_processing_time": _summarize_list(b["processing"]),
            "delay_time": _summarize_list(b["delay"]),
        }

    return combo_stats


def flatten_mode_combo_stats(mode_combo_stats):
    flat = {}

    for (orig, dest), metrics in mode_combo_stats.items():
        prefix = f"{orig}_to_{dest}"
        for metric_name, stat_dict in metrics.items():
            for stat, value in stat_dict.items():
                col = f"{prefix}__{metric_name}__{stat}"
                flat[col] = value

    return flat


def _summarize_list(values):
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }