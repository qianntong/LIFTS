import numpy as np
from collections import defaultdict


def compute_all_metrics(container_events, analyze_start, analyze_end):
    """
    Final statistics:
    One simulation -> 3 rows (train / truck / vessel)
    Each row contains:
    - IC processing time stats (origin_mode == mode)
    - OC processing time stats (destination_mode == mode)
    - delay time stats (arrival_actual - arrival_expected)
    """

    events = _filter_events(container_events, analyze_start, analyze_end)
    mode_stats = _compute_mode_level(events)

    return mode_stats


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


def _compute_mode_level(events):
    """
    Aggregate statistics directly at mode level:
    train / truck / vessel

    For each mode:
    - IC processing time: containers with origin_mode == mode
    - OC processing time: containers with destination_mode == mode
    - Delay time: arrival-side delay for this mode
    """

    buckets = defaultdict(lambda: {
        "IC_processing": [],
        "OC_processing": [],
        "delay": [],
    })

    for info in events.values():
        origin_mode = info.get("origin_mode")
        dest_mode = info.get("destination_mode")

        arrival_actual = info.get("arrival_actual")
        departure = info.get("departure")
        arrival_expected = info.get("arrival_expected")

        # IC processing time (by origin mode)
        if (
            origin_mode is not None
            and departure is not None
            and np.isfinite(departure)
        ):
            ic_time = departure - arrival_actual
            if np.isfinite(ic_time):
                buckets[origin_mode]["IC_processing"].append(ic_time)

        # OC processing time (by destination mode)
        if (
            dest_mode is not None
            and departure is not None
            and np.isfinite(departure)
        ):
            oc_time = departure - arrival_actual
            if np.isfinite(oc_time):
                buckets[dest_mode]["OC_processing"].append(oc_time)

        # Delay time (arrival-side, by mode)
        if (
            origin_mode is not None
            and arrival_expected is not None
            and np.isfinite(arrival_expected)
        ):
            delay = arrival_actual - arrival_expected
            if np.isfinite(delay):
                buckets[origin_mode]["delay"].append(delay)

    # Summarize per mode
    mode_stats = {}
    for mode, b in buckets.items():
        mode_stats[mode] = {
            "IC_processing_time": _summarize_list(b["IC_processing"]),
            "OC_processing_time": _summarize_list(b["OC_processing"]),
            "delay_time": _summarize_list(b["delay"]),
        }

    return mode_stats


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
