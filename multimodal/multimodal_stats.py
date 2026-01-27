import numpy as np
from collections import defaultdict


def compute_all_metrics(container_events, analyze_start, analyze_end):
    """
    Compute IC / OC / delay statistics within an observation window.

    Parameters
    ----------
    container_events : dict
        container_id -> {
            "origin_mode": str,
            "origin_id": str,
            "destination_mode": str,
            "destination_id": str,
            "arrival_actual": float,
            "arrival_expected": float or None,
            "departure": float
        }

    analyze_start, analyze_end : float
        Observation window

    Returns
    -------
    metrics : dict
        {
            "resource_level": dict,
            "mode_level": dict
        }
    """
    filtered = _filter_events(container_events, analyze_start, analyze_end)
    resource_stats = _compute_resource_level(filtered)
    mode_stats = _aggregate_to_mode_level(resource_stats)

    return {
        "resource_level": resource_stats,
        "mode_level": mode_stats,
    }


def _filter_events(container_events, analyze_start, analyze_end):
    """
    Keep containers with valid arrival/departure times
    and arrival within the observation window.
    """
    result = {}

    for cid, info in container_events.items():
        arr = info.get("arrival_actual")
        dep = info.get("departure")

        if arr is None or dep is None:
            continue
        if not np.isfinite(arr) or not np.isfinite(dep):
            continue
        if not (analyze_start <= arr <= analyze_end):
            continue

        result[cid] = info

    return result


def _compute_resource_level(events):
    """
    Compute IC / OC processing time and delay per resource (mode, id).
    """
    stats = defaultdict(lambda: {
        "IC": {
            "processing_time": [],
            "delay_time": [],
        },
        "OC": {
            "processing_time": [],
            "delay_time": [],
        }
    })

    for info in events.values():
        arrival_actual = info["arrival_actual"]
        departure = info["departure"]
        arrival_expected = info.get("arrival_expected")

        proc_time = departure - arrival_actual
        if not np.isfinite(proc_time):
            continue

        # IC: aggregate by origin
        origin_key = (info["origin_mode"], info["origin_id"])
        stats[origin_key]["IC"]["processing_time"].append(proc_time)

        if arrival_expected is not None and np.isfinite(arrival_expected):
            delay = arrival_actual - arrival_expected
            if np.isfinite(delay):
                stats[origin_key]["IC"]["delay_time"].append(delay)

        # OC: aggregate by destination
        dest_key = (info["destination_mode"], info["destination_id"])
        stats[dest_key]["OC"]["processing_time"].append(proc_time)

        if arrival_expected is not None and np.isfinite(arrival_expected):
            delay = arrival_actual - arrival_expected
            if np.isfinite(delay):
                stats[dest_key]["OC"]["delay_time"].append(delay)

    summarized = {}
    for key, v in stats.items():
        summarized[key] = {
            "IC": _summarize(v["IC"]),
            "OC": _summarize(v["OC"]),
        }

    return summarized


def _aggregate_to_mode_level(resource_stats):
    """
    Aggregate resource-level statistics to mode-level statistics.
    """
    mode_buckets = defaultdict(lambda: {
        "IC_processing_time": [],
        "OC_processing_time": [],
        "delay_time": [],
    })

    for (mode, _), stat in resource_stats.items():
        mode_buckets[mode]["IC_processing_time"].extend(
            stat["IC"]["processing_time"]
        )
        mode_buckets[mode]["OC_processing_time"].extend(
            stat["OC"]["processing_time"]
        )
        mode_buckets[mode]["delay_time"].extend(
            stat["IC"]["delay_time"] + stat["OC"]["delay_time"]
        )

    mode_stats = {}
    for mode, buckets in mode_buckets.items():
        mode_stats[mode] = {
            "IC_processing_time": _summarize_list(buckets["IC_processing_time"]),
            "OC_processing_time": _summarize_list(buckets["OC_processing_time"]),
            "delay_time": _summarize_list(buckets["delay_time"]),
        }

    return mode_stats


def _summarize(bucket):
    return {
        "processing_time": _summarize_list(bucket["processing_time"]),
        "delay_time": _summarize_list(bucket["delay_time"]),
    }


def _summarize_list(values):
    if not values:
        return {"min": None, "max": None, "mean": None, "std": None}

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}

    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }
