import os
import csv
import copy
import yaml
from multimodal_simulation import load_config
from multimodal_runner import run_experiment
from collections import OrderedDict

def deep_merge(base, override):
    """
    Recursively merge override into base without mutating base.
    """
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def generate_overrides():
    """
    Control-variable experiment:
    - Only train frequency changes
    - Destination split is exogenous and enumerated
    """

    WEEK_HOURS = 7 * 24  # 168

    vessel_batch_size = 1000
    train_batch_size = 100
    truck_batch_size = 1

    vessel_per_week = 1
    vessel_headway = WEEK_HOURS / vessel_per_week  # 168

    SPLIT_GRID = [
        {
            "vessel": {"train": 0.4, "truck": 0.6},
            "train":  {"vessel": 0.4, "truck": 0.6},
            "truck":  {"vessel": 0.4, "train": 0.6},
        },

        {
            "vessel": {"train": 0.5, "truck": 0.5},
            "train":  {"vessel": 0.5, "truck": 0.5},
            "truck":  {"vessel": 0.5, "train": 0.5},
        },

        {
            "vessel": {"train": 0.6, "truck": 0.4},
            "train":  {"vessel": 0.6, "truck": 0.4},
            "truck":  {"vessel": 0.6, "train": 0.4},
        },
    ]

    # ---- vary train frequency only ----
    for trains_per_week in range(5, 8):
        train_headway = WEEK_HOURS / trains_per_week

        for split in SPLIT_GRID:
            override = {
                "timetable": {
                    "vessel": {
                        "enabled": True,
                        "headway": vessel_headway,
                        "batch_size": vessel_batch_size,
                        "destination_split": split["vessel"],
                    },
                    "train": {
                        "enabled": True,
                        "headway": train_headway,
                        "batch_size": train_batch_size,
                        "destination_split": split["train"],
                    },
                    "truck": {
                        "enabled": True,
                        "headway": 0.06,  # continuous arrival
                        "batch_size": truck_batch_size,
                        "destination_split": split["truck"],
                    },
                }
            }

            yield override



# def generate_overrides():
#     """
#     Vessel headway is defined by arrivals per 7 days (168 hours), not by total sim length.
#     Simulation length = 504 hours = 3 weeks
#     """
#
#     WEEK_HOURS = 7 * 24  # 168
#
#     vessel_batch_size = 1000
#     train_batch_size = 100
#     truck_batch_size = 1
#
#     vessel_per_week = 1  # fixed: 1 vessel per 168h
#     vessel_headway = WEEK_HOURS / vessel_per_week  # = 168
#
#     # 10 trains per week baseline, then trade for trucks
#     base_trains_per_week = 10
#
#     for trains_per_week in range(1,11,1):    # change # of trains
#         trucks_per_week = (base_trains_per_week - trains_per_week) * 100  # 0, 100, 200 ...
#
#         # volumes within ONE week window (168h)
#         V_vessel = vessel_per_week * vessel_batch_size
#         V_train = trains_per_week * train_batch_size
#         V_truck = trucks_per_week * truck_batch_size
#
#         def safe_split(a, b):
#             s = a + b
#             if s <= 0:
#                 return 0.5, 0.5
#             return a / s, b / s
#
#         # destination splits (each origin splits to the other two modes)
#         p_train_from_vessel, p_truck_from_vessel = safe_split(V_train, V_truck)
#         p_vessel_from_train, p_truck_from_train = safe_split(V_vessel, V_truck)
#         p_vessel_from_truck, p_train_from_truck = safe_split(V_vessel, V_train)
#
#         override = {
#             "timetable": {
#                 "vessel": {
#                     "enabled": True,
#                     "headway": vessel_headway,        # 168
#                     "batch_size": vessel_batch_size,  # 1000
#                     "destination_split": {
#                         "train": p_train_from_vessel,
#                         "truck": p_truck_from_vessel,
#                     },
#                 },
#                 "train": {
#                     "enabled": True,
#                     "headway": WEEK_HOURS / max(trains_per_week, 1),
#                     "batch_size": train_batch_size,
#                     "destination_split": {
#                         "vessel": p_vessel_from_train,
#                         "truck": p_truck_from_train,
#                     },
#                 },
#                 "truck": {
#                     "enabled": True,
#                     "headway": 0.06, # not applicable bc waiting for OC: WEEK_HOURS / max(trucks_per_week, 1),
#                     "batch_size": truck_batch_size,
#                     "destination_split": {
#                         "vessel": p_vessel_from_truck,
#                         "train": p_train_from_truck,
#                     },
#                 },
#             }
#         }
#
#         yield override


def flatten_metrics_to_row(run_meta, override, metrics, config_snapshot):
    """
    Flatten one run into a single CSV row.

    Column order:
      1) metadata
      2) input parameters
      3) OD-combo output metrics
    """

    row = OrderedDict()

    # 1) METADATA
    for k in ["random_seed", "run_id", "run_time_sec", "success", "error_message"]:
        if k in run_meta:
            row[k] = run_meta[k]

    # 2) INPUT PARAMETERS
    # ---- timetable inputs ----
    for mode in ["train", "truck", "vessel"]:
        cfg = override.get("timetable", {}).get(mode, {})
        row[f"{mode}_enabled"] = cfg.get("enabled", False)
        row[f"{mode}_headway"] = cfg.get("headway")
        row[f"{mode}_batch_size"] = cfg.get("batch_size")

        dest_split = cfg.get("destination_split", {})
        for dest_mode, p in dest_split.items():
            row[f"{mode}_to_{dest_mode}_share"] = p

    # ---- terminal resources ----
    terminal_cfg = config_snapshot.get("terminal", {})
    row["cranes_per_berth"] = terminal_cfg.get("cranes_per_berth")
    row["cranes_per_track"] = terminal_cfg.get("cranes_per_track")
    row["hostler_number"] = terminal_cfg.get("hostler_number")
    row["hostler_diesel_percentage"] = terminal_cfg.get("hostler_diesel_percentage")

    # ---- yard parameters ----
    yard_cfg = config_snapshot.get("yard", {})
    row["berth_number"] = yard_cfg.get("berth_number")
    row["track_number"] = yard_cfg.get("track_number")

    # 3) OUTPUTS: OD-COMBO
    MODES = ["train", "truck", "vessel"]
    METRICS = ["avg_container_processing_time", "delay_time"]
    STATS = ["count", "min", "max", "mean", "std"]

    # 3.1 OD-metric × stat
    for orig in MODES:
        for dest in MODES:
            if orig == dest:
                continue
            prefix = f"{orig}_to_{dest}"
            for metric_name in METRICS:
                for stat_name in STATS:
                    col = f"{prefix}.{metric_name}.{stat_name}"
                    row[col] = None

    # 3.2 current metrics filling
    # metrics: {(orig, dest): {metric_name: {stat: value}}}
    for key, metric_block in metrics.items():
        if not (isinstance(key, tuple) and len(key) == 2):
            continue

        orig, dest = key
        if orig == dest:
            continue

        if not isinstance(metric_block, dict):
            continue

        for metric_name, stat_dict in metric_block.items():
            if metric_name not in METRICS:
                continue
            if not isinstance(stat_dict, dict):
                continue

            for stat_name, value in stat_dict.items():
                if stat_name not in STATS:
                    continue

                col = f"{orig}_to_{dest}.{metric_name}.{stat_name}"
                row[col] = value

    return row


def main():
    base_config_path = "input/config.yaml"
    runs_root = "output/runs"
    global_csv_path = "output/experiment_results.csv"

    os.makedirs(runs_root, exist_ok=True)
    os.makedirs(os.path.dirname(global_csv_path), exist_ok=True)

    base_config = load_config(base_config_path)

    write_header = not os.path.exists(global_csv_path)

    with open(global_csv_path, "a", newline="") as global_f:
        global_writer = None

        for run_idx, override in enumerate(generate_overrides()):
            run_name = f"run_{run_idx:05d}"
            run_dir = os.path.join(runs_root, run_name)
            os.makedirs(run_dir, exist_ok=True)

            config_snapshot = deep_merge(base_config, override)

            with open(os.path.join(run_dir, "config_snapshot.yaml"), "w") as f:
                yaml.safe_dump(config_snapshot, f, sort_keys=False)

            result = run_experiment(
                random_seed=base_config["simulation"]["random_seed"],
                config_snapshot=config_snapshot,
                run_id=run_idx,
                run_dir=run_dir,
            )

            run_meta = result["metadata"]
            metrics = result["metrics"]

            row = flatten_metrics_to_row(run_meta, override, metrics, config_snapshot)

            if global_writer is None:
                global_writer = csv.DictWriter(global_f, fieldnames=row.keys())
                if write_header:
                    global_writer.writeheader()
                    write_header = False

            global_writer.writerow(row)

            print(f"[{run_name}] "
                f"train_enabled={row.get('train_enabled')} "
                f"truck_enabled={row.get('truck_enabled')} "
                f"success={row.get('success')}"
            )

    print("Finished all runs.")


if __name__ == "__main__":
    main()