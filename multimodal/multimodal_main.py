import os
import csv
import uuid
import copy
import yaml

from multimodal_simulation import load_config
from multimodal_runner import run_experiment


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
    Headway is defined by arrivals per 7 days (168 hours), not by total sim length.

    Simulation length = 504 hours = 3 weeks
    So a mode with headway=168 will arrive 3 times in the simulation.
    """

    WEEK_HOURS = 7 * 24  # 168

    vessel_batch_size = 1000
    train_batch_size = 100
    truck_batch_size = 1

    vessel_per_week = 1  # fixed: 1 vessel per 168h
    vessel_headway = WEEK_HOURS / vessel_per_week  # = 168

    # 10 trains per week baseline, then trade for trucks
    base_trains_per_week = 10

    for trains_per_week in [10, 9, 8]:
        trucks_per_week = (base_trains_per_week - trains_per_week) * 100  # 0, 100, 200 ...

        # volumes within ONE week window (168h)
        V_vessel = vessel_per_week * vessel_batch_size
        V_train = trains_per_week * train_batch_size
        V_truck = trucks_per_week * truck_batch_size

        def safe_split(a, b):
            s = a + b
            if s <= 0:
                return 0.5, 0.5
            return a / s, b / s

        # destination splits (each origin splits to the other two modes)
        p_train_from_vessel, p_truck_from_vessel = safe_split(V_train, V_truck)
        p_vessel_from_train, p_truck_from_train = safe_split(V_vessel, V_truck)
        p_vessel_from_truck, p_train_from_truck = safe_split(V_vessel, V_train)

        override = {
            "timetable": {
                "vessel": {
                    "enabled": True,
                    "headway": vessel_headway,        # 168
                    "batch_size": vessel_batch_size,  # 1000
                    "destination_split": {
                        "train": p_train_from_vessel,
                        "truck": p_truck_from_vessel,
                    },
                },
                "train": {
                    "enabled": True,
                    "headway": WEEK_HOURS / max(trains_per_week, 1),
                    "batch_size": train_batch_size,
                    "destination_split": {
                        "vessel": p_vessel_from_train,
                        "truck": p_truck_from_train,
                    },
                },
                "truck": {
                    "enabled": True,
                    "headway": 0.05,
                    "batch_size": truck_batch_size,
                    "destination_split": {
                        "vessel": p_vessel_from_truck,
                        "train": p_train_from_truck,
                    },
                },
            }
        }

        yield override



def flatten_metrics_to_row(run_meta, override, metrics):
    """
    Flatten one run into a single CSV row.
    """
    row = {}

    # ---- metadata ----
    for k, v in run_meta.items():
        row[k] = v

    # ---- inputs (timetable only, explicit & stable) ----
    for mode in ["train", "truck", "vessel"]:
        cfg = override.get("timetable", {}).get(mode, {})
        row[f"{mode}_enabled"] = cfg.get("enabled", False)
        row[f"{mode}_headway"] = cfg.get("headway")
        row[f"{mode}_batch_size"] = cfg.get("batch_size")

    # ---- outputs (mode-level statistics) ----
    mode_level = metrics.get("mode_level", {})
    for mode, stat in mode_level.items():
        for metric_name, stat_dict in stat.items():
            for stat_name, value in stat_dict.items():
                row[f"{mode}.{metric_name}.{stat_name}"] = value

    return row


def main():
    base_config_path = "input/config.yaml"
    runs_root = "output/runs"
    global_csv_path = "output/experiment_results.csv"

    os.makedirs(runs_root, exist_ok=True)
    os.makedirs(os.path.dirname(global_csv_path), exist_ok=True)

    base_config = load_config(base_config_path)
    sim_length = base_config["simulation"]["length"]

    write_header = not os.path.exists(global_csv_path)

    with open(global_csv_path, "a", newline="") as global_f:
        global_writer = None

        for run_idx, override in enumerate(generate_overrides()):
            run_name = f"run_{run_idx:03d}"
            run_dir = os.path.join(runs_root, run_name)
            os.makedirs(run_dir, exist_ok=True)

            # ---- build config snapshot ----
            config_snapshot = deep_merge(base_config, override)

            # ---- dump config snapshot ----
            with open(os.path.join(run_dir, "config_snapshot.yaml"), "w") as f:
                yaml.safe_dump(config_snapshot, f, sort_keys=False)

            # ---- run experiment (simulation + staged outputs) ----
            result = run_experiment(
                config_snapshot=config_snapshot,
                random_seed=1000 + run_idx,
                run_dir=run_dir,
            )

            run_meta = result["metadata"]
            metrics = result["metrics"]

            # ---- flatten & append global CSV ----
            row = flatten_metrics_to_row(run_meta, override, metrics)

            if global_writer is None:
                global_writer = csv.DictWriter(global_f, fieldnames=row.keys())
                if write_header:
                    global_writer.writeheader()
                    write_header = False

            global_writer.writerow(row)

            print(
                f"[{run_name}] "
                f"train_enabled={row.get('train_enabled')} "
                f"truck_enabled={row.get('truck_enabled')} "
                f"success={row.get('success')}"
            )

    print("Finished all runs.")


if __name__ == "__main__":
    main()
