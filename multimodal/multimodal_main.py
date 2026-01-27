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


def generate_overrides(sim_length):
    """
    Experiment design:
      - Vessel: 1 vessel every 7 days, batch size 1000
      - Train / Truck trade-off:
            (10 trains, 0 trucks),
            (9 trains, 100 trucks),
            (8 trains, 200 trucks), ...
    """
    vessel_headway = 7 * 24  # hours
    base_train_count = 10

    for n_train in range(base_train_count, -1, -1):
        n_truck = (base_train_count - n_train) * 100

        override = {
            "timetable": {
                "vessel": {
                    "enabled": True,
                    "headway": vessel_headway,
                    "batch_size": 1000,
                }
            }
        }

        if n_train > 0:
            override["timetable"]["train"] = {
                "enabled": True,
                "headway": sim_length / n_train,
                "batch_size": 100,
            }
        else:
            override["timetable"]["train"] = {"enabled": False}

        if n_truck > 0:
            override["timetable"]["truck"] = {
                "enabled": True,
                "headway": sim_length / n_truck,
                "batch_size": 1,
            }
        else:
            override["timetable"]["truck"] = {"enabled": False}

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

        for run_idx, override in enumerate(generate_overrides(sim_length)):
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
