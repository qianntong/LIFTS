import os
import csv
import copy
import yaml
from collections import OrderedDict
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


def generate_splits(step=0.1):
    """
    Generate destination split grid from 0.1 to 1.0.
    vessel->train = p, vessel->truck = 1-p
    train ->vessel = p, train ->truck = 1-p
    """
    p = step
    while p <= 1.0 + 1e-9:
        yield {
            "vessel": {
                "train": round(p, 2),
                "truck": round(1 - p, 2),
            },
            "train": {
                "vessel": round(p, 2),
                "truck": round(1 - p, 2),
            },
        }
        p += step


def generate_overrides():
    """
    Enumerate experiment configurations.

    Exogenous variables:
      - train frequency
      - OD destination splits
      - terminal resource levels

    Timetable realization and truck arrivals are endogenous.
    """

    # ---- terminal resource grid ----
    CRANES_PER_BERTH_LIST = [12]
    CRANES_PER_TRACK_LIST = [4]
    HOSTLER_NUMBER_LIST = [80]

    for trains_per_week in [5]:
        for split in generate_splits(step=0.1):
            for cranes_per_berth in CRANES_PER_BERTH_LIST:
                for cranes_per_track in CRANES_PER_TRACK_LIST:
                    for hostler_number in HOSTLER_NUMBER_LIST:

                        yield {
                            "timetable": {
                                "vessel": {
                                    "enabled": True,
                                    "weekly_num": 1,
                                    "batch_size": 1000,
                                    "destination_split": split["vessel"],
                                },
                                "train": {
                                    "enabled": True,
                                    "weekly_num": trains_per_week,
                                    "batch_size": 100,
                                    "destination_split": split["train"],
                                },
                            },
                            "terminal": {
                                "cranes_per_berth": cranes_per_berth,
                                "cranes_per_track": cranes_per_track,
                                "hostler_number": hostler_number,
                            },
                        }


def get_experiment_csv_header():
    """
    Define a fixed, explicit CSV schema.
    """

    header = []

    # ---- metadata ----
    header += [
        "random_seed",
        "run_id",
        "run_time_sec",
        "success",
        "error_message",
    ]

    # ---- structural inputs ----
    for mode in ["vessel", "train"]:
        header.append(f"{mode}_weekly_num")
        header.append(f"{mode}_batch_size")
        for dest in ["train", "truck", "vessel"]:
            if dest != mode:
                header.append(f"{mode}_to_{dest}_share")

    # ---- terminal resources ----
    header += [
        "cranes_per_berth",
        "cranes_per_track",
        "hostler_number",
    ]

    # ---- yard parameters ----
    header += [
        "berth_number",
        "track_number",
    ]

    # ---- OD-combo metrics ----
    MODES = ["train", "truck", "vessel"]
    METRICS = ["avg_container_processing_time", "delay_time"]
    STATS = ["count", "min", "max", "mean", "std"]

    for orig in MODES:
        for dest in MODES:
            if orig == dest:
                continue
            for metric in METRICS:
                for stat in STATS:
                    header.append(f"{orig}_to_{dest}.{metric}.{stat}")

    return header


def flatten_metrics_to_row(run_meta, override, metrics, config_snapshot):
    row = OrderedDict()

    # ---- metadata ----
    row["random_seed"] = run_meta.get("random_seed")
    row["run_id"] = run_meta.get("run_id")
    row["run_time_sec"] = run_meta.get("run_time_sec")
    row["success"] = run_meta.get("success")
    row["error_message"] = run_meta.get("error_message")

    # ---- structural inputs ----
    for mode in ["vessel", "train"]:
        cfg = override["timetable"][mode]
        row[f"{mode}_weekly_num"] = cfg.get("weekly_num")
        row[f"{mode}_batch_size"] = cfg.get("batch_size")

        for dest in ["train", "truck", "vessel"]:
            if dest != mode:
                row[f"{mode}_to_{dest}_share"] = cfg["destination_split"].get(dest)

    # ---- terminal resources ----
    terminal_cfg = config_snapshot.get("terminal", {})
    row["cranes_per_berth"] = terminal_cfg.get("cranes_per_berth")
    row["cranes_per_track"] = terminal_cfg.get("cranes_per_track")
    row["hostler_number"] = terminal_cfg.get("hostler_number")

    # ---- yard parameters ----
    yard_cfg = config_snapshot.get("yard", {})
    row["berth_number"] = yard_cfg.get("berth_number")
    row["track_number"] = yard_cfg.get("track_number")

    # ---- initialize OD metrics ----
    for col in get_experiment_csv_header():
        row.setdefault(col, None)

    # ---- fill metrics ----
    for (orig, dest), metric_block in metrics.items():
        if orig == dest:
            continue
        for metric_name, stat_dict in metric_block.items():
            for stat_name, value in stat_dict.items():
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
    csv_header = get_experiment_csv_header()

    write_header = not os.path.exists(global_csv_path)

    with open(global_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        if write_header:
            writer.writeheader()

        for run_idx, override in enumerate(generate_overrides()):
            run_name = f"run_{run_idx:05d}"
            run_dir = os.path.join(runs_root, run_name)
            os.makedirs(run_dir, exist_ok=True)

            config_snapshot = deep_merge(base_config, override)

            with open(os.path.join(run_dir, "config_snapshot.yaml"), "w") as cf:
                yaml.safe_dump(config_snapshot, cf, sort_keys=False)

            result = run_experiment(
                random_seed=42,
                config_snapshot=config_snapshot,
                run_id=run_idx,
                run_dir=run_dir,
            )

            row = flatten_metrics_to_row(
                result["metadata"],
                override,
                result["metrics"],
                config_snapshot,
            )

            writer.writerow(row)

            print(
                f"[{run_name}] "
                f"train_weekly_num={row['train_weekly_num']} "
                f"success={row['success']}"
            )

    print("Finished all runs.")


if __name__ == "__main__":
    main()