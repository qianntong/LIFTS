import os
import csv
import copy
import yaml
import math
from collections import OrderedDict
from multimodal_simulation import load_config
from multimodal_runner import run_experiment


def deep_merge(base, override):
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def generate_splits(step=0.1):
    p = 0.2
    while p <= 0.6:
        yield {
            "vessel": {
                "train": round(p, 2),
                "truck": round(1 - p, 2),
            }
        }
        p += step


def estimate_hostler_number(hostler_cycle_time_hr, cranes_per_berth, cranes_per_track):
    x = hostler_cycle_time_hr * 30 * (cranes_per_berth + cranes_per_track)  #  30x = x / (2/60)
    return max(1, int(math.ceil(x)))


def generate_overrides():

    CRANES_PER_BERTH_LIST = list(range(2))
    CRANES_PER_TRACK_LIST =  list(range(2, 11))
    VESSEL_BATCH_LIST = list(range(1100, 1101, 350))


    for vessel_batch in VESSEL_BATCH_LIST:
        for split in generate_splits(step=0.1):
            for cpb in CRANES_PER_BERTH_LIST:
                for cpt in CRANES_PER_TRACK_LIST:
                    HOSTLER_CYCLE_TIME_HR = 0.5 + (vessel_batch/1000)
                    num_trains = math.floor(vessel_batch * split["vessel"]["train"]/100)
                    hostler_number = HOSTLER_CYCLE_TIME_HR / 0.028  * (cpb * 2 + cpt * num_trains)   #600

                    yield {
                        "timetable": {
                            "vessel": {
                                "enabled": True,
                                "weekly_num": 1,
                                "batch_size": vessel_batch,
                                "destination_split": split["vessel"],
                            },
                            "train": {
                                "enabled": True,
                                "batch_size": 100,
                            },
                        },
                        "terminal": {
                            "cranes_per_berth": cpb,
                            "cranes_per_track": cpt,
                            "hostler_number": hostler_number,
                        },
                    }


def get_experiment_csv_header():

    header = [
        "random_seed",
        "run_id",
        "run_time_sec",
        "success",
        "error_message",
    ]

    # ---- vessel input ----
    header += [
        "vessel_weekly_num",
        "vessel_batch_size",
        "vessel_to_train_share",
        "vessel_to_truck_share",
    ]

    # ---- demand ----
    header += [
        "train_weekly_num",
        "train_batch_size",
        "truck_weekly_volume",
    ]

    # ---- resources ----
    header += [
        "cranes_per_berth",
        "cranes_per_track",
        "hostler_number",
        "berth_number",
        "track_number",
    ]

    # ---- performance metrics ----
    MODES = ["train", "truck", "vessel"]
    METRICS = ["avg_container_processing_time"]
    STATS = ["count", "min", "max", "mean", "std"]

    for orig in MODES:
        for dest in MODES:
            if orig == dest:
                continue
            for metric in METRICS:
                for stat in STATS:
                    header.append(f"{orig}_to_{dest}.{metric}.{stat}")

    return header


def flatten_metrics_to_row(run_meta, override, metrics, config_snapshot, weekly_summary):

    row = OrderedDict()

    # META
    row["random_seed"] = run_meta.get("random_seed")
    row["run_id"] = run_meta.get("run_id")
    row["run_time_sec"] = run_meta.get("run_time_sec")
    row["success"] = run_meta.get("success")
    row["error_message"] = run_meta.get("error_message")

    # VESSEL INPUT
    cfg = override.get("timetable", {}).get("vessel", {})
    full_cfg = config_snapshot.get("timetable", {}).get("vessel", {})

    row["vessel_weekly_num"] = (
        cfg.get("weekly_num") if "weekly_num" in cfg
        else full_cfg.get("weekly_num")
    )

    row["vessel_batch_size"] = (
        cfg.get("batch_size") if "batch_size" in cfg
        else full_cfg.get("batch_size")
    )

    split_dict = cfg.get(
        "destination_split",
        full_cfg.get("destination_split", {})
    )

    row["vessel_to_train_share"] = split_dict.get("train")
    row["vessel_to_truck_share"] = split_dict.get("truck")

    # WEEKLY DEMAND
    row["train_weekly_num"] = weekly_summary.get("train_weekly_num")
    row["train_batch_size"] = weekly_summary.get("train_batch_size")
    row["truck_weekly_volume"] = weekly_summary.get("truck_weekly_volume")

    # TERMINAL RESOURCES
    terminal_cfg = config_snapshot.get("terminal", {})
    yard_cfg = config_snapshot.get("yard", {})

    row["berth_number"] = yard_cfg.get("berth_number")
    row["cranes_per_berth"] = terminal_cfg.get("cranes_per_berth")
    row["cranes_per_track"] = terminal_cfg.get("cranes_per_track")
    row["hostler_number"] = terminal_cfg.get("hostler_number")
    row["track_number"] = weekly_summary.get("train_weekly_num")    # tracks depend on trains

    # INITIALIZE ALL HEADER FIELDS
    for col in get_experiment_csv_header():
        row.setdefault(col, None)

    # PERFORMANCE METRICS
    for (orig, dest), metric_block in metrics.items():
        if orig == dest:
            continue
        for metric_name, stat_dict in metric_block.items():
            for stat_name, value in stat_dict.items():
                col = f"{orig}_to_{dest}.{metric_name}.{stat_name}"
                row[col] = round(value,2)

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
                result["weekly_summary"],   # train and truck demand
            )

            writer.writerow(row)

            print(
                f"[{run_name}] "
                f"vessel_batch={row['vessel_batch_size']} "
                f"success={row['success']}"
            )

    print("Finished all runs.")


if __name__ == "__main__":
    main()
