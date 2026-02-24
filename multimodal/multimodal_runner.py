import json
from multimodal_simulation import *
from multimodal_stats import compute_all_metrics, flatten_mode_combo_stats
from multimodal_timetable import *

def run_experiment(random_seed, config_snapshot, run_id, run_dir=None):
    """
    Run ONE simulation
        compute container process & delay time metrics for various mode combos.
    """
    container_events.clear()
    assert len(container_events) == 0, "[WARNING] container_events NOT cleared before run!!!!"

    metadata = {
        "random_seed": random_seed,
        "run_id": run_id,
        "run_time_sec": None,
        "success": True,
        "error_message": "",
    }

    weekly_summary = {}
    start_time = time.perf_counter()

    try:
        env = simpy.Environment()

        terminal = Terminal(env, config_snapshot)
        timetable, weekly_summary = generate_timetable(config_snapshot,terminal) # verbose=True

        # Dump timetable
        if run_dir is not None:
            _dump_timetable(
                timetable,
                os.path.join(run_dir, "timetable.csv")
            )

        # Process arrival
        for entry in timetable:
            if entry["mode"] in ["train", "vessel"]:
                env.process(train_vessel_arrival_process(env, terminal, entry))
            elif entry["mode"] == "truck":
                env.process(truck_arrival_process(env, terminal, entry))

        # Run simulation
        sim_length = config_snapshot["simulation"]["length"]
        env.run(until=sim_length)

        # Dump container events
        container_metrics = export_container_events_to_three_csvs(
                filepath=run_dir,
                prefix="container"
            )

        # Compute statistics
        metrics = compute_all_metrics(
            container_events=container_metrics,
            analyze_start=terminal.analyze_start,
            analyze_end=terminal.analyze_end,
        )

        # Dump metrics (both wide + long)
        if run_dir is not None and metrics:
            _dump_mode_level_metrics(
                metrics,
                os.path.join(run_dir, "metrics_mode_level.csv")
            )
            _dump_mode_level_metrics_long(
                metrics,
                os.path.join(run_dir, "metrics_mode_level_long.csv")
            )

            flat_metrics = flatten_mode_combo_stats(metrics)
            metadata.update(flat_metrics)

    except Exception as e:
        metadata["success"] = False
        metadata["error_message"] = repr(e)
        metrics = {}

    finally:
        metadata["run_time_sec"] = time.perf_counter() - start_time

        if run_dir is not None:
            with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
                json.dump(metadata, f, indent=2)

    return {
        "metadata": metadata,
        "metrics": metrics,
        "weekly_summary": weekly_summary,
    }


def _dump_timetable(timetable, path):
    if not timetable:
        return

    fieldnames = sorted(timetable[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in timetable:
            writer.writerow(row)


def _dump_mode_level_metrics(mode_level_metrics, path):
    """
    mode_level_metrics:
      {
        "train": {...},
        "truck": {...},
        "vessel": {...}
      }
    """

    rows = []
    for mode, stat in mode_level_metrics.items():
        row = {"mode": mode}
        for metric_name, stat_dict in stat.items():
            for k, v in stat_dict.items():
                row[f"{metric_name}_{k}"] = v
        rows.append(row)

    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _dump_mode_level_metrics_long(mode_level_metrics, path):
    rows = []

    for mode, stat in mode_level_metrics.items():
        for metric_name, stat_dict in stat.items():
            for stat_name, value in stat_dict.items():
                rows.append({
                    "mode": mode,
                    "metric": metric_name,
                    "stat": stat_name,
                    "value": value,
                })

    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "metric", "stat", "value"]
        )
        writer.writeheader()
        writer.writerows(rows)