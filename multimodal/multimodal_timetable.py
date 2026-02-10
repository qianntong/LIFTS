import random
import yaml
from pathlib import Path


def calculate_weekly_balancing_params(train_entries, vessel_entries, week_length):
    """
    Capacity-aware weekly truck balancing.

    Principles:
    - Train capacity is a HARD constraint
    - Vessel supply must be fully cleared
    - Truck is the final balancer
    """

    weekly = {}

    def week_index(t):
        return int(t // week_length) if week_length > 0 else 0

    for e in train_entries:
        w = week_index(e["arrival_time"])
        rec = weekly.setdefault(
            w,
            {
                "train_total": 0.0,
                "vessel_total": 0.0,
                "train_to_vessel": 0.0,
                "vessel_to_train": 0.0,
            },
        )

        b = float(e["batch_size"])
        rec["train_total"] += b
        rec["train_to_vessel"] += b * float(
            e["destination_split"].get("vessel", 0.0)
        )

    # Aggregate vessels
    for e in vessel_entries:
        w = week_index(e["arrival_time"])
        rec = weekly.setdefault(
            w,
            {
                "train_total": 0.0,
                "vessel_total": 0.0,
                "train_to_vessel": 0.0,
                "vessel_to_train": 0.0,
            },
        )

        b = float(e["batch_size"])
        rec["vessel_total"] += b
        rec["vessel_to_train"] += b * float(
            e["destination_split"].get("train", 0.0)
        )

    # Derive truck balancing
    for w, rec in weekly.items():

        train_total = rec["train_total"]
        vessel_total = rec["vessel_total"]

        train_to_vessel = rec["train_to_vessel"]
        vessel_to_train = rec["vessel_to_train"]

        # 1. lock train capacity
        train_capacity_used = vessel_to_train
        train_capacity_left = max(0.0, train_total - train_capacity_used)

        # 2. train supply to vessel
        train_supply_to_vessel = min(train_to_vessel, vessel_total)

        # 3. vessel remainder → truck
        truck_to_vessel = max(0.0, vessel_total - train_supply_to_vessel)

        # 4. truck to train (capacity-limited)
        vessel_to_truck_demand = max(0.0, vessel_total - vessel_to_train)
        truck_to_train = min(train_capacity_left, vessel_to_truck_demand)

        truck_total = int(round(truck_to_vessel + truck_to_train))

        if truck_total <= 0:
            rec.update(
                {
                    "truck_total": 0,
                    "truck_to_vessel": 0,
                    "truck_to_train": 0,
                    "truck_split": {},
                    "truck_headway": None,
                }
            )
            continue

        p_v = truck_to_vessel / truck_total
        p_t = truck_to_train / truck_total

        split = {}
        if p_v > 0:
            split["vessel"] = p_v
        if p_t > 0:
            split["train"] = p_t

        rec.update(
            {
                "truck_total": truck_total,
                "truck_to_vessel": int(round(truck_to_vessel)),
                "truck_to_train": int(round(truck_to_train)),
                "truck_split": split,
                "truck_headway": week_length / truck_total,
            }
        )

    return weekly



def generate_timetable(config, verbose=True):
    """
    Generate multimodal timetable.

    Parameters
    ----------
    config : dict
        Parsed YAML config
    verbose : bool
        Whether to print detailed logs (default True)
    """

    def log(msg=""):
        if verbose:
            print(msg)

    sim_length = float(config["simulation"]["length"])
    timetable_cfg = config["timetable"]

    WEEK_LENGTH = 168.0
    max_week = int(sim_length // WEEK_LENGTH)

    arrival_counter = {"vessel": 0, "train": 0, "truck": 0}
    timetable = []

    vessel_entries = []
    train_entries = []

    # def sample_week_times(start, end, n):
    #     ts = [random.uniform(start, end) for _ in range(n)]
    #     ts.sort()
    #     return ts

    def sample_week_times(start, end, n):
        if n <= 0:
            return []
        step = (end - start) / n
        return [start + (i + 0.5) * step for i in range(n)]

    log("\n================ TIMETABLE ================")
    log(f"Simulation length = {sim_length:.1f} h  ({max_week} weeks)\n")

    tmp_vessel = [{
        "arrival_time": 0.0,
        "batch_size": timetable_cfg["vessel"]["batch_size"],
        "destination_split": timetable_cfg["vessel"]["destination_split"],
    }]

    tmp_train = [
        {
            "arrival_time": 0.0,
            "batch_size": timetable_cfg["train"]["batch_size"],
            "destination_split": timetable_cfg["train"]["destination_split"],
        }
        for _ in range(int(timetable_cfg["train"]["weekly_num"]))
    ]

    baseline = calculate_weekly_balancing_params(
        tmp_train, tmp_vessel, WEEK_LENGTH
    )[0]

    log("[INPUT]")
    log(
        f"  Vessel: weekly_num={timetable_cfg['vessel']['weekly_num']}, "
        f"batch_size={timetable_cfg['vessel']['batch_size']}, "
        f"split={timetable_cfg['vessel']['destination_split']}"
    )
    log(
        f"  Train : weekly_num={timetable_cfg['train']['weekly_num']}, "
        f"batch_size={timetable_cfg['train']['batch_size']}, "
        f"split={timetable_cfg['train']['destination_split']}"
    )
    log(
        f"  Truck : weekly_num={baseline['truck_total']}, "
        f"batch_size={timetable_cfg['truck']['batch_size']}, "
        f"split={baseline['truck_split']}\n"
    )

    # Weekly loop
    for w in range(max_week):
        week_start = w * WEEK_LENGTH
        week_end = (w + 1) * WEEK_LENGTH

        log(f"[Week {w}] ({week_start:.0f} – {week_end:.0f} h)")

        # Vessel arrivals
        v_cfg = timetable_cfg["vessel"]
        v_times = sample_week_times(
            week_start, week_end, int(v_cfg["weekly_num"])
        )

        for t in v_times:
            arrival_counter["vessel"] += 1
            e = {
                "mode": "vessel",
                "arrival_id": arrival_counter["vessel"],
                "arrival_time": round(t, 4),
                "batch_size": int(v_cfg["batch_size"]),
                "destination_split": dict(v_cfg["destination_split"]),
                "week": w,
            }
            vessel_entries.append(e)
            timetable.append(e)

        # Train arrivals
        t_cfg = timetable_cfg["train"]
        t_times = sample_week_times(
            week_start, week_end, int(t_cfg["weekly_num"])
        )

        for t in t_times:
            arrival_counter["train"] += 1
            e = {
                "mode": "train",
                "arrival_id": arrival_counter["train"],
                "arrival_time": round(t, 4),
                "batch_size": int(t_cfg["batch_size"]),
                "destination_split": dict(t_cfg["destination_split"]),
                "week": w,
            }
            train_entries.append(e)
            timetable.append(e)

        # ------------------
        # Truck arrivals (balancing)
        # ------------------
        weekly_balancing = calculate_weekly_balancing_params(
            train_entries, vessel_entries, WEEK_LENGTH
        )
        rec = weekly_balancing.get(w, {})

        truck_total = int(rec.get("truck_total", 0))
        truck_split = rec.get("truck_split", {})

        log(f"  Vessel arrivals = {int(v_cfg['weekly_num'])}")
        log(f"  Train  arrivals = {int(t_cfg['weekly_num'])}")
        log(f"  Truck  arrivals = {truck_total}")

        if truck_total > 0:
            log(f"    truck_split = {truck_split}")

        truck_times = sample_week_times(week_start, week_end, truck_total)
        for t in truck_times:
            arrival_counter["truck"] += 1
            timetable.append(
                {
                    "mode": "truck",
                    "arrival_id": arrival_counter["truck"],
                    "arrival_time": round(t, 4),
                    "batch_size": int(
                        timetable_cfg["truck"].get("batch_size", 1)
                    ),
                    "destination_split": truck_split,
                    "week": w,
                }
            )

        log("")

    timetable.sort(key=lambda x: x["arrival_time"])

    # Final summary
    log("================ FINAL ARRIVAL COUNTS ================")
    log(f"  Vessel arrivals total = {arrival_counter['vessel']}")
    log(f"  Train  arrivals total = {arrival_counter['train']}")
    log(f"  Truck  arrivals total = {arrival_counter['truck']}")
    log("=====================================================\n")

    return timetable


from collections import defaultdict

def export_mode_timetable_txt(
    timetable,
    mode,
    filepath,
    max_rows_per_week=None,
):
    """
    Export per-week timetable for a given mode (train / vessel / truck).

    Parameters
    ----------
    timetable : list[dict]
        Generated timetable
    mode : str
        'train', 'vessel', or 'truck'
    filepath : str
        Output txt path
    max_rows_per_week : int or None
        Limit number of rows per week (useful for truck)
        None = no limit
    """

    events_by_week = defaultdict(list)

    # -----------------------
    # Group by week
    # -----------------------
    for e in timetable:
        if e["mode"] == mode:
            events_by_week[e["week"]].append(e)

    with open(filepath, "w") as f:

        for w in sorted(events_by_week.keys()):
            f.write(f"[Week {w}]\n")

            # sort by arrival time
            events = sorted(
                events_by_week[w],
                key=lambda x: x["arrival_time"]
            )

            for i, ev in enumerate(events):

                if max_rows_per_week is not None and i >= max_rows_per_week:
                    remaining = len(events) - max_rows_per_week
                    f.write(
                        f"  ... ({remaining} more {mode} arrivals omitted)\n"
                    )
                    break

                label = mode.capitalize()
                line = (
                    f"  {label} {ev['arrival_id']} | "
                    f"time = {ev['arrival_time']:.2f} h | "
                    f"batch = {ev['batch_size']}"
                )

                # destination split (useful esp. for truck)
                if "destination_split" in ev:
                    line += f" | split = {ev['destination_split']}"

                f.write(line + "\n")

            f.write("\n")


# if __name__ == "__main__":
#
#     random.seed(42)
#
#     config_path = Path("input/config.yaml")
#
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#
#     timetable = generate_timetable(config, verbose=True)
#
#     for mode in ["train", "vessel", "truck"]:
#         export_mode_timetable_txt(
#             timetable,
#             mode=f"{mode}",
#             filepath=f"output/ContainerLog/{mode}_timetable.txt")
