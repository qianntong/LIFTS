import math
from collections import defaultdict
from multimodal_simulation import *

def calculate_weekly_balancing_params(vessel_batch, p_v_to_t, train_batch_size):
    """
    Symmetric assumption:
      T->V = V->T
      H->V = V->H
    Train capacity hard constraint (integer train_batch_size)

    Truck arrivals definition (your convention):
      weekly_trucks = H->V + H->T
    """

    V = int(vessel_batch)

    # Vessel unload
    V_to_T = int(round(V * p_v_to_t))
    V_to_H = V - V_to_T

    # Train capacity
    n_trains = int(math.ceil(V_to_T / train_batch_size))
    train_total = n_trains * train_batch_size

    # Symmetric vessel load
    T_to_V = V_to_T
    H_to_V = V_to_H

    # Train residuals
    H_to_T = train_total - V_to_T
    T_to_H = train_total - T_to_V  # equals H_to_T under symmetry

    # Truck arrivals (ONLY inbound trucks)
    truck_total = int(H_to_V + H_to_T)

    return {
        "V": V,
        "V_to_T": V_to_T,
        "V_to_H": V_to_H,
        "T_to_V": T_to_V,
        "H_to_V": H_to_V,
        "H_to_T": H_to_T,
        "T_to_H": T_to_H,
        "train_trips": n_trains,
        "train_total": train_total,
        "truck_total": truck_total,
    }


# def _sample_week_times(start, end, n):
#     if n <= 0:
#         return []
#     step = (end - start) / n
#     return [start + (i + 0.5) * step for i in range(n)]


def _sample_week_times(start, end, n):
    if n <= 0:
        return []
    step = (end - start) / n
    return [start + i * step for i in range(n)]

def _truck_week_times(start, n):
    if n <= 0:
        return []
    step = 0
    return [start + i * step for i in range(n)]


def generate_timetable(config, terminal, verbose=None):
    """
    Return: list[dict] timetable entries, each entry includes:
      - mode, arrival_id, arrival_time, batch_size, destination_split, week
    """
    def log(msg=""):
        if verbose:
            print(msg)

    sim_length = float(config["simulation"]["length"])
    tt = config["timetable"]

    vessel_cfg = tt["vessel"]
    train_cfg = tt["train"]
    truck_cfg = tt.get("truck", {"batch_size": 1})

    # Inputs
    vessel_weekly_num = int(vessel_cfg.get("weekly_num", 1))
    vessel_batch = int(vessel_cfg["batch_size"])
    p_v_to_t = float(vessel_cfg["destination_split"]["train"])
    train_batch_size = int(train_cfg["batch_size"])
    WEEK_LENGTH = 168.0

    # written as matrix header
    weekly_summary = {
        "train_weekly_num": 0,
        "train_batch_size": train_batch_size,
        "truck_weekly_volume": 0,
    }

    max_week = int(sim_length // WEEK_LENGTH)

    timetable = []
    arrival_counter = {"vessel": 0, "train": 0, "truck": 0}

    log("\n================ TIMETABLE ================")
    log(f"Simulation length = {sim_length:.1f} h ({max_week} weeks)\n")

    for w in range(max_week):
        week_start = w * WEEK_LENGTH
        week_end = (w + 1) * WEEK_LENGTH

        # ---- weekly flow computation (one-week volumes) ----
        flows = calculate_weekly_balancing_params(
            vessel_batch=vessel_batch,
            p_v_to_t=p_v_to_t,
            train_batch_size=train_batch_size,
        )
        weekly_summary["train_weekly_num"] = flows["train_trips"]
        weekly_summary["truck_weekly_volume"] = flows["truck_total"]
        terminal.track_number = flows["train_trips"]

        # ---- derive endogenous splits for train/truck ----
        # Train inbound split (train -> vessel / truck)
        if flows["train_total"] > 0:
            train_split = {
                "vessel": flows["T_to_V"] / flows["train_total"],
                "truck":  flows["T_to_H"] / flows["train_total"],
            }
        else:
            train_split = {}

        # Truck inbound split (truck -> vessel / train)
        if flows["truck_total"] > 0:
            truck_split = {
                "vessel": flows["H_to_V"] / flows["truck_total"],
                "train":  flows["H_to_T"] / flows["truck_total"],
            }
        else:
            truck_split = {}

        log(f"[Week {w}] ({week_start:.0f} – {week_end:.0f} h)")
        log(f"  flows: V->T={flows['V_to_T']}, V->H={flows['V_to_H']}, "
            f"H->T={flows['H_to_T']}, H->V={flows['H_to_V']}")
        log(f"  train_trips={flows['train_trips']}, truck_total={flows['truck_total']}")
        log(f"  train_split={train_split}")
        log(f"  truck_split={truck_split}\n")

        # ---- Vessel arrivals ----
        v_times = _sample_week_times(week_start, week_end, vessel_weekly_num)
        for t in v_times:
            arrival_counter["vessel"] += 1
            timetable.append({
                "mode": "vessel",
                "arrival_id": arrival_counter["vessel"],
                "arrival_time": round(t, 4),
                "batch_size": vessel_batch,
                "destination_split": dict(vessel_cfg["destination_split"]),
                "week": w,
            })

        # ---- Train arrivals ----
        t_times = _sample_week_times(week_start, week_end, flows["train_trips"])
        for t in t_times:
            arrival_counter["train"] += 1
            timetable.append({
                "mode": "train",
                "arrival_id": arrival_counter["train"],
                "arrival_time": round(t, 4),
                "batch_size": train_batch_size,
                "destination_split": dict(train_split),
                "week": w,
            })

        # ---- Truck arrivals----
        truck_times = _truck_week_times(week_start, flows["truck_total"])
        for t in truck_times:
            arrival_counter["truck"] += 1
            timetable.append({
                "mode": "truck",
                "arrival_id": arrival_counter["truck"],
                "arrival_time": round(t, 4),
                "batch_size": int(truck_cfg.get("batch_size", 1)),
                "destination_split": dict(truck_split),
                "week": w,
            })

    timetable.sort(key=lambda x: x["arrival_time"])

    log("================ FINAL ARRIVAL COUNTS ================")
    log(f"  Vessel arrivals total = {arrival_counter['vessel']}")
    log(f"  Train  arrivals total = {arrival_counter['train']}")
    log(f"  Truck  arrivals total = {arrival_counter['truck']}")
    log("=====================================================\n")

    return timetable, weekly_summary


def export_mode_timetable_txt(timetable, mode, filepath, max_rows_per_week=None):
    events_by_week = defaultdict(list)
    for e in timetable:
        if e["mode"] == mode:
            events_by_week[e["week"]].append(e)

    with open(filepath, "w") as f:
        for w in sorted(events_by_week.keys()):
            f.write(f"[Week {w}]\n")
            events = sorted(events_by_week[w], key=lambda x: x["arrival_time"])

            for i, ev in enumerate(events):
                if max_rows_per_week is not None and i >= max_rows_per_week:
                    remaining = len(events) - max_rows_per_week
                    f.write(f"  ... ({remaining} more {mode} arrivals omitted)\n")
                    break

                line = (
                    f"  {mode.capitalize()} {ev['arrival_id']} | "
                    f"time={ev['arrival_time']:.2f} h | "
                    f"batch={ev['batch_size']} | "
                    f"split={ev.get('destination_split', {})}"
                )
                f.write(line + "\n")
            f.write("\n")


# # Test
# import math
# from collections import defaultdict, Counter
#
# class DummyTerminal:
#     def __init__(self, track_number):
#         self.track_number = track_number
#
#
# if __name__ == "__main__":
#
#     config = {
#         "simulation": {
#             "length": 168 * 2  # 2 weeks
#         },
#         "terminal": {
#             "track_number": 4
#         },
#         "timetable": {
#             "vessel": {
#                 "weekly_num": 1,
#                 "batch_size": 1000,
#                 "destination_split": {
#                     "train": 0.5,
#                     "truck": 0.5
#                 }
#             },
#             "train": {
#                 "batch_size": 100
#             },
#             "truck": {
#                 "batch_size": 1
#             }
#         }
#     }
#
#     terminal = DummyTerminal(
#         track_number=config["terminal"]["track_number"]
#     )
#
#     timetable, weekly_summary = generate_timetable(
#         config=config,
#         terminal=terminal,
#         verbose=True
#     )
#
#     print("\n========== TERMINAL INFO ==========")
#     print("Track number:", terminal.track_number)
#
#     print("\n========== WEEKLY SUMMARY ==========")
#     print(weekly_summary)
#
#     print("\n========== MODE COUNTS ==========")
#     mode_count = Counter([e["mode"] for e in timetable])
#     print(mode_count)
#
#     print("\n========== FIRST 10 EVENTS ==========")
#     for e in timetable[:10]:
#         print(e)
#
#     print("\nTotal events:", len(timetable))