import math
import simpy
from typing import Dict, List


def should_decouple(train_schedule: Dict, track_capacity: int) -> bool:
    """Return True if the total cars exceed the yard track capacity."""
    total_cars = train_schedule.get("full_cars", 0) + train_schedule.get("empty_cars", 0)
    return total_cars > track_capacity


def decouple_train(train_schedule: Dict, track_capacity: int) -> List[Dict]:
    """
    Split an over-length train into multiple sub-trains based on track capacity.
    Each sub-train is treated as an independent entity for processing.
    """
    train_id = train_schedule["train_id"]
    total_cars = train_schedule["full_cars"] + train_schedule["empty_cars"]
    n_segments = math.ceil(total_cars / track_capacity)

    print(f"\n{'=' * 70}")
    print(f"[DECOUPLE] Train {train_id}: total={total_cars}, capacity={track_capacity}, "
          f"splitting into {n_segments} segments")
    print(f"{'=' * 70}")

    sub_trains = []
    remaining_full = train_schedule["full_cars"]
    remaining_empty = train_schedule["empty_cars"]
    remaining_oc = train_schedule["oc_number"]
    remaining_truck = train_schedule["truck_number"]

    for i in range(n_segments):
        # Dynamic split: allocate cars to each sub-train
        cars_in_segment = min(track_capacity, remaining_full + remaining_empty)
        segment_full = min(remaining_full, cars_in_segment)
        segment_empty = max(0, cars_in_segment - segment_full)

        # Distribute OC and truck evenly across remaining segments
        segment_oc = math.ceil(remaining_oc / (n_segments - i))
        segment_truck = math.ceil(remaining_truck / (n_segments - i))

        sub_train = {
            "master_train_id": train_id,
            "train_id": f"{train_id}-{i + 1}",
            "sub_train_id": f"{train_id}-{i + 1}",
            "segment_index": i,
            "n_segments": n_segments,
            "is_last_segment": (i == n_segments - 1),

            # Inherit timing
            "arrival_time": train_schedule["arrival_time"],
            "departure_time": train_schedule["departure_time"],

            # Car allocation
            "full_cars": segment_full,
            "empty_cars": segment_empty,

            # OC/truck allocation
            "oc_number": segment_oc,
            "truck_number": segment_truck,
        }
        sub_trains.append(sub_train)

        print(f"  Segment {i + 1}/{n_segments}: Full={segment_full}, Empty={segment_empty}, "
              f"OC={segment_oc}, Truck={segment_truck}")

        # Update remaining resources
        remaining_full -= segment_full
        remaining_empty -= segment_empty
        remaining_oc -= segment_oc
        remaining_truck -= segment_truck

    print()  # blank line
    return sub_trains


def handle_decoupled_trains(env: simpy.Environment, terminal, sub_trains: List[Dict], process_sub_train_func) -> simpy.events.Process:
    """
    Launch and coordinate all sub-trains in parallel.
    Wait for all of them to finish, then perform coupling and final departure.
    """
    master_id = sub_trains[0]["master_train_id"]
    n_segments = len(sub_trains)

    print(f"[Time={env.now:6.2f}]    Train {master_id} → starting {n_segments} sub-trains")

    # Record mapping between master and sub-trains
    if not hasattr(terminal, "master_train_mapping"):
        terminal.master_train_mapping = {}
    terminal.master_train_mapping[master_id] = [st["sub_train_id"] for st in sub_trains]

    # Create completion events for each sub-train
    completion_events = []

    for st in sub_trains:
        sub_id = st["sub_train_id"]
        completion_event = env.event()
        completion_events.append(completion_event)

        print(f"[Time={env.now:6.2f}]   ├─ Launching sub-train {sub_id} "
              f"({st['full_cars']} cars)")

        env.process(
            process_sub_train_func(
                env,
                terminal,
                st,
                completion_event=completion_event  # Each sub-train must trigger succeed()
            )
        )

    # Wait for all sub-trains
    print(f"[Time={env.now:6.2f}]    Waiting for all {n_segments} sub-trains to complete...")
    yield env.all_of(completion_events)

    print(f"[Time={env.now:6.2f}]    All sub-trains of Train {master_id} completed.")
    print(f"[Time={env.now:6.2f}]    Train {master_id} departs.")

    # Perform coupling and trigger master departure
    yield from couple_trains(env, terminal, sub_trains)

    # Register master completion event
    if not hasattr(terminal, "master_train_events"):
        terminal.master_train_events = {}
    master_done_event = env.event()
    terminal.master_train_events[master_id] = master_done_event
    master_done_event.succeed()

    print(f"[Time={env.now:6.2f}]   Train {master_id} departure event triggered.")


def couple_trains(env: simpy.Environment,terminal,sub_trains: List[Dict]) -> simpy.events.Process:
    """
    Physically couple sub-trains back into a single master train.
    Includes optional coupling time and track release.
    """
    master_id = sub_trains[0]["master_train_id"]
    n_segments = len(sub_trains)
    couple_time = 1.0  # hr

    print(f"\n[Time={env.now:6.2f}]     [COUPLE] Train {master_id}: reassembling {n_segments} segments")
    yield env.timeout(couple_time)

    # Track release
    if hasattr(terminal, "tracks"):
        for st in sub_trains:
            terminal.tracks.put(st["segment_index"] + 1)
        print(f"[Time={env.now:6.2f}]   Tracks released for Train {master_id}")

    # Compute maximum processing time (if available)
    if hasattr(terminal, "train_times"):
        max_time = max(terminal.train_times.get(st["sub_train_id"], 0) for st in sub_trains)
        terminal.train_times[master_id] = max_time
        print(f"[Time={env.now:6.2f}]   Recorded master train total time: {max_time:.3f} hrs")

    # Trigger master departure event
    if not hasattr(terminal, "train_departed_events"):
        terminal.train_departed_events = {}
    if master_id not in terminal.train_departed_events:
        terminal.train_departed_events[master_id] = env.event()
    if not terminal.train_departed_events[master_id].triggered:
        terminal.train_departed_events[master_id].succeed()

    print(f"[Time={env.now:6.2f}] Train {master_id} DEPARTED (coupled)\n")


def get_train_id(train_schedule: Dict) -> str:
    """Return consistent train_id whether master or sub-train."""
    return train_schedule.get("sub_train_id", train_schedule["train_id"])


def is_sub_train(train_schedule: Dict) -> bool:
    """Return True if the train is a sub-train."""
    return "sub_train_id" in train_schedule or "master_train_id" in train_schedule


def print_timetable_info(train_schedule: Dict, prefix: str = ""):
    """Print train details for debugging."""
    train_id = get_train_id(train_schedule)
    is_sub = is_sub_train(train_schedule)

    print(f"{prefix}Train ID: {train_id}")
    print(f"{prefix}Type: {'Sub-train' if is_sub else 'Master train'}")

    if is_sub:
        print(f"{prefix}Master ID: {train_schedule.get('master_train_id', 'N/A')}")
        print(f"{prefix}Segment: {train_schedule.get('segment_index', 0) + 1}/"
              f"{train_schedule.get('n_segments', 1)}")

    print(f"{prefix}Arrival: {train_schedule.get('arrival_time', 0):.3f} hrs")
    print(f"{prefix}Departure: {train_schedule.get('departure_time', 0):.3f} hrs")
    print(f"{prefix}Full cars: {train_schedule.get('full_cars', 0)}")
    print(f"{prefix}Empty cars: {train_schedule.get('empty_cars', 0)}")
    print(f"{prefix}OC number: {train_schedule.get('oc_number', 0)}")
    print(f"{prefix}Truck number: {train_schedule.get('truck_number', 0)}")