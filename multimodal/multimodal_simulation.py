from dataclasses import dataclass
from typing import Optional
import simpy
import csv
import os
import time
from multimodal_timetable import *

container_events = {}


@dataclass
class Container:
    id: int
    origin_mode: str          # 'vessel', 'train', or 'truck'
    origin_id: int
    destination_mode: str     # 'vessel', 'train', or 'truck'
    destination_id: Optional[int] = None   # later assigned

    def __str__(self) -> str:
        return f"C-{self.id}-{self.origin_mode}-{self.origin_id}-{self.destination_mode}-{self.destination_id}"


@dataclass
class Crane:
    id: int
    location: str              # 'trackside', 'berthside'
    parent_id: int = None      # track_id or berth_id
    crane_type: str = 'Diesel'

    def __str__(self):
        return f"Crane-{self.location}-{self.parent_id}-{self.id}"


@dataclass
class Truck:
    id: int = 0
    truck_type: str = 'Diesel'

    def __str__(self) -> str:
        return f"Truck-{self.id}-{self.truck_type}"


@dataclass
class Hostler:
    type: str = 'Diesel'
    id: int = 0

    def to_string(self) -> str:
        return f'{self.id}-{self.type}'


class Terminal:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        yard_cfg = config["yard"]
        term_cfg = config["terminal"]
        gate_cfg = config["gates"]
        simul_cfg = config["simulation"]

        self.analyze_start = simul_cfg["analyze_start"]
        self.analyze_end = simul_cfg["analyze_end"]

        self.yard_type = yard_cfg["yard_type"]
        self.receiving_track_numbers = int(yard_cfg["receiving_track_numbers"])
        self.railcar_length = float(yard_cfg["railcar_length"])
        self.d_f = float(yard_cfg["d_f"])
        self.d_x = float(yard_cfg["d_x"])

        self.hostler_number = int(term_cfg["hostler_number"])
        self.hostler_diesel_percentage = float(term_cfg["hostler_diesel_percentage"])
        self.in_gate_numbers = int(gate_cfg["in_gate_numbers"])
        self.out_gate_numbers = int(gate_cfg["out_gate_numbers"])

        # tracks cranes
        self.track_number = int(yard_cfg["track_number"])
        self.tracks = simpy.Store(env, capacity=self.track_number)
        for track_id in range(1, self.track_number + 1):
            self.tracks.put(track_id)

        self.cranes_per_track = int(term_cfg["cranes_per_track"])
        self.cranes_by_track = {
            track_id: simpy.Store(env, capacity=self.cranes_per_track)
            for track_id in range(1, self.track_number + 1)
        }

        for track_id in range(1, self.track_number + 1):
            for crane_id in range(1, self.cranes_per_track + 1):
                self.cranes_by_track[track_id].put(
                    Crane(
                        id=crane_id,
                        location="Trackside",
                        parent_id=track_id,
                        crane_type="Hybrid"
                    )
                )

        # dock berths and cranes
        self.berth_number = int(yard_cfg["berth_number"])
        self.berths = simpy.Store(env, capacity=self.berth_number)

        for berth_id in range(1, self.berth_number + 1):
            self.berths.put(berth_id)

        self.cranes_per_berth = int(term_cfg["cranes_per_berth"])
        self.cranes_by_berth = {
            berth_id: simpy.Store(env, capacity=self.cranes_per_berth)
            for berth_id in range(1, self.berth_number + 1)
        }

        for berth_id in range(1, self.berth_number + 1):
            for crane_id in range(1, self.cranes_per_berth + 1):
                self.cranes_by_berth[berth_id].put(
                    Crane(
                        id=crane_id,
                        location="Dock",
                        parent_id=berth_id,
                        crane_type="Hybrid"
                    )
                )

        hostler_total = self.hostler_number
        hostler_diesel = round(hostler_total * self.hostler_diesel_percentage)
        hostler_electric = hostler_total - hostler_diesel

        self.hostler_pool = simpy.Store(env, capacity=hostler_total)

        hostlers = (
            [Hostler(id=i, type="Diesel") for i in range(hostler_diesel)] +
            [Hostler(id=i + hostler_diesel, type="Electric") for i in range(hostler_electric)]
        )

        for h in hostlers:
            self.hostler_pool.put(h)

        self.truck_pool = simpy.Store(env, capacity=10**6)

        # Single physical stack (FIFO). Holds both prestaged inventory and newly unloaded containers.
        # NOTE: We deliberately use simpy.Store (not FilterStore) to preserve FIFO order under conditional picks.
        self.container_stack = simpy.FilterStore(env, capacity=10**6)

        self.trackside_chassis = simpy.FilterStore(env, capacity=10**6)
        self.berthside_chassis = simpy.FilterStore(env, capacity=10**6)

        self.in_gates = simpy.Resource(env, capacity=self.in_gate_numbers)
        self.out_gates = simpy.Resource(env, capacity=self.out_gate_numbers)

        self.VEHICLE_TRAVEL_TIME = 1 / 600
        self.CONTAINERS_PER_CRANE_MOVE_MEAN = 2 / 60
        self.CRANE_MOVE_DEV_TIME = 1 / 3600
        self.TRUCK_DIESEL_PERCENTAGE = 1
        self.TRUCK_ARRIVAL_MEAN = 2 / 60
        self.TRUCK_INGATE_TIME = 2 / 60
        self.TRUCK_OUTGATE_TIME = 2 / 60
        self.TRUCK_INGATE_TIME_DEV = 2 / 60
        self.TRUCK_OUTGATE_TIME_DEV = 2 / 60


class ArrivalContext:
    def __init__(self, env, mode: str, arrival_id: int, batch_size):
        """
        mode: 'train' or 'vessel'
        arrival_id: train_id or vessel_id
        """
        self.env = env
        self.mode = mode
        self.id = arrival_id
        self.batch_size = batch_size
        self.loaded_containers = []

        self.arrived = env.event()        # arrival happens
        self.ic_unloaded = env.event()    # all inbound containers unloaded to chassis
        self.ic_cleared = env.event()     # all inbound containers moved to stack
        self.oc_ready = env.event()       # outbound containers prepared at chassis
        self.oc_loaded = env.event()      # all outbound containers loaded
        self.departed = env.event()       # departure completed

    def is_inbound(self, container) -> bool:
        return container.origin_mode == self.mode and container.origin_id == self.id

    def is_outbound(self, container) -> bool:
        return (container.destination_mode == self.mode
                and container.destination_id == self.id)

    def __str__(self):
        return f"{self.mode.capitalize()}-{self.id}"


class TruckContext:
    def __init__(self, env, truck_id: int):
        self.env = env
        self.mode = 'truck'
        self.id = truck_id

        self.arrived = env.event()        # arrives at in-gate
        self.ic_dropped = env.event()     # ic dropped to stack
        self.oc_picked = env.event()      # oc picked from stack
        self.departed = env.event()       # leaves terminal

    def is_inbound(self, container) -> bool:
        return container.origin_mode == 'truck'

    def is_outbound(self, container) -> bool:
        return container.destination_mode == 'truck'

    def __str__(self):
        return f"Truck-{self.id}"


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def record_event(container, event, time, extra=None):
    key = (
        container.origin_mode,
        container.origin_id,
        container.destination_mode,
        container.destination_id,
        container.id
    )

    if key not in container_events:
        container_events[key] = {
            "origin_mode": container.origin_mode,
            "origin_id": container.origin_id,
            "destination_mode": container.destination_mode,
            "destination_id": container.destination_id,
            "index": container.id,
            "timeline": {}
        }

    container_events[key]["timeline"][event] = {
        "time": time,
        **(extra or {})
    }


def normalize_container_events(container_events):
    """
    Normalize container event records by consolidating lifecycle updates.

    Container identity is defined by:
        (origin_mode, origin_id, destination_mode, container_id)

    destination_id is treated as a lifecycle attribute and may be updated
    over time without creating a new container record.
    """
    normalized = {}

    for key, info in container_events.items():
        origin_mode, origin_id, dest_mode, dest_id, index = key

        norm_key = (origin_mode, origin_id, dest_mode, index)

        if norm_key not in normalized:
            normalized[norm_key] = {
                "origin_mode": origin_mode,
                "origin_id": origin_id,
                "destination_mode": dest_mode,
                "destination_id": None,
                "container_id": index,
                "timeline": {}
            }

        if dest_id is not None:
            normalized[norm_key]["destination_id"] = dest_id

        # Merge timelines (keep latest timestamp per event if collision)
        for event, ev_info in info.get("timeline", {}).items():
            if event not in normalized[norm_key]["timeline"]:
                normalized[norm_key]["timeline"][event] = ev_info
            else:
                old_t = normalized[norm_key]["timeline"][event].get("time")
                new_t = ev_info.get("time")
                if old_t is None or (new_t is not None and new_t > old_t):
                    normalized[norm_key]["timeline"][event] = ev_info

    return normalized


def generate_containers(arrival_entry):
    containers = []
    mode = arrival_entry["mode"]
    arrival_id = arrival_entry["arrival_id"]
    batch_size = arrival_entry["batch_size"]
    destination_split = arrival_entry["destination_split"]

    destinations = list(destination_split.keys())
    weights = list(destination_split.values())
    for i in range(batch_size):
        dest_mode = random.choices(destinations, weights=weights, k=1)[0]
        c = Container(
            id=i,
            origin_mode=mode,
            origin_id=arrival_id,
            destination_mode=dest_mode,
            destination_id=None
        )
        containers.append(c)
    return containers


# def prestage_containers_at_t0(env, terminal, config):
#     """At T=0, inject one-week vessel container batch into the stack.
#
#     This implements the "pre-stage" rule:
#     all containers associated with one-week vessels appear in the stack at time 0.
#     We create containers using the vessel destination_split and mark:
#       - origin_mode='vessel', origin_id start from 0
#       - destination_mode in {'train','truck',...} per vessel split
#       - destination_id=None (assigned later when the receiving mode actually picks it)
#     """
#     vessel_cfg = config.get("timetable", {}).get("vessel", {})
#     if not vessel_cfg.get("enabled", True):
#         return
#
#     batch_size = int(vessel_cfg.get("batch_size", 0))
#     destination_split = vessel_cfg.get("destination_split", {})
#     if batch_size <= 0 or not destination_split:
#         return
#
#     yield env.timeout(0)
#
#     destinations = list(destination_split.keys())
#     weights = list(destination_split.values())
#
#     for i in range(batch_size):
#         dest_mode = random.choices(destinations, weights=weights, k=1)[0]
#         c = Container(
#             id=i,
#             origin_mode="vessel",
#             origin_id=0,
#             destination_mode=dest_mode,
#             destination_id=None
#         )
#         yield terminal.container_stack.put(c)
#         record_event(c, "arrival_expected", 0.0)
#         record_event(c, "arrival_actual", 0.0)


def check_oc_ready(ctx, chassis_store):
    count = sum(1 for c in chassis_store.items if ctx.is_outbound(c))
    return count == ctx.batch_size


def stack_get_first_matching(env, stack, predicate, poll_interval=0.001):
    """FIFO conditional pop from a simpy.Store.

    simpy.FilterStore does not guarantee FIFO under filters. This helper preserves FIFO by scanning
    stack.items in order and removing the first match. If nothing matches, it waits briefly and retries.
    """
    while True:
        for i, item in enumerate(list(stack.items)):
            if predicate(item):
                stack.items.pop(i)
                return item
        yield env.timeout(poll_interval)


def hostler_ic_oc_truck_process(env, terminal, chassis_store, ctx, ic):
    """
    One complete IC -> OC cycle handled by a single hostler
    """
    # 1. get hostler
    hostler = yield terminal.hostler_pool.get()

    # 2. travel to chassis and pick IC
    park_chassis_travel_time = 0.04
    yield env.timeout(park_chassis_travel_time)
    yield chassis_store.get(lambda c: c == ic)
    record_event(ic, "hostler_IC_pick_up", env.now)

    # 3. move IC to stack
    chassis_stack_travel_time = 0.04
    yield env.timeout(chassis_stack_travel_time)
    record_event(ic, "hostler_IC_drop_off", env.now)

    # 4. IC dropped and picked up by trucks
    if ic.destination_mode == "truck":
        truck = yield terminal.truck_pool.get()
        ic.destination_id = truck.id
        travel_time_to_gate = 0.04
        yield env.timeout(travel_time_to_gate)
        record_event(ic, "departure", env.now)
        yield terminal.truck_pool.put(truck)
    # IC dropped to container stack, and about to be OC when train/vessel arrives
    else:
        yield terminal.container_stack.put(ic)

    # # 5. find OC (FIFO among matches) and bind destination
    oc = yield terminal.container_stack.get(lambda c: (c.destination_mode == ic.origin_mode and c.destination_id is None))
    oc.destination_id = ic.origin_id
    record_event(oc, "hostler_OC_pick_up", env.now)

    # matching_ocs = [ c for c in terminal.container_stack.items
    #     if c.destination_mode == ic.origin_mode and c.destination_id is None]
    #
    # if matching_ocs:
    #     oc = yield terminal.container_stack.get(lambda c: (c.destination_mode == ic.origin_mode and c.destination_id is None))
    #     oc.destination_id = ic.origin_id
    #     record_event(oc, "hostler_OC_pick_up", env.now)
    # else:
    #     # no OC available for this mode → do nothing here
    #     # truck-demand MUST be handled by truck arrival, not hostler
    #     return

    # 6. move OC back to chassis
    yield env.timeout(chassis_stack_travel_time)
    yield chassis_store.put(oc)
    record_event(oc, "hostler_OC_drop_off", env.now)

    # 7. release hostler
    yield terminal.hostler_pool.put(hostler)

    # 8. IC cleared check (arrival-level)
    if not any(ctx.is_inbound(c) for c in chassis_store.items):
        if not ctx.ic_cleared.triggered:
            ctx.ic_cleared.succeed()
            print(f"[{env.now:.2f}] All IC cleared from chassis for {ctx}")

    # 9. OC ready check
    if not ctx.oc_ready.triggered and check_oc_ready(ctx, chassis_store):
        ctx.oc_ready.succeed()
        print(f"[{env.now:.2f}] {ctx} OC ready on chassis")


def crane_unload_inbound_process(env, terminal, mode, ctx, inbound_containers):
    """
    Generator process: start crane unload for inbound containers
    """
    resource_type, resource_id = ctx.assigned_resource

    if resource_type == "track":
        crane_pool = terminal.cranes_by_track[resource_id]
        chassis_store = terminal.trackside_chassis
    elif resource_type == "berth":
        crane_pool = terminal.cranes_by_berth[resource_id]
        chassis_store = terminal.berthside_chassis
    else:
        raise ValueError("Unknown resource type")

    remaining = len(inbound_containers)

    def unload_one(container):
        nonlocal remaining

        crane = yield crane_pool.get()

        unload_time = 0.02
        yield env.timeout(unload_time)
        record_event(container, "chassis_IC_unload", env.now)

        yield chassis_store.put(container)
        yield crane_pool.put(crane)

        remaining -= 1
        if remaining == 0 and not ctx.ic_unloaded.triggered:
            ctx.ic_unloaded.succeed()
            print(f"[{env.now:.2f}] {ctx} IC unloaded to chassis")

    for c in inbound_containers:
        env.process(unload_one(c))

    return chassis_store


def crane_load_outbound_process(env, terminal, ctx, chassis_store):
    """
    Generator process: load outbound containers from chassis to train/vessel using cranes.
    This MUST be a generator if you want to call it via env.process(...).
    """
    # Make it a real process + ensure OC is ready before loading
    yield ctx.oc_ready

    resource_type, resource_id = ctx.assigned_resource
    if resource_type == "track":
        crane_pool = terminal.cranes_by_track[resource_id]
    elif resource_type == "berth":
        crane_pool = terminal.cranes_by_berth[resource_id]
    else:
        raise ValueError("Unknown resource type")

    def load_one():
        crane = yield crane_pool.get()
        try:
            # Wait until one outbound exists on chassis
            while True:
                outbound = None
                for c in chassis_store.items:
                    if ctx.is_outbound(c):
                        outbound = c
                        break
                if outbound is not None:
                    break
                yield env.timeout(0.0001)

            # Remove it from chassis
            yield chassis_store.get(lambda c: c == outbound)

            # Load time
            load_time = 0.02
            yield env.timeout(load_time)

            ctx.loaded_containers.append(outbound)
            record_event(outbound, "chassis_OC_load", env.now)
        finally:
            yield crane_pool.put(crane)

    # Launch parallel crane moves
    procs = [env.process(load_one()) for _ in range(ctx.batch_size)]
    yield simpy.events.AllOf(env, procs)

    if not ctx.oc_loaded.triggered:
        ctx.oc_loaded.succeed()
        print(f"[{env.now:.2f}] {ctx} OC loaded")


def train_vessel_arrival_process(env, terminal, arrival_entry):
    mode = arrival_entry["mode"]
    arrival_id = arrival_entry["arrival_id"]
    batch_size = arrival_entry["batch_size"]

    ctx = ArrivalContext(env, mode, arrival_id, batch_size)
    yield env.timeout(arrival_entry["arrival_time"] - env.now)

    # assign track/berth
    if mode == "train":
        track_id = yield terminal.tracks.get()
        ctx.assigned_resource = ("track", track_id)
    elif mode == "vessel":
        berth_id = yield terminal.berths.get()
        ctx.assigned_resource = ("berth", berth_id)
    else:
        raise ValueError("Unknown mode in arrival process")

    ctx.arrived.succeed()
    print(f"[{env.now:.2f}] {ctx} ARRIVED")

    inbound_containers = generate_containers(arrival_entry)
    for c in inbound_containers:
        record_event(c, "arrival_actual", env.now)
        record_event(c, "arrival_expected", arrival_entry["arrival_time"])

    # crane unload to chassis
    chassis_store = crane_unload_inbound_process(env, terminal, mode, ctx, inbound_containers)
    yield ctx.ic_unloaded

    # all OC ready: hostlers move IC -> stack and swap in OC -> chassis
    for ic in list(inbound_containers):
        env.process(hostler_ic_oc_truck_process(env, terminal, chassis_store, ctx, ic))
    yield ctx.ic_cleared
    yield ctx.oc_ready
    env.process(crane_load_outbound_process(env, terminal, ctx, chassis_store))

    # depart
    env.process(train_vessel_departure_process(env, terminal, ctx))

    return ctx


def truck_arrival_process(env, terminal, arrival_entry):
    arrival_id = arrival_entry["arrival_id"]
    truck_ctx = TruckContext(env, arrival_id)
    yield env.timeout(arrival_entry["arrival_time"] - env.now)

    truck = Truck(id=arrival_id)
    truck_ctx.truck = truck
    truck_ctx.arrived.succeed()

    container = generate_containers(arrival_entry)[0]
    with terminal.in_gates.request() as req:
        yield req
        yield env.timeout(terminal.TRUCK_INGATE_TIME)

    yield terminal.container_stack.put(container)
    record_event(container, "arrival_actual", env.now)
    record_event(container, "arrival_expected", arrival_entry["arrival_time"])
    yield terminal.truck_pool.put(truck)
    truck_ctx.ic_dropped.succeed()

    return truck_ctx


def train_vessel_departure_process(env, terminal, ctx):
    yield ctx.oc_loaded

    ctx.departed.succeed()
    print(f"[{env.now:.2f}] {ctx} DEPARTED")

    for c in ctx.loaded_containers:
        record_event(c, "departure", env.now)

    resource_type, resource_id = ctx.assigned_resource
    if resource_type == "track":
        yield terminal.tracks.put(resource_id)
    elif resource_type == "berth":
        yield terminal.berths.put(resource_id)


def export_container_events_to_three_csvs(filepath, prefix):
    """
    Export container events to three CSVs (train / vessel / truck),
    AND return standardized per-container metrics for downstream statistics.
    """
    normalized_events = normalize_container_events(container_events)

    all_events = sorted({
        event
        for info in normalized_events.values()
        for event in info["timeline"].keys()
    })

    base_columns = [
        "origin_mode",
        "origin_id",
        "destination_mode",
        "destination_id",
        "container_id"
    ]

    header = base_columns + all_events

    files = {
        "train": open(os.path.join(filepath, f"{prefix}_train.csv"), "w", newline=""),
        "vessel": open(os.path.join(filepath, f"{prefix}_vessel.csv"), "w", newline=""),
        "truck": open(os.path.join(filepath, f"{prefix}_truck.csv"), "w", newline="")
    }

    writers = {
        mode: csv.writer(f)
        for mode, f in files.items()
    }

    for writer in writers.values():
        writer.writerow(header)

    container_metrics = {}

    for key, info in normalized_events.items():
        origin_mode, origin_id, dest_mode, index = key

        # Only export OD-complete containers for downstream statistics
        if origin_mode is None or dest_mode is None or origin_mode == dest_mode:
            continue

        if origin_mode not in writers:
            continue

        destination_id = info.get("destination_id")
        timeline = info["timeline"]

        row = [
            origin_mode,
            origin_id,
            dest_mode,
            destination_id,
            index
        ]

        for event in all_events:
            row.append(
                timeline[event]["time"] if event in timeline else ""
            )

        writers[origin_mode].writerow(row)

        arrival_actual = None
        arrival_expected = None
        departure = None

        if "arrival_actual" in timeline:
            arrival_actual = timeline["arrival_actual"].get("time")

        if "arrival_expected" in timeline:
            arrival_expected = timeline["arrival_expected"].get("time")

        # departure: take the LAST depart/exit-like event
        for event, ev_info in timeline.items():
            if "depart" in event or "exit" in event:
                t = ev_info.get("time")
                if t is not None:
                    departure = t

        container_metrics[key] = {
            "origin_mode": origin_mode,
            "origin_id": origin_id,
            "destination_mode": dest_mode,
            "destination_id": destination_id,
            "arrival_actual": arrival_actual,
            "arrival_expected": arrival_expected,
            "departure": departure,
        }

    for f in files.values():
        f.close()

    return container_metrics


# def main():
#     random.seed(42)
#     env = simpy.Environment()
#     start_time = time.time()
#
#     input_config_yaml = "input/config.yaml"
#     config = load_config(input_config_yaml)
#     output_dir = "output/ContainerLog/"
#     os.makedirs(output_dir, exist_ok=True)
#
#     terminal = Terminal(env, config)
#     # env.process(prestage_containers_at_t0(env, terminal, config))
#     timetable = generate_timetable(config, verbose=False)
#
#     for entry in timetable:
#         if entry["mode"] in ["train", "vessel"]:
#             env.process(train_vessel_arrival_process(env, terminal, entry))
#         elif entry["mode"] == "truck":
#             env.process(truck_arrival_process(env, terminal, entry))
#
#     sim_length = float(config["simulation"]["length"])
#     env.run(until=sim_length)
#
#     export_container_events_to_three_csvs(output_dir, "container")
#     print(f"Simulation costs {(time.time() - start_time):.2f} s. Results saved!")
#
#
# if __name__ == "__main__":
#     main()