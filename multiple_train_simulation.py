import simpy
import random
from distances import *
from vehicle import *


class Terminal:
    def __init__(self, env, truck_capacity, chassis_count):
        self.env = env
        self.tracks = simpy.Store(env, capacity=2)
        for track_id in range(1, self.tracks.capacity + 1):
            self.tracks.put(track_id)
        # Example of number of cranes per track
        self.cranes_per_track = {
            1: 1,
            2: 1
        }
        self.cranes = simpy.FilterStore(env, sum(self.cranes_per_track.values()))

        self.all_trucks_arrived_events = {}  # condition for train arrival
        self.train_ic_unload_events = {}
        self.train_ic_unload_count = {}  # condition for train_ic_picked
        self.train_ic_picked_events = {}  # condition 1 for crane loading
        self.train_oc_prepared_events = {}  # condition 2 for crane loading
        self.train_start_load_events = {}  # condition 1 for train departure
        self.train_end_load_events = {}  # condition 2 for train departure
        self.train_departed_events = {}

        self.IC_COUNT = {}
        self.OC_COUNT = {}
        self.total_ic = {}
        self.total_oc = {}

        for item in self.cranes_per_track.items():
            track_id = item[0]
            cranes_on_this_track = item[1]
            for crane_number in range(1, cranes_on_this_track + 1):
                self.cranes.put(crane(type='Hybrid', id=crane_number, track_id=track_id))

        self.train_pool_stores = simpy.Store(env, capacity=99)  # train queue capacity
        self.train_ic_stores = simpy.FilterStore(env, capacity=9999)
        self.train_oc_stores = simpy.Store(env, capacity=9999)
        self.in_gates = simpy.Resource(env, state.IN_GATE_NUMBERS)
        self.out_gates = simpy.Resource(env, state.OUT_GATE_NUMBERS)
        self.oc_store = simpy.Store(env, capacity=9999)
        self.parking_slots = simpy.FilterStore(env, capacity=9999)  # store ic and oc in the parking area
        self.chassis = simpy.FilterStore(env, capacity=9999)
        self.parked_hostlers = simpy.Store(env, capacity=99)
        self.active_hostlers = simpy.Store(env, capacity=99)
        self.truck_store = simpy.Store(env)

        # Hostler setup
        hostler_diesel = round(state.HOSTLER_NUMBER * state.HOSTLER_DIESEL_PERCENTAGE)
        hostler_electric = state.HOSTLER_NUMBER - hostler_diesel
        hostlers = [hostler(id=i, type="Diesel") for i in range(hostler_diesel)] + \
                   [hostler(id=i + hostler_diesel, type="electric") for i in range(hostler_electric)]
        for hostler_id in hostlers:
            self.parked_hostlers.put(hostler_id)


def record_container_event(container, event_type, timestamp):
    global state
    if type(container) is str:
        container_string = container
    else:
        container_string = container.to_string()

    if container_string not in state.container_events:
        state.container_events[container_string] = {}
    state.container_events[container_string][event_type] = timestamp


# def save_vehicle_and_performance_metrics(state):
#     out_path = utilities.package_root() / 'output' / 'double_track_results'
#
#     container_excel_path = out_path / f"double_track_simulation_crane_{state.CRANE_NUMBER}_hostler_{state.HOSTLER_NUMBER}.xlsx"
#     vehicle_excel_path = out_path / f"double_track_vehicle_log_crane_{state.CRANE_NUMBER}_hostler_{state.HOSTLER_NUMBER}.xlsx"
#
#     if not container_excel_path.exists():
#         print(f"[Error] Container Excel not found: {container_excel_path}")
#         return
#     if not vehicle_excel_path.exists():
#         print(f"[Error] Vehicle Excel not found: {vehicle_excel_path}")
#         return
#
#     ic_time, oc_time, total_time = calculate_container_processing_time(container_excel_path)
#     ic_energy, oc_energy, total_energy = calculate_vehicle_energy(vehicle_excel_path)
#     print_and_save_metrics(ic_time, oc_time, total_time, ic_energy, oc_energy, total_energy)
#     return print_and_save_metrics(ic_time, oc_time, total_time, ic_energy, oc_energy, total_energy)


def emission_calculation(status, vehicle_category, vehicle, travel_time):
    global state
    vehicle_category = vehicle_category.capitalize()
    vehicle_type = vehicle.type.capitalize()
    if status == 'loaded':
        emission_unit = state.FULL_EMISSIONS_RATES[vehicle_category][vehicle_type]
    if status == 'empty':
        emission_unit = state.IDLE_EMISSIONS_RATES[vehicle_category][vehicle_type]
    emissions = emission_unit * travel_time

    return emissions


def truck_entry(env, terminal, truck, oc, train_schedule):
    global state
    ingate_request = terminal.in_gates.request()
    yield ingate_request

    # Assume each truck takes 1 OC, and drop OC to the closest parking lot according to triangular distribution
    # Assign IDs for OCs
    truck_travel_time = state.TRUCK_INGATE_TIME + random.uniform(0, state.TRUCK_INGATE_TIME_DEV)
    yield env.timeout(truck_travel_time)
    terminal.in_gates.release(ingate_request)
    # emissions = emission_calculation('loaded', 'truck', truck, truck_travel_time)
    # record_vehicle_event('truck', truck, f'entry_OC_{oc.id}', 'loaded', truck_travel_time, emissions, 'end', env.now)
    record_container_event(oc.to_string(), 'truck_arrival', env.now)

    ## Calculate truck speed according to the current density
    # truck_travel_time = simulate_truck_travel(truck, train_schedule, terminal, total_lane_length, d_t_min, d_t_max)
    # emissions = emission_calculation('loaded', 'truck', truck, truck_travel_time)
    # record_vehicle_event('truck', truck, f'entry_OC_{oc.id}', 'loaded', truck_travel_time, emissions, 'end', env.now)

    # Assume each truck takes 1 OC, and drop OC to the closest parking lot according to triangular distribution
    # Assign IDs for OCs
    record_container_event(oc.to_string(), 'truck_dropoff', env.now)
    terminal.parking_slots.put(oc)
    # emissions = emission_calculation('loaded', 'truck', truck, truck_travel_time)
    # record_vehicle_event('truck', truck, f'dropoff_OC_{oc.id}', 'loaded', truck_travel_time, emissions, 'end', env.now)


def empty_truck(env, terminal, truck_id):
    global state
    ingate_request = terminal.in_gates.request()
    yield ingate_request
    # if state.log_level > loggingLevel.NONE:
    #     print(f"Time {env.now:.3f}: {truck_id} passed the in-gate and is entering the terminal with empty loading")
    truck_travel_time = state.TRUCK_INGATE_TIME + random.uniform(0, state.TRUCK_INGATE_TIME_DEV)
    yield env.timeout(truck_travel_time)  # truck passing gate time: 1 sec (demo_parameters.TRUCK_INGATE_TIME and TRUCK_INGATE_TIME_DEV)
    terminal.in_gates.release(ingate_request)
    # Note the arrival of empty trucks will not be recorded due to excel output dimensions
    emissions = emission_calculation('empty', 'truck', truck_id, truck_travel_time)
    # record_vehicle_event('truck', truck_id, 'pass_gate', 'empty', truck_travel_time, emissions, 'end', env.now)


def truck_arrival(env, terminal, train_schedule):
    global state
    truck_number = train_schedule["truck_number"]
    num_diesel = round(truck_number * state.TRUCK_DIESEL_PERCENTAGE)
    num_electric = truck_number - num_diesel

    trucks = [truck(type="Diesel", id=i, train_id=train_schedule['train_id']) for i in range(num_diesel)] + \
             [truck(type="Electric", id=i + num_diesel, train_id=train_schedule['train_id']) for i in
              range(num_electric)]

    terminal.total_oc[train_schedule['train_id']] = train_schedule["oc_number"]

    for oc_id in range(terminal.OC_COUNT[train_schedule['train_id']],
                       terminal.OC_COUNT[train_schedule['train_id']] + terminal.total_oc[train_schedule['train_id']]):
        terminal.oc_store.put(container(type='Outbound', id=oc_id, train_id=train_schedule['train_id']))
        oc_id += 1
    # print(f"Time {env.now:.3f}: oc_store has {len(terminal.oc_store.items)} OC" )

    truck_entries_needed = train_schedule['oc_number']
    empty_truck_needed = len(trucks) - truck_entries_needed

    for i in range(truck_entries_needed):
        this_truck = trucks.pop(0)
        oc = yield terminal.oc_store.get()
        env.process(truck_entry(env, terminal, this_truck, oc, train_schedule))
        yield terminal.truck_store.put(this_truck)

    for i in range(empty_truck_needed):
        this_truck = trucks.pop(0)
        env.process(empty_truck(env, terminal, this_truck))
        yield terminal.truck_store.put(this_truck)

    terminal.all_trucks_arrived_events[train_schedule['train_id']].succeed()  # if all oc are dropped by trucks, train is allowed to enter
    print(f"Trucks: all OC for Train-{train_schedule['train_id']} have prepared. Train-{train_schedule['train_id']} can enter.")


def check_ic_picked_complete(env, terminal, train_schedule):
    train_id = train_schedule['train_id']
    remaining_ic = sum((getattr(item, 'type', None) == 'Inbound') and (getattr(item, 'train_id', None) == train_id) for item in terminal.chassis.items)
    # print(remaining_ic)
    # print(terminal.chassis.items)
    if (remaining_ic == 0
            and terminal.train_ic_unload_events[train_schedule['train_id']].triggered
            and not terminal.train_ic_picked_events[train_schedule['train_id']].triggered):
        terminal.train_ic_picked_events[train_schedule['train_id']].succeed()
        print(f"[Event] All ICs for Train-{train_schedule['train_id']} have been picked up by hostlers.")


def crane_unload_process(env, terminal, train_schedule, track_id):
    train_id = train_schedule['train_id']
    n = train_schedule['full_cars']
    unloaded = 0

    while unloaded < n:
        crane = yield terminal.cranes.get(lambda c: c.track_id == track_id)
        ic = yield terminal.train_ic_stores.get(lambda x: x.train_id == train_id)

        yield env.timeout(state.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME))
        record_container_event(ic.to_string(), 'crane_unload', env.now)
        yield terminal.cranes.put(crane)
        yield terminal.chassis.put(ic)

        env.process(container_process(env, terminal, train_schedule))
        unloaded += 1

    if unloaded == n and not terminal.train_ic_unload_events[train_id].triggered:
        terminal.train_ic_unload_events[train_id].succeed()
        print(f"[Event] All ICs for train-{train_schedule['train_id']} have been unloaded.")


def get_hostler(terminal):
    parked_available = len(terminal.parked_hostlers.items) > 0
    active_available = len(terminal.active_hostlers.items) > 0
    if (active_available is False) and parked_available:
        assigned_hostler = terminal.parked_hostlers.get()
    else:
        assigned_hostler = terminal.active_hostlers.get()

    return assigned_hostler


def return_hostler(terminal, assigned_hostler, travel_time_to_active, travel_time_to_parking):
    active_hostlers_needed = len(terminal.active_hostlers.get_queue) > 0
    if active_hostlers_needed:
        yield terminal.active_hostlers.put()
    else:
        current_veh_num = 2 * state.HOSTLER_NUMBER - (len(terminal.parked_hostlers.items()) + len(terminal.active_hostlers.items()))
        simulate_hostler_travel(assigned_hostler, current_veh_num, total_lane_length, d_h_min, d_h_max)
        yield terminal.parked_hostler.put()


def container_process(env, terminal, train_schedule):
    global state
    '''
    It is designed to transfer both inbound and outbound containers (IC-OC loop).
    The main simulation process is as follows:
    1. A hostler picks up an IC, and drops off IC at parking slot.
    2. A truck picks up the IC, and leaves the gate
    3. The hostler picks up an OC, and drops off OC at the chassis.
    4. Once all OCs are prepared (all_oc_prepared), triggers the event of crane loading.
    '''

    # 1. An IC is waiting to be picked up by a hostler
    ic = yield terminal.chassis.get(lambda x: x.type == 'Inbound')
    assigned_hostler = yield get_hostler(terminal)

    # 2. hostler travel time 1: the empty hostler pick up the IC
    current_veh_num = 2 * state.HOSTLER_NUMBER - len(terminal.parked_hostlers.items) - len(terminal.active_hostlers.items)
    hostler_travel_time_to_track = simulate_hostler_track_travel(assigned_hostler.to_string(), current_veh_num, total_lane_length, d_tr_min, d_tr_mean, d_tr_max)
    yield env.timeout(hostler_travel_time_to_track)

    # 3. hostler travel time 2: the loaded hostler going to drop off the IC
    yield env.timeout(0.1)
    terminal.parking_slots.put(ic)
    record_container_event(ic.to_string(), 'hostler_pickup', env.now)

    check_ic_picked_complete(env, terminal, train_schedule)
    yield terminal.train_ic_unload_events[train_schedule['train_id']]

    # 3. Hostler drop off IC to parking slot
    yield env.timeout(0.1)  # side-pick
    record_container_event(ic.to_string(), 'hostler_dropoff', env.now)

    # 4. Assign a truck to pick up IC
    assigned_truck = yield terminal.truck_store.get()
    ic = yield terminal.parking_slots.get(lambda x: x.type == 'Inbound')
    record_container_event(ic.to_string(), 'truck_pickup', env.now)
    env.process(truck_exit(env, terminal, assigned_truck, ic, train_schedule))

    # # 5. Assign the hostler to pick up an OC (FIFO)
    # # print("parking slots", terminal.parking_slots.items)
    # if sum(item.type == 'Outbound' for item in terminal.parking_slots.items) > 0:
    #     oc = yield terminal.parking_slots.get(lambda x: x.type == 'Outbound')
    #
    #     # 5.1 The hostler picks up an OC
    #     hostler_reposition_travel_time = simulate_reposition_travel(assigned_hostler, current_veh_num, total_lane_length, d_h_min, d_h_max)
    #     yield env.timeout(hostler_reposition_travel_time)
    #
    #     # print(f"hostler picked up {oc.to_string()}")
    #     record_container_event(oc.to_string(), 'hostler_pickup', env.now)
    #
    #     # 5.2 The hostler drops off OC at the chassis
    #     current_veh_num = 2 * state.HOSTLER_NUMBER - len(terminal.parked_hostlers.items) - len(terminal.active_hostlers.items)
    #     hostler_travel_time_to_chassis = simulate_hostler_travel(assigned_hostler, current_veh_num, total_lane_length,d_h_min, d_h_max)
    #     current_veh_num = 2 * state.HOSTLER_NUMBER - len(terminal.parked_hostlers.items) - len(terminal.active_hostlers.items)
    #     hostler_travel_time_to_track = simulate_hostler_track_travel(assigned_hostler.to_string(), current_veh_num, total_lane_length, d_tr_min, d_tr_mean, d_tr_max)
    #     yield env.timeout(hostler_travel_time_to_chassis + hostler_travel_time_to_track)
    #     yield terminal.chassis.put(oc)
    #     record_container_event(oc.to_string(), 'hostler_dropoff', env.now)
    #
    #     # 5.3 return hostler
    #     return_hostler(terminal, assigned_hostler,
    #                    travel_time_to_active=0,
    #                    travel_time_to_parking=0.1
    #                    )
    #
    # # 7. todo: Check OCs are all prepared for a train
    # train_id = train_schedule['train_id']
    # outbound_remaining = sum((item.type == 'Outbound' and item.train_id == train_id) for item in terminal.parking_slots.items)
    # chassis_remaining = sum((item.type == 'Outbound' and item.train_id == train_id) for item in terminal.chassis.items)
    #
    # if (outbound_remaining != 0) and (chassis_remaining == train_schedule['oc_number']) and (terminal.train_oc_prepared_events[train_schedule['train_id']].triggered == False):
    # # outbound_count = sum((getattr(item, 'type', None) == 'Outbound') and (getattr(item, 'train_id', None) == train_schedule['train_id']) for item in terminal.chassis.items)
    # # if (outbound_count == train_schedule['oc_number']
    #     terminal.train_oc_prepared_events[train_schedule['train_id']].succeed()
    #     print(f"Time {env.now:.3f}: All OCs for {train_schedule['train_id']} are ready on chassis.")
    # else:
    #     return


def handle_remaining_oc(env, terminal, train_schedule):
    train_id = train_schedule['train_id']

    while True:
        # 1) how many oc remaining? & how many oc prepared?
        outbound_remaining = sum(
            (item.type == 'Outbound') and (item.train_id == train_id)
            for item in terminal.parking_slots.items)

        chassis_remaining = sum(
            (item.type == 'Outbound') and (item.train_id == train_id)
            for item in terminal.chassis.items)

        if outbound_remaining == 0 and chassis_remaining == train_schedule['oc_number']:
            print(f"Time {env.now:.3f}: All OC handled for train {train_id}.")
            if not terminal.train_oc_prepared_events[train_id].triggered:
                terminal.train_oc_prepared_events[train_id].succeed()
            return

        # 2) transport OC from paring slots
        oc = yield terminal.parking_slots.get(
            lambda x: (x.type == 'Outbound') and (x.train_id == train_id)
        )
        record_container_event(oc.to_string(), 'hostler_pickup', env.now)

        # 3) assign hostler
        assigned_hostler = yield get_hostler(terminal)
        current_veh_num = 2 * state.HOSTLER_NUMBER - len(terminal.parked_hostlers.items) - len(terminal.active_hostlers.items)
        hostler_travel_time_to_parking = simulate_hostler_travel(
            assigned_hostler.to_string(), current_veh_num, total_lane_length, d_h_min, d_h_max)
        yield env.timeout(hostler_travel_time_to_parking)
        # print(f"ICs are prepared, but OCs remaining: {len([it for it in terminal.oc_store.items if it.train_id == train_id])}")

        # 4) hostler loaded with an OC -> chassis
        dropoff_veh_num = 2 * state.HOSTLER_NUMBER - len(terminal.parked_hostlers.items) - len(terminal.active_hostlers.items)
        travel_time_to_chassis = simulate_reposition_travel(assigned_hostler, dropoff_veh_num, total_lane_length, d_h_min, d_h_max)
        yield env.timeout(travel_time_to_chassis)
        print(f"Time {env.now:.3f}: Hostler {assigned_hostler} dropped off OC {oc} onto chassis")
        yield terminal.chassis.put(oc)
        record_container_event(oc.to_string(), 'hostler_dropoff', env.now)

        # 5) hostler travel back
        return_hostler(terminal, assigned_hostler,
            travel_time_to_active=0,
            travel_time_to_parking=hostler_travel_time_to_parking
        )


def truck_exit(env, terminal, truck, ic, train_schedule):
    global state
    out_gate_request = terminal.out_gates.request()
    yield out_gate_request
    # print(f"Time {env.now:.3f}: Truck {truck.id} with IC-{ic.id}-Train-{train_schedule['train_id']} is passing through the out-gate and leaving the terminal")
    truck_travel_time = state.TRUCK_OUTGATE_TIME + random.uniform(0, state.TRUCK_OUTGATE_TIME_DEV)
    yield env.timeout(truck_travel_time)
    terminal.out_gates.release(out_gate_request)
    record_container_event(ic.to_string(), 'truck_exit', env.now)
    emissions = emission_calculation('loaded', 'truck', truck, truck_travel_time)
    # record_vehicle_event('truck', truck, f"leave_gate_IC_{ic.id}-Train-{train_schedule['train_id']}", 'loaded',
    #                      truck_travel_time, emissions, 'end', env.now)

    yield terminal.truck_store.put(truck)


def crane_load_process(env, terminal, track_id, train_schedule):
    global state

    yield terminal.train_start_load_events[train_schedule['train_id']]

    while sum(item.type == 'Outbound' and item.train_id == train_schedule['train_id'] for item in terminal.chassis.items) > 0:  # if there still has OC on chassis
        crane_item = yield terminal.cranes.get(lambda x: x.track_id == track_id)
        # oc = yield terminal.chassis.get(lambda x: x.type == 'Outbound')
        oc = yield terminal.chassis.get(
            lambda x: x.type == 'Outbound' and x.train_id == train_schedule['train_id']
        )

        crane_load_move_time = state.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME)
        yield env.timeout(crane_load_move_time)  # loading time, depends on container parameter**
        record_container_event(oc.to_string(), 'crane_load', env.now)
        print(f"Time {env.now:.3f}: Crane {crane_item.to_string()} finished loading {oc} onto the train.")

        yield terminal.train_oc_stores.put(oc)
        yield terminal.cranes.put(crane_item)

    terminal.train_end_load_events[train_schedule['train_id']].succeed()
    print(f"Time {env.now:.3f}: All OCs loaded. Train-{train_schedule['train_id']} is fully loaded and ready to depart.")


def handle_train_departure(env, terminal, train_schedule, train_id, track_id, arrival_time):
    global state

    if env.now < train_schedule["departure_time"]:
        print(f"Time {env.now}: [EARLY] Train {train_id} departs from the track {track_id}.")
    elif env.now == train_schedule["departure_time"]:
        print(f"Time {env.now}: [In Time] Train {train_id} departs from the track {track_id}.")
    else:
        delay_time = env.now - train_schedule["departure_time"]
        print(f"Time {env.now}: [DELAYED] Train {train_id} has been delayed for {delay_time} hours from the track {track_id}.")

    terminal.tracks.put(track_id)

    for oc_id in range(terminal.OC_COUNT[train_schedule['train_id']],
                       terminal.OC_COUNT[train_schedule['train_id']] + train_schedule['oc_number']):
        record_container_event(f"OC-{oc_id}-Train-{train_schedule['train_id']}", 'train_depart',env.now)  # loop: assign container_id range(current_oc, current_oc + train_schedule['full_cars'])
    state.time_per_train[train_schedule['train_id']] = env.now - arrival_time

    # For container ID record: track inbound and outbound containers for each train
    terminal.IC_COUNT[train_schedule['train_id']] += train_schedule['full_cars']
    terminal.OC_COUNT[train_schedule['train_id']] += train_schedule['oc_number']
    # print(f"terminal oc_num for {train_schedule['train_id']}: {terminal.OC_COUNT[train_schedule['train_id']]}")

    terminal.train_departed_events[train_schedule['train_id']].succeed()


def train_process_per_track(env, terminal, track_id, train_schedule, train_id, arrival_time, oc_needed):
    # Crane unload & hostler process ICs
    env.process(crane_unload_process(env, terminal, train_schedule, track_id))

    # check before crane loading
    # condition 1: all ic picked
    yield terminal.train_ic_picked_events[train_schedule['train_id']]
    print(f"[Event]: All {train_schedule['full_cars']} IC picked events are finished for train {train_schedule['train_id']}.")

    # condition 2 & 3: no OCs on parking slots - OCs remaining -> process rest OCs; all OC prepared
    if sum((item.type == 'Outbound') and (item.train_id == train_id) for item in terminal.parking_slots.items) != 0:
        env.process(handle_remaining_oc(env, terminal, train_schedule))
    yield terminal.train_oc_prepared_events[train_schedule['train_id']]
    print(f"[Event]: All {oc_needed} OCs are ready on chassis and no OCs remaining on parking slots for train {train_schedule['train_id']}.")
    print("chassis:", terminal.chassis.items)

    # crane loading
    # only when 1. all_ic_picked (chassis), 2. all_oc_picked (parking slots) & 3. all_oc_prepared (chassis) satisfied -> crane loading starts
    terminal.train_start_load_events[train_schedule['train_id']].succeed()
    env.process(crane_load_process(env, terminal, track_id=track_id, train_schedule=train_schedule))
    yield terminal.train_end_load_events[train_schedule['train_id']]

    # train departure
    handle_train_departure(env, terminal, train_schedule, train_id, track_id, arrival_time)


def initialize_train_events(env, terminal, train_id):
    for name in [
        "train_ic_unload_events",
        "train_oc_prepared_events",
        "train_ic_picked_events",
        "train_start_load_events",
        "train_end_load_events",
        "train_departed_events",
    ]:
        if not hasattr(terminal, name):
            setattr(terminal, name, {})
        d = getattr(terminal, name)

        if train_id not in d or d[train_id].triggered:
            d[train_id] = env.event()


def process_train_arrival(env, terminal, train_schedule):
    global state

    train_id = train_schedule["train_id"]
    arrival_time = train_schedule["arrival_time"]
    oc_needed = train_schedule["oc_number"]

    # Create events as processing conditions
    terminal.all_trucks_arrived_events[train_schedule['train_id']] = env.event() # condition for train arrival

    # Initialize dictionary
    delay_list = {}

    # Initialize IC & OC count (generating container ID)
    terminal.IC_COUNT[train_id] = 1
    terminal.OC_COUNT[train_id] = 1

    # All trucks arrive before train arrives
    env.process(truck_arrival(env, terminal, train_schedule))

    # Train arrival
    yield terminal.all_trucks_arrived_events[train_schedule['train_id']]
    if env.now <= arrival_time:
        yield env.timeout(arrival_time - env.now)
        print(f"Time {env.now:.3f}: [In Time] Train {train_schedule['train_id']}.")
        delay_time = 0
    else:
        delay_time = env.now - arrival_time
        if f"train_id_{train_id}" not in delay_list:
            delay_list[f"train_id_{train_id}"] = {}
        delay_list[f"train_id_{train_id}"]["arrival"] = delay_time
        print(f"Time {env.now:.3f}: [DELAYED] Train {train_schedule['train_id']} has been delayed for {delay_time} hours.")
    state.train_delay_time[train_schedule['train_id']] = delay_time
    terminal.train_pool_stores.put(train_schedule['train_id'])

    # Track assignment
    track_id = yield terminal.tracks.get()
    if track_id is None:
        print(f"Time {env.now:.3f}: Train {train_id} is waiting for a next available track.")
        return
    else:
        train_id = yield terminal.train_pool_stores.get()
        print(f"Time {env.now:.3f}: Train {train_id} has been assigned to track {track_id}.")

        # Initialize train with loaded ICs
        ic_num = terminal.IC_COUNT[train_schedule['train_id']]
        for ic_id in range(ic_num, ic_num + train_schedule['full_cars']):
            ic = container(type='Inbound', id=ic_id, train_id=train_schedule['train_id'])
            terminal.train_ic_stores.put(ic)
            record_container_event(ic.to_string(), 'train_arrival_expected', train_schedule['arrival_time'])
            record_container_event(ic.to_string(), 'train_arrival_actual', env.now)

        # Initialize events for each train
        initialize_train_events(env, terminal, train_id)
        # Train processed separately on each track
        env.process(train_process_per_track(env, terminal, track_id, train_schedule, train_id, arrival_time, oc_needed))


def run_simulation(train_consist_plan: pl.DataFrame, terminal: str, out_path=None):
    global state
    state.terminal = terminal
    state.train_consist_plan = train_consist_plan
    state.initialize()

    # Create environment
    env = simpy.Environment()

    # Train timetable: shorter headway
    train_timetable = [
        {"train_id": 31, "arrival_time": 0, "departure_time": 10, "empty_cars": 0, "full_cars": 3, "oc_number": 1,
         "truck_number": 3},  # test: ic > oc
        {"train_id": 13, "arrival_time": 0, "departure_time": 10, "empty_cars": 0, "full_cars": 1, "oc_number": 3,
         "truck_number": 3},  # test: ic < oc
        {"train_id": 22, "arrival_time": 0, "departure_time": 10, "empty_cars": 0, "full_cars": 2, "oc_number": 2,
         "truck_number": 2},  # test: ic = oc
        ]

    truck_number = max([entry['truck_number'] for entry in train_timetable])
    chassis_count = max([entry['empty_cars'] + entry['full_cars'] for entry in train_timetable])
    terminal = Terminal(env, truck_capacity=truck_number, chassis_count=chassis_count)

    # Trains arrive according to timetable
    print("train timetable:")
    for i, train_schedule in enumerate(train_timetable):
        print(train_schedule)
        env.process(process_train_arrival(env, terminal, train_schedule))

    print("*"*30)
    print(f"Tracks: {len(terminal.tracks.items)}; Cranes: {len(terminal.cranes.items)}; Hostlers: {len(terminal.parked_hostlers.items)}")
    print("*" * 30)

    env.run(until=state.sim_time)

    container_data = (
        pl.from_dicts(
            [dict(event, **{'container_id': container_id}) for container_id, event in state.container_events.items()],
            infer_schema_length=None
        )
        .with_columns(
            pl.when(
                pl.col("truck_exit").is_not_null() & pl.col("train_arrival_expected").is_not_null()
            )
            .then(
                pl.col("truck_exit") - pl.col("train_arrival_expected")
            )
            .when(
                pl.col("train_depart").is_not_null()
            )
            .then(
                pl.col("crane_load") - pl.col("truck_arrival")
            )
            .otherwise(None)
            .alias("container_processing_time")
        )
        .sort("container_id")
        .select(pl.col("container_id"), pl.exclude("container_id"))
    )
    if out_path is not None:
        container_data.write_excel(out_path / f"double_track_simulation_crane_{state.CRANE_NUMBER}_hostler_{state.HOSTLER_NUMBER}.xlsx")

    # # Use save_average_times and save_vehicle_logs for vehicle related logs
    # save_energy_to_excel(state)
    # save_vehicle_and_performance_metrics(state)

    print("Done!")
    return None


if __name__ == "__main__":
    run_simulation(
        train_consist_plan=pl.read_csv(utilities.package_root() / 'input' / 'train_consist_plan.csv'),
        terminal="Allouez",
        out_path=utilities.package_root() / 'output' / 'double_track_results'
    )

    print("simulation time:", state.time_per_train)
    print(f"Average train processing time: {sum(state.time_per_train) / len(state.time_per_train) if state.time_per_train else 0:.2f}")
    print(f"output root is {utilities.package_root()}")
    print("Simulation completed. ")