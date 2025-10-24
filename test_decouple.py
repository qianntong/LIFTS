import simpy
from decouple import should_decouple, decouple_train, handle_decoupled_trains


class MiniTerminal:
    def __init__(self, env):
        self.env = env
        self.tracks = simpy.Store(env, capacity=3)
        for i in range(1, 3):
            self.tracks.put(i)

        # For logging and synchronization
        self.train_times = {}
        self.train_departed_events = {}
        self.master_train_events = {}
        self.master_train_mapping = {}


def process_sub_train_arrival(env, terminal, sub_train, completion_event=None):
    sub_id = sub_train["sub_train_id"]
    work_time = 3.0   # this supposed to call train process func

    print(f"[Time={env.now:6.2f}] Sub-train {sub_id} starts work (takes {work_time:.2f} hrs)")
    yield env.timeout(work_time)

    terminal.train_times[sub_id] = work_time
    print(f"[Time={env.now:6.2f}] Sub-train {sub_id} finished work")

    if completion_event and not completion_event.triggered:
        completion_event.succeed()


def test_decouple():
    env = simpy.Environment()
    terminal = MiniTerminal(env)

    # Sample train schedule (too long → must split)
    train_schedule = {
        "train_id": 1,
        "arrival_time": 0.1,
        "departure_time": 10.1,
        "empty_cars": 0,
        "full_cars": 70,     # 70 cars, capacity only 25
        "oc_number": 70,
        "truck_number": 70,
    }

    track_capacity = 25 # math.ceil(terminal length (ft) / 60 ft)

    # Step 1. Check if split required
    if should_decouple(train_schedule, track_capacity):
        print("\n[TEST] Train too long, splitting...\n")
        sub_trains = decouple_train(train_schedule, track_capacity)
    else:
        print("\n[TEST] No decoupling required.\n")
        sub_trains = [train_schedule]

    # Step 2. Launch decoupled operations
    env.process(handle_decoupled_trains(env, terminal, sub_trains, process_sub_train_arrival))

    # Step 3. Run simulation
    print("\n[TEST] Starting simulation...\n")
    env.run()

    # Step 4. Summary
    print("\n=== Summary ===")
    for k, v in terminal.train_times.items():
        print(f"Train {k}: total time = {v:.2f} hrs")

    print(f"Master departure events: {list(terminal.train_departed_events.keys())}")
    print(f"Tracks remaining: {len(terminal.tracks.items)}")


if __name__ == "__main__":
    test_decouple()