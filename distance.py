from parameters import *
from scipy.stats import triang, uniform
import math
import yaml
import pandas as pd
from pathlib import Path


def load_config(config_path="input/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_layout(config):
    layout_path = Path(config["layout"]["file_path"])
    batch_size = config["simulation"]["train_batch_size"]
    df = pd.read_excel(layout_path)

    row = df.loc[df["train batch (k)"] == batch_size]
    if row.empty:
        raise ValueError(f"No layout found for train batch size {batch_size}")
    row = row.iloc[0]

    layout = {
        "M": int(row["rows (M)"]),
        "N": int(row["cols (N)"]),
        "n_t": int(row["trainlanes (n_t)"]),
        "n_p": int(row["parknglanes (n_p)"]),
        "n_r": int(row["blocklen (n_r)"]),
    }
    return layout


def load_config_and_layout(config_path="input/config.yaml"):
    config = load_config(config_path)
    layout = load_layout(config)
    return config, layout


def calculate_distances(config_path="input/config.yaml"):
    config, layout = load_config_and_layout(config_path)

    k = config["simulation"]["train_batch_size"]
    M, N, n_t, n_p, n_r = layout["M"], layout["N"], layout["n_t"], layout["n_p"], layout["n_r"]

    BL_l = 10 * n_r       # each block length
    BL_w = 80             # each block width
    P = 10                # aisle width

    A_yard = M * 10 * n_r + (M + 1) * n_p * P  # yard vertical
    B_yard = N * 80 + (N + 1) * n_p * P        # yard perpendicular
    total_lane_length = A_yard * (N + 1) + B_yard * (M + 1)

    d_t = n_t * BL_l
    d_y = M * N * BL_w
    d_g = (n_t + n_p) * 5.0
    total_distance = d_t + d_y + d_g

    # print(f"[INFO] Layout parameters: k={k}, M={M}, N={N}, n_t={n_t}, n_p={n_p}, n_r={n_r}")
    # print(f"[INFO] Yard geometry: A={A_yard:.1f}, B={B_yard:.1f}, total_lane_length={total_lane_length:.1f}")
    # print(f"[INFO] Base distances: d_t={d_t:.2f}, d_y={d_y:.2f}, d_g={d_g:.2f}, total={total_distance:.2f}")


    if YARD_TYPE == 'parallel':
        # d_h: hostler distance
        d_h_min = n_t * P + 1.5 * n_p * P
        d_h_max = n_t * P + A_yard - n_p * P + B_yard - n_p * P
        d_h_avg = (d_h_max + d_h_min) / 2

        # d_r: repositioning distance
        d_r_min = 5 * n_r + 40
        d_r_max = ugly_sigma(M) * (10 * n_r + n_p * P) + ugly_sigma(N) * (80 + n_p * P)
        d_r_avg = (d_r_max + d_r_min) / 2

        # d_t: truck distance
        d_t_min = 0.5 * n_p * P
        d_t_max = B_yard - n_p * P + A_yard - n_p * P
        d_t_avg = (d_t_max + d_t_min) / 2

        # d_tr: inter-track distance
        d_tr_min = 0.5 * l_c + d_f + d_x + 0.5 * n_p * P
        term = max(0, ((mu - 0.5) * (1 - mu)) / mu)
        d_tr_mean = term * n_max * l_c + ((n - 1) / 2) * l_c + d_f + d_x + 0.5 * n_p * P
        d_tr_max = n_max * l_c + d_f + d_x + 0.5 * n_p * P

    elif YARD_TYPE == 'perpendicular':
        d_h_min = n_t * P + 1.5 * n_p * P
        d_h_avg = 10 * n_r * M + 80 * N + (M + N + 1.5) * n_p * P + 2 * n_t * P
        d_h_max = n_t * P + A_yard - n_p * P + B_yard - n_p * P

        d_r_min = 0
        d_r_avg = 5 * n_r + 40 + ugly_sigma(M) * (10 * n_r + n_p * P) + ugly_sigma(N) * (80 + n_p * P)
        d_r_max = 10 * n_r + 80 + A_yard - n_p * P + B_yard - n_p * P

        d_t_min = 1.5 * n_p * P
        d_t_avg = 0.5 * (B_yard + A_yard - 0.5 * n_p * P)
        d_t_max = B_yard + A_yard - 2 * n_p * P

        d_tr_min = 0.5 * l_c + d_f + d_x + 0.5 * n_p * P
        term = max(0, ((mu - 0.5) * (1 - mu)) / mu)
        d_tr_mean = term * n_max * l_c + ((n - 1) / 2) * l_c + d_f + d_x + 0.5 * n_p * P
        d_tr_max = n_max * l_c + d_f + d_x + 0.5 * n_p * P

    else:
        raise ValueError("Invalid YARD_TYPE, choose 'parallel' or 'perpendicular'.")

    # print(f"[INFO] Yard type: {YARD_TYPE}")
    # print(f"[INFO] d_h_avg={d_h_avg:.2f}, d_r_avg={d_r_avg:.2f}, d_t_avg={d_t_avg:.2f}")

    # return total_distance
    return {
        "M": M, "N": N, "n_t": n_t, "n_p": n_p, "n_r": n_r,
        "P": P,
        "total_lane_length": total_lane_length,
        "d_h_min": d_h_min, "d_h_max": d_h_max,
        "d_r_min": d_r_min, "d_r_max": d_r_max,
        "d_t_min": d_t_min, "d_t_max": d_t_max,
        "d_tr_min": d_tr_min, "d_tr_mean": d_tr_mean, "d_tr_max": d_tr_max
    }



# Yard & Track fixed

YARD_TYPE = 'parallel'  # choose 'perpendicular' or 'parallel'
l_track = 1000   # The length of the whole track (ft)
l_c = 20         # The length of a railcar and joint (ft)
n_max = 10       # The maximum allowed railcars per track
d_f = 15         # The offset distance between crossing and train (ft)
d_x = 10         # The distance between two tracks (ft)
n = 6            # The actual railcars on the track
mu = n / n_max   # Ratio of train length and track length


def ugly_sigma(x):
    total_sum = 0
    for i in range(1, x):
        total_sum += 2 * i * (x - i)
    return total_sum / (x ** 2)


def speed_density(avg_density, vehicle_type, N):
    if vehicle_type == 'hostler':
        speed = 8 * math.e ** ((-1.5 * N - 0.5) * avg_density)
    elif vehicle_type == 'truck':
        speed = 10 * math.e ** ((-3.5 * N - 0.5) * avg_density)
    else:
        raise ValueError("Invalid vehicle type. Choose 'hostler' or 'truck'.")
    return speed


def simulate_truck_travel(truck_id, train_schedule, terminal, config_path="input/config.yaml"):
    params = calculate_distances(config_path)
    N = params["N"]
    total_lane_length = params["total_lane_length"]
    d_t_min, d_t_max = params["d_t_min"], params["d_t_max"]

    d_t_dist = 3.28 * uniform(loc=d_t_min, scale=(d_t_max - d_t_min)).rvs()
    current_veh_num = train_schedule["truck_number"] - len(terminal.truck_store.items)
    veh_density = current_veh_num / total_lane_length
    truck_speed = speed_density(veh_density, 'truck', N)
    truck_travel_time = d_t_dist / (2 * truck_speed * 3600)
    return truck_travel_time, d_t_dist, truck_speed, veh_density


def simulate_hostler_travel(hostler_id, current_veh_num, config_path="input/config.yaml"):
    params = calculate_distances(config_path)
    total_lane_length, N = params["total_lane_length"], params["N"]
    d_h_min, d_h_max = params["d_h_min"], params["d_h_max"]

    d_h_dist = 3.28 * uniform(loc=d_h_min, scale=(d_h_max - d_h_min)).rvs()
    veh_density = current_veh_num / total_lane_length
    hostler_speed = speed_density(veh_density, 'hostler', N)
    hostler_travel_time = d_h_dist / (hostler_speed * 3600)
    return hostler_travel_time, d_h_dist, hostler_speed, veh_density


def simulate_reposition_travel(hostler_id, current_veh_num, config_path="input/config.yaml"):
    params = calculate_distances(config_path)
    total_lane_length, N = params["total_lane_length"], params["N"]
    d_r_min, d_r_max = params["d_r_min"], params["d_r_max"]

    d_r_dist = 3.28 * uniform(loc=d_r_min, scale=(d_r_max - d_r_min)).rvs()
    veh_density = current_veh_num / total_lane_length
    hostler_speed = speed_density(veh_density, 'hostler', N)
    hostler_reposition_travel_time = d_r_dist / (hostler_speed * 3600)
    return hostler_reposition_travel_time, d_r_dist, hostler_speed, veh_density


def simulate_hostler_track_travel(hostler_id, current_veh_num, config_path="input/config.yaml"):
    params = calculate_distances(config_path)
    total_lane_length, N = params["total_lane_length"], params["N"]
    d_tr_min, d_tr_mean, d_tr_max = params["d_tr_min"], params["d_tr_mean"], params["d_tr_max"]

    c = 3.28 * (d_tr_mean - d_tr_min) / (d_tr_max - d_tr_min)
    d_tr_dist = triang(c, loc=d_tr_min, scale=d_tr_max - d_tr_min).rvs()
    veh_density = current_veh_num / total_lane_length
    hostler_speed = speed_density(veh_density, 'hostler', N)
    hostler_travel_time = d_tr_dist / (2 * hostler_speed * 3600)
    return hostler_travel_time


# # test
# if __name__ == "__main__":
#     total = calculate_distances("input/config.yaml")
#     print(f"Total estimated distance: {total:.2f}")