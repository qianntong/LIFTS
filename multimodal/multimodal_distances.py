import math
import yaml
import pandas as pd
from scipy.stats import uniform, triang

FT_TO_M = 1 / 3.28
MPH_TO_MPS = 0.44704


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_layout(config):
    layout_cfg = config["layout"]
    yard_cfg = config["yard"]

    mode = layout_cfg.get("mode", "fixed").lower()

    if mode == "adaptive":
        df = pd.read_excel(layout_cfg["file_path"])

        train_batch = config["timetable"]["train"]["batch_size"]
        num_tracks = yard_cfg["track_number"]

        # capacity definition: IC + OC on all tracks
        capacity = train_batch * num_tracks * 2

        row = df.loc[df["train batch (k)"] == capacity]
        if row.empty:
            raise ValueError(
                f"No adaptive layout found for capacity={capacity}"
            )
        row = row.iloc[0]

        M = int(row["rows (M)"])
        N = int(row["cols (N)"])
        n_t = int(row["trainlanes (n_t)"])
        n_p = int(row["parklanes (n_p)"])
        n_r = int(row["blocklen (n_r)"])

    else:
        M = int(layout_cfg["M"])
        N = int(layout_cfg["N"])
        n_t = int(layout_cfg["n_t"])
        n_p = int(layout_cfg["n_p"])
        n_r = int(layout_cfg["n_r"])

    return {
        "M": M,
        "N": N,
        "n_t": n_t,
        "n_p": n_p,
        "n_r": n_r,
        "mode": mode,
    }


def calculate_distances(config, actual_railcars=None):
    """
    Compute yard geometry and distance distributions.
    All distances returned in meters.
    """

    yard_cfg = config["yard"]
    layout = resolve_layout(config)

    M = layout["M"]
    N = layout["N"]
    n_t = layout["n_t"]
    n_p = layout["n_p"]
    n_r = layout["n_r"]


    # basic geometry
    P = 12 * FT_TO_M                      # lane width
    BL_l = 10 * n_r * FT_TO_M             # block length
    BL_w = 80 * FT_TO_M                   # block width

    A_yard = M * BL_l + (M + 1) * n_p * P
    B_yard = N * BL_w + (N + 1) * n_p * P
    total_lane_length = A_yard * (N + 1) + B_yard * (M + 1)

    # rail occupancy (mu)
    railcar_length = yard_cfg["railcar_length"] * FT_TO_M
    train_batch = config["timetable"]["train"]["batch_size"]

    n_max = train_batch
    n = 0 if actual_railcars is None else int(actual_railcars)

    mu = 1.0 if n == 0 else min(1.0, (n * railcar_length) / A_yard)

    # yard-type-specific distances
    YARD_TYPE = yard_cfg["yard_type"].lower()

    l_c = 60 * FT_TO_M
    d_f = yard_cfg["d_f"] * FT_TO_M
    d_x = yard_cfg["d_x"] * FT_TO_M

    if YARD_TYPE == "parallel":
        # truck
        d_t_min = 0.5 * n_p * P
        d_t_max = A_yard + B_yard - 2 * n_p * P

        # hostler reposition
        d_r_min = 5 * n_r * FT_TO_M + 40 * FT_TO_M
        d_r_max = (
            ugly_sigma(M) * (10 * n_r * FT_TO_M + n_p * P)
            + ugly_sigma(N) * (80 * FT_TO_M + n_p * P)
        )

        # inter-track
        d_tr_min = l_c + d_f + d_x + n_p * P + 50 * FT_TO_M
        d_tr_mean = (
            ((n_max - 1) / 2) * l_c
            + d_f + d_x + n_p * P + 50 * FT_TO_M
        )
        d_tr_max = (
            n_max * l_c
            + d_f + d_x + n_p * P + 50 * FT_TO_M
        )

    elif YARD_TYPE == "perpendicular":
        d_t_min = 1.5 * n_p * P
        d_t_max = A_yard + B_yard - 2 * n_p * P

        d_r_min = 0.0
        d_r_max = (
            10 * n_r * FT_TO_M
            + 80 * FT_TO_M
            + A_yard + B_yard - 2 * n_p * P
        )

        d_tr_min = 0.5 * l_c + d_f + d_x + 0.5 * n_p * P + 50 * FT_TO_M
        d_tr_mean = (
            ((n - 1) / 2) * l_c
            + d_f + d_x + 0.5 * n_p * P + 50 * FT_TO_M
        )
        d_tr_max = (
            n_max * l_c
            + d_f + d_x + 0.5 * n_p * P + 50 * FT_TO_M
        )

    else:
        raise ValueError("yard_type must be 'parallel' or 'perpendicular'")

    return {
        "layout": layout,
        "yard": {
            "A_yard": A_yard,
            "B_yard": B_yard,
            "total_lane_length": total_lane_length,
        },
        "rail": {
            "n": n,
            "n_max": n_max,
            "mu": mu,
        },
        "distance": {
            "truck": (d_t_min, d_t_max),
            "reposition": (d_r_min, d_r_max),
            "intertrack": (d_tr_min, d_tr_mean, d_tr_max),
        },
    }


def ugly_sigma(x):
    total = 0
    for i in range(1, x):
        total += 2 * i * (x - i)
    return total / (x ** 2)


def speed_density(avg_density, vehicle_type, N):
    """
    avg_density: veh/m
    return speed in mph
    """
    if vehicle_type == "hostler":
        return 5 * math.exp((-1.5 * N - 0.5) * avg_density)
    elif vehicle_type == "truck":
        return 10 * math.exp((-3.5 * N - 0.5) * avg_density)
    else:
        raise ValueError("vehicle_type must be 'truck' or 'hostler'")


def compute_travel_time(distance_m,current_veh_num,vehicle_type,total_lane_length,N):

    density = current_veh_num / total_lane_length
    speed_mph = speed_density(density, vehicle_type, N)
    speed_mps = speed_mph * MPH_TO_MPS
    travel_time = distance_m / speed_mps
    return travel_time, speed_mps, density


def simulate_truck_travel(current_veh_num, config):
    params = calculate_distances(config)
    d_min, d_max = params["distance"]["truck"]

    dist = uniform(d_min, d_max - d_min).rvs()
    return compute_travel_time(
        dist,
        current_veh_num,
        "truck",
        params["yard"]["total_lane_length"],
        params["layout"]["N"],
    )


def simulate_reposition_travel(current_veh_num, config):
    params = calculate_distances(config)
    d_min, d_max = params["distance"]["reposition"]

    dist = uniform(d_min, d_max - d_min).rvs()
    return compute_travel_time(
        dist,
        current_veh_num,
        "hostler",
        params["yard"]["total_lane_length"],
        params["layout"]["N"],
    )


def simulate_hostler_track_travel(current_veh_num, config):
    params = calculate_distances(config)
    d_min, d_mean, d_max = params["distance"]["intertrack"]

    if not (d_min < d_mean < d_max):
        raise ValueError("Invalid triangular distribution parameters")

    c = (d_mean - d_min) / (d_max - d_min)
    c = min(max(c, 1e-6), 1 - 1e-6)

    dist = triang(c, loc=d_min, scale=d_max - d_min).rvs()
    return compute_travel_time(
        dist,
        current_veh_num,
        "hostler",
        params["yard"]["total_lane_length"],
        params["layout"]["N"],
    )
