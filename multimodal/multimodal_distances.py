import math
import yaml
import pandas as pd
from scipy.stats import uniform, triang
from typing import Optional, List


FT_TO_M = 1/3.28


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_layout(config: dict, basis_mode: Optional[str] = None) -> dict:

    """
    Resolve yard layout.

    layout.mode:
      - "fixed": use M,N,n_t,n_p,n_r from config
      - "adaptive": read from excel and select by capacity

    layout.capacity_basis (only for adaptive):
      - "train" | "vessel" | "max"

    basis_mode parameter overrides layout.capacity_basis if provided.
    """
    layout_cfg = config["layout"]
    yard_cfg = config["yard"]

    mode = layout_cfg.get("mode", "fixed").lower()

    if mode == "adaptive":
        df = pd.read_excel(layout_cfg["file_path"])

        if basis_mode is None:
            basis_mode = layout_cfg.get("capacity_basis", "train")
        basis_mode = basis_mode.lower()

        if basis_mode == "train":
            batch = int(config["timetable"]["train"]["batch_size"])
        elif basis_mode == "vessel":
            batch = int(config["timetable"]["vessel"]["batch_size"])
        elif basis_mode == "max":
            batch = max(
                int(config["timetable"]["train"]["batch_size"]),
                int(config["timetable"]["vessel"]["batch_size"]),
            )
        else:
            raise ValueError("capacity_basis must be train, vessel, or max")

        # num_tracks = int(yard_cfg["track_number"])

        # capacity definition: IC + OC
        capacity = batch * 2

        row = df.loc[df["train batch (k)"] == capacity]
        if row.empty:
            raise ValueError(f"No adaptive layout found for capacity={capacity}")

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

    return {"M": M, "N": N, "n_t": n_t, "n_p": n_p, "n_r": n_r, "mode": mode}


def ugly_sigma(x: int) -> float:
    total = 0.0
    for i in range(1, x):
        total += 2 * i * (x - i)
    return total / (x**2)


def speed_density(avg_density: float, vehicle_type: str, N: int) -> float:
    """
    avg_density: veh/m
    return speed in mps
    """
    if vehicle_type == "hostler":
        return 7.5 * math.exp((-1.5 * N - 0.5) * avg_density)
    if vehicle_type == "truck":
        return 10 * math.exp((-3.5 * N - 0.5) * avg_density)

    raise ValueError("vehicle_type must be 'truck' or 'hostler'")


from typing import Optional

class DistanceModel:
    """
    Single-instance distance + travel time model.
    - Distances are sampled from distributions determined by layout.
    - Travel time converts sampled distance -> time via speed-density function.
    - Composite paths are represented as lists of segment modes.
    """

    def __init__(self, config: dict, basis_mode: Optional[str] = None):
        self.config = config
        self.layout = resolve_layout(config, basis_mode=basis_mode)

        self._compute_geometry()
        self._compute_distance_ranges()

    def _compute_geometry(self) -> None:
        yard_cfg = self.config["yard"]
        layout = self.layout

        M = layout["M"]
        N = layout["N"]
        n_p = layout["n_p"]
        n_r = layout["n_r"]

        # basic geometry
        P = 12 * FT_TO_M  # lane width
        BL_l = 10 * n_r * FT_TO_M  # block length
        BL_w = 80 * FT_TO_M  # block width

        A_yard = M * BL_l + (M + 1) * n_p * P
        B_yard = N * BL_w + (N + 1) * n_p * P

        self.A_yard = A_yard
        self.B_yard = B_yard
        self.total_lane_length = 12

        self.N = N

        # rail occupancy params (kept for future use)
        self.railcar_length = float(yard_cfg["railcar_length"]) * FT_TO_M
        self.train_batch = int(self.config["timetable"]["train"]["batch_size"])

    def _compute_distance_ranges(self) -> None:
        """
        Your summarized rules:
          1) Truck distance always uses a fixed min/max expression.
          2) Hostler:
             - stack->track uses d_hostler + d_intertrack (composite)
             - stack->vessel uses d_hostler
          3) Hostler reposition uses d_reposition
          4) Track->stack uses d_hostler + d_intertrack, Vessel->stack uses d_hostler
        """
        yard_cfg = self.config["yard"]
        layout = self.layout

        M = layout["M"]
        N = layout["N"]
        n_p = layout["n_p"]
        n_r = layout["n_r"]

        P = 12 * FT_TO_M
        l_c = 60 * FT_TO_M
        d_f = 15 * FT_TO_M
        d_x = 15 * FT_TO_M

        yard_type = str(yard_cfg["yard_type"]).lower()

        if yard_type == "parallel":
            # 1) truck
            d_t_min = 0.5 * n_p * P
            d_t_max = (self.total_lane_length / (N + 1)) + (self.total_lane_length / (M + 1)) - 2 * n_p * P
            self.d_truck = (d_t_min, d_t_max)

            # 2) hostler generic (yard movement)
            self.d_hostler = (d_t_min, d_t_max)

            # 3) reposition
            d_r_min = 5 * n_r * FT_TO_M + 40 * FT_TO_M
            d_r_max = (ugly_sigma(M) * (10 * n_r * FT_TO_M + n_p * P) + ugly_sigma(N) * (80 * FT_TO_M + n_p * P))
            self.d_reposition = (d_r_min, d_r_max)

            # 4) intertrack (triangular)
            n_max = self.train_batch
            d_tr_min = l_c + d_f + d_x + n_p * P + 50 * FT_TO_M
            d_tr_mean = ((n_max - 1) / 2) * l_c + d_f + d_x + n_p * P + 50 * FT_TO_M
            d_tr_max = n_max * l_c + d_f + d_x + n_p * P + 50 * FT_TO_M
            self.d_intertrack = (d_tr_min, d_tr_mean, d_tr_max)

        elif yard_type == "perpendicular":
            # You didn't re-state perpendicular formulas in the summary;
            # we keep a consistent structure with your original file.
            d_t_min = 1.5 * n_p * P
            d_t_max = self.A_yard + self.B_yard - 2 * n_p * P
            self.d_truck = (d_t_min, d_t_max)
            self.d_hostler = (d_t_min, d_t_max)

            d_r_min = 0.0
            d_r_max = (
                10 * n_r * FT_TO_M
                + 80 * FT_TO_M
                + self.A_yard + self.B_yard - 2 * n_p * P
            )
            self.d_reposition = (d_r_min, d_r_max)

            n_max = self.train_batch
            d_tr_min = 0.5 * l_c + d_f + d_x + 0.5 * n_p * P + 50 * FT_TO_M
            d_tr_mean = ((n_max - 1) / 2) * l_c + d_f + d_x + 0.5 * n_p * P + 50 * FT_TO_M
            d_tr_max = n_max * l_c + d_f + d_x + 0.5 * n_p * P + 50 * FT_TO_M
            self.d_intertrack = (d_tr_min, d_tr_mean, d_tr_max)

        else:
            raise ValueError("yard_type must be 'parallel' or 'perpendicular'")


    def sample_distance(self, mode: str) -> float:
        """
        Segment sampler.
        mode:
          - "truck"
          - "hostler"
          - "hostler_reposition"
          - "hostler_intertrack"
        """
        if mode == "truck":
            d_min, d_max = self.d_truck
            return uniform(d_min, d_max - d_min).rvs()

        if mode == "hostler":
            d_min, d_max = self.d_hostler
            return uniform(d_min, d_max - d_min).rvs()

        if mode == "hostler_reposition":
            d_min, d_max = self.d_reposition
            return uniform(d_min, d_max - d_min).rvs()

        if mode == "hostler_intertrack":
            d_min, d_mean, d_max = self.d_intertrack
            c = (d_mean - d_min) / (d_max - d_min)
            c = min(max(c, 1e-6), 1 - 1e-6)
            return triang(c, loc=d_min, scale=d_max - d_min).rvs()

        raise ValueError(f"Unknown distance mode: {mode}")

    def compute_travel_time_s(self, distance_m: float, current_veh_num: int, vehicle_type: str) -> float:
        density = current_veh_num / (self.total_lane_length * FT_TO_M)
        speed_mps = speed_density(density, vehicle_type, self.N)
        travel_time_s = distance_m / speed_mps
        return travel_time_s

    def compute_travel_time_hr(self, modes: list[str], current_veh_num: int, vehicle_type: str) -> float:
        """
        Compute travel time by summing per-segment times.
        This is safer than summing distances and dividing once.
        """
        total_seconds = 0.0
        for m in modes:
            dist = self.sample_distance(m)
            total_seconds += self.compute_travel_time_s(dist, current_veh_num, vehicle_type)
        total_hours = total_seconds / 3600
        # print(f"[{vehicle_type}]: {modes} travel time (hr) with {current_veh_num}:{total_hours}")
        return total_hours

    def get_hostler_path(
            self,
            origin_mode: Optional[str],
            destination_mode: Optional[str],
            stage: str
    ) -> List[str]:

        """
        stage:
          - "ic_move": origin_mode must be provided ("train"|"vessel")
          - "oc_move": destination_mode must be provided ("train"|"vessel")
          - "reposition": no mode needed

        Rules:
          - stack -> track: hostler + intertrack
          - stack -> vessel: hostler
          - track -> stack: hostler + intertrack
          - vessel -> stack: hostler
          - reposition: hostler_reposition
        """
        if stage == "reposition":
            return ["hostler_reposition"]

        if stage == "ic_move":
            if origin_mode == "train":
                return ["hostler", "hostler_intertrack"]
            if origin_mode == "vessel":
                return ["hostler"]
            raise ValueError(f"ic_move expects origin_mode train/vessel, got {origin_mode}")

        if stage == "oc_move":
            if destination_mode == "train":
                return ["hostler", "hostler_intertrack"]
            if destination_mode == "vessel":
                return ["hostler"]
            raise ValueError(f"oc_move expects destination_mode train/vessel, got {destination_mode}")

        raise ValueError(f"Unknown stage: {stage}")

    def get_truck_path(self, stage=None):
        return ["truck"]