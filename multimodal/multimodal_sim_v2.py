import math
import pandas as pd


class NodePerformanceModel:
    """
    NPF container with:
      - 6-modes flow balancing (V,T,H)
      - layout lookup (capacity = 2*vessel_batch) from layout.xlsx
      - geometry -> total_lane_length
      - speed-density -> hostler round-trip time, truck single-trip time
      - headway + crane terms
    """

    FT_TO_M = 1 / 3.28

    def __init__(
        self,
        vessel_batch: int,
        p_v_to_t: float,
        train_batch_size: int,
        horizon_hr: float = 168.0,
        vessel_headway_hr: float = 168.0,
        cranes_per_berth: int = 2,
        berth_number: int = 1,
        cranes_per_track: int = 2,
        track_number: int = 1,
        hostler_number: int = 10,
        track_crane_rate: float = 30.0,  # containers/hour
        dock_crane_rate: float = 35.0,   # containers/hour
        layout_file: str = "/Users/qianqiantong/PycharmProjects/LIFTS/multimodal/input/layout.xlsx",
    ):
        # time concepts
        self.horizon_hr = float(horizon_hr)
        self.vessel_headway_hr = float(vessel_headway_hr)

        # flow inputs
        self.vessel_batch = int(vessel_batch)
        self.p_v_to_t = float(p_v_to_t)
        self.train_batch_size = int(train_batch_size)

        # resources
        self.cranes_per_berth = int(cranes_per_berth)
        self.berth_number = int(berth_number)
        self.cranes_per_track = int(cranes_per_track)
        self.track_number = int(track_number)
        self.hostler_number = int(hostler_number)

        self.total_dock_cranes = self.cranes_per_berth * self.berth_number
        self.total_track_cranes = self.cranes_per_track * self.track_number

        # rates
        self.track_crane_rate = float(track_crane_rate)
        self.dock_crane_rate = float(dock_crane_rate)

        # layout
        self.layout_file = layout_file

        # flows + headway
        self._compute_weekly_flows()
        self.train_headway_hr = self.horizon_hr / max(self.n_trains, 1)

        # layout geometry (depends on vessel_batch -> capacity)
        self._load_layout_geometry()


    # 6-modes flow balancing
    def _compute_weekly_flows(self) -> None:
        V = self.vessel_batch

        # Vessel unload
        self.V_to_T = int(round(V * self.p_v_to_t))
        self.V_to_H = V - self.V_to_T

        # Train capacity (integer trips)
        self.n_trains = int(math.ceil(self.V_to_T / self.train_batch_size))
        self.train_total = self.n_trains * self.train_batch_size

        # Symmetry
        self.T_to_V = self.V_to_T
        self.H_to_V = self.V_to_H

        # Residual balancing on train capacity
        self.H_to_T = self.train_total - self.V_to_T
        self.T_to_H = self.train_total - self.T_to_V

        # Truck arrivals (inbound trucks only per your definition)
        self.truck_total = int(self.H_to_V + self.H_to_T)


    # layout + geometry
    def _load_layout_geometry(self) -> None:
        capacity = 2 * self.vessel_batch  # per your rule
        df = pd.read_excel(self.layout_file)
        row = df.loc[df["train batch (k)"] == capacity]
        if row.empty:
            raise ValueError(f"No layout found for capacity={capacity} in {self.layout_file}")
        row = row.iloc[0]

        self.M = int(row["rows (M)"])
        self.N = int(row["cols (N)"])
        self.n_t = int(row["trainlanes (n_t)"])
        self.n_p = int(row["parklanes (n_p)"])
        self.n_r = int(row["blocklen (n_r)"])

        P = 12 * self.FT_TO_M
        BL_l = 10 * self.n_r * self.FT_TO_M
        BL_w = 80 * self.FT_TO_M

        self.A_yard = self.M * BL_l + (self.M + 1) * self.n_p * P
        self.B_yard = self.N * BL_w + (self.N + 1) * self.n_p * P

        # total lane length (meters)
        self.total_lane_length = (self.N + 1) * self.A_yard + (self.M + 1) * self.B_yard

    # ----------------------------
    # headway terms
    # ----------------------------
    def half_vessel_headway(self) -> float:
        return self.vessel_headway_hr / 2.0

    def half_train_headway(self) -> float:
        return self.train_headway_hr / 2.0

    # ----------------------------
    # crane time terms
    # ----------------------------
    def dock_clearance_time(self, volume: int) -> float:
        cap = max(self.total_dock_cranes, 1) * self.dock_crane_rate  # containers/hr
        return float(volume) / cap  # hr

    def track_clearance_time(self, volume: int) -> float:
        cap = max(self.total_track_cranes, 1) * self.track_crane_rate  # containers/hr
        return float(volume) / cap  # hr

    def dock_service_time_per_container(self) -> float:
        # per-container time when you want 35 containers/hr (as in your bullets)
        return 1.0 / self.dock_crane_rate

    def track_service_time_per_container(self) -> float:
        # per-container time when you want 30 containers/hr (as in your bullets)
        return 1.0 / self.track_crane_rate

    def crane_utilization(self, volume: int, is_dock: bool = True) -> float:
        # clearance ratio form
        if is_dock:
            capacity = max(self.total_dock_cranes, 1) * self.dock_crane_rate
        else:
            capacity = max(self.total_track_cranes, 1) * self.track_crane_rate
        return float(volume) / capacity

    # ----------------------------
    # speed-density + sigma
    # ----------------------------
    @staticmethod
    def ugly_sigma(x: int) -> float:
        total = 0.0
        for i in range(1, x):
            total += 2 * i * (x - i)
        return total / (x**2)

    @staticmethod
    def speed_density(avg_density: float, vehicle_type: str, N: int) -> float:
        # avg_density: veh/m, return m/s
        if vehicle_type == "hostler":
            return 7.5 * math.exp((-1.5 * N - 0.5) * avg_density)
        if vehicle_type == "truck":
            return 10.0 * math.exp((-3.5 * N - 0.5) * avg_density)
        raise ValueError("vehicle_type must be hostler or truck")

    # hostler round trip time (hours)
    # round trip = 2*E[d_single] + E[d_reposition]
    def hostler_travel_time(self) -> float:
        P = 12 * self.FT_TO_M

        # single trip distance range (uniform)
        d_t_min = 0.5 * self.n_p * P
        d_t_max = (self.A_yard / (self.N + 1)) + (self.B_yard / (self.M + 1)) - 2 * self.n_p * P

        # reposition distance range (uniform)
        d_r_min = 5 * self.n_r * self.FT_TO_M + 40 * self.FT_TO_M
        d_r_max = (
            self.ugly_sigma(self.M) * (10 * self.n_r * self.FT_TO_M + self.n_p * P)
            + self.ugly_sigma(self.N) * (80 * self.FT_TO_M + self.n_p * P)
        )

        E_loaded = 0.5 * (d_t_min + d_t_max)
        E_rep = 0.5 * (d_r_min + d_r_max)

        total_distance_m = 2.0 * E_loaded + E_rep

        density = self.hostler_number / self.total_lane_length
        speed_mps = self.speed_density(density, "hostler", self.N)

        return (total_distance_m / speed_mps) / 3600.0

    # ----------------------------
    # truck single trip time (hours)
    # single trip = E[d_single] / v(density)
    # ----------------------------
    def truck_single_trip_time(self) -> float:
        P = 12 * self.FT_TO_M

        d_t_min = 0.5 * self.n_p * P
        d_t_max = (self.A_yard / (self.N + 1)) + (self.B_yard / (self.M + 1)) - 2 * self.n_p * P

        E_trip = 0.5 * (d_t_min + d_t_max)

        density = self.truck_total / self.total_lane_length
        speed_mps = self.speed_density(density, "truck", self.N)

        return (E_trip / speed_mps) / 3600.0

    # ----------------------------
    # 6 OD performance functions (hours)
    # ----------------------------
    def perf_VT(self) -> float:
        # Vessel -> Train
        return (
            self.half_vessel_headway()
            + self.dock_clearance_time(self.vessel_batch)
            + self.hostler_travel_time()
            + self.track_clearance_time(self.vessel_batch)
        )

    def perf_TV(self) -> float:
        # Train -> Vessel
        return (
            self.half_train_headway()
            + self.hostler_travel_time()
            + self.track_service_time_per_container()
            + self.dock_clearance_time(self.vessel_batch)
            + self.half_vessel_headway()
        )

    def perf_TH(self) -> float:
        # Train -> Truck
        return (
            self.track_service_time_per_container()
            + self.half_train_headway()
            + 0.5 * self.hostler_travel_time()
            + self.truck_single_trip_time()
        )

    def perf_HT(self) -> float:
        # Truck -> Train (use train headway, not vessel headway)
        return (
            self.track_service_time_per_container()
            + 0.5 * self.hostler_travel_time()
            + self.half_train_headway()
        )

    def perf_VH(self) -> float:
        # Vessel -> Truck
        return (
            self.half_vessel_headway()
            + 0.5 * self.hostler_travel_time()
            + self.dock_service_time_per_container()
            + self.truck_single_trip_time()
        )

    def perf_HV(self) -> float:
        # Truck -> Vessel
        return (
            self.dock_service_time_per_container()
            + 0.5 * self.hostler_travel_time()
            + self.half_vessel_headway()
        )


def run_grid_and_export_excel(
    output_excel: str = "/Users/qianqiantong/PycharmProjects/LIFTS/multimodal/output/npf.xlsx",
    layout_file: str = "/Users/qianqiantong/PycharmProjects/LIFTS/multimodal/input/layout.xlsx",
):

    CRANES_PER_BERTH_LIST = list(range(2, 6, 1))
    CRANES_PER_TRACK_LIST = list(range(2, 11, 1))
    VESSEL_BATCH_LIST = list(range(1100, 1101, 100))
    Vessel_to_train_split = [round(x, 1) for x in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]


    HORIZON_HR = 168.0
    VESSEL_HEADWAY_HR = 168.0
    TRAIN_BATCH_SIZE = 100

    BERTH_NUMBER = 1
    TRACK_NUMBER = 1
    HOSTLER_NUMBER = 100  # set your value here

    TRACK_CRANE_RATE = 30.0
    DOCK_CRANE_RATE = 35.0

    rows = []

    for cpb in CRANES_PER_BERTH_LIST:
        for cpt in CRANES_PER_TRACK_LIST:
            for vessel_batch in VESSEL_BATCH_LIST:
                for split in Vessel_to_train_split:
                    try:
                        m = NodePerformanceModel(
                            vessel_batch=vessel_batch,
                            p_v_to_t=split,
                            train_batch_size=TRAIN_BATCH_SIZE,
                            horizon_hr=HORIZON_HR,
                            vessel_headway_hr=VESSEL_HEADWAY_HR,
                            cranes_per_berth=cpb,
                            berth_number=BERTH_NUMBER,
                            cranes_per_track=cpt,
                            track_number=TRACK_NUMBER,
                            hostler_number=HOSTLER_NUMBER,
                            track_crane_rate=TRACK_CRANE_RATE,
                            dock_crane_rate=DOCK_CRANE_RATE,
                            layout_file=layout_file,
                        )

                        # flows
                        flows = {
                            "V_to_T": m.V_to_T,
                            "V_to_H": m.V_to_H,
                            "T_to_V": m.T_to_V,
                            "T_to_H": m.T_to_H,
                            "H_to_V": m.H_to_V,
                            "H_to_T": m.H_to_T,
                            "n_trains": m.n_trains,
                            "train_total": m.train_total,
                            "truck_total": m.truck_total,
                        }

                        # performance times (hours)
                        perf = {
                            "time_V_T_hr": m.perf_VT(),
                            "time_T_V_hr": m.perf_TV(),
                            "time_T_H_hr": m.perf_TH(),
                            "time_H_T_hr": m.perf_HT(),
                            "time_V_H_hr": m.perf_VH(),
                            "time_H_V_hr": m.perf_HV(),
                        }

                        # aux = {
                        #     "hostler_round_trip_hr": m.hostler_travel_time(),
                        #     "truck_single_trip_hr": m.truck_single_trip_time(),
                        #     "half_vessel_headway_hr": m.half_vessel_headway(),
                        #     "half_train_headway_hr": m.half_train_headway(),
                        #     "dock_clearance_hr": m.dock_clearance_time(m.vessel_batch),
                        #     "track_clearance_hr": m.track_clearance_time(m.vessel_batch),
                        #     "rho_dock_clearance": m.crane_utilization(m.vessel_batch, is_dock=True),
                        #     "rho_track_clearance": m.crane_utilization(m.vessel_batch, is_dock=False),
                        # }

                        rows.append({
                            # inputs
                            "cranes_per_berth": cpb,
                            "cranes_per_track": cpt,
                            "berth_number": BERTH_NUMBER,
                            "track_number": TRACK_NUMBER,
                            "hostler_number": HOSTLER_NUMBER,
                            "vessel_batch": vessel_batch,
                            "p_v_to_t": split,
                            "train_batch_size": TRAIN_BATCH_SIZE,
                            "horizon_hr": HORIZON_HR,
                            "vessel_headway_hr": VESSEL_HEADWAY_HR,

                            # # layout picked from excel
                            # "layout_capacity": 2 * vessel_batch,
                            # "M": m.M,
                            # "N": m.N,
                            # "n_t": m.n_t,
                            # "n_p": m.n_p,
                            # "n_r": m.n_r,
                            # "A_yard_m": m.A_yard,
                            # "B_yard_m": m.B_yard,
                            # "total_lane_length_m": m.total_lane_length,

                            # flows + times
                            **flows,
                            **perf,
                            # **aux,
                        })

                    except Exception as e:
                        # keep row with error info (so you can see missing capacities/layout issues)
                        rows.append({
                            "cranes_per_berth": cpb,
                            "cranes_per_track": cpt,
                            "berth_number": BERTH_NUMBER,
                            "track_number": TRACK_NUMBER,
                            "hostler_number": HOSTLER_NUMBER,
                            "vessel_batch": vessel_batch,
                            "p_v_to_t": split,
                            "train_batch_size": TRAIN_BATCH_SIZE,
                            "horizon_hr": HORIZON_HR,
                            "vessel_headway_hr": VESSEL_HEADWAY_HR,
                            "layout_capacity": 2 * vessel_batch,
                            "error": str(e),
                        })

    df_out = pd.DataFrame(rows)

    # write to excel
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="npf_grid")

    print(f"Saved: {output_excel}")
    print(f"Rows: {len(df_out)}")


if __name__ == "__main__":
    run_grid_and_export_excel()