import os
import re
import pandas as pd

root_dir = '/Users/qianqiantong/PycharmProjects/LIFTS/output/single_track_results'
layout_path = '/Users/qianqiantong/PycharmProjects/LIFTS/input/layout.xlsx'
layout_df = pd.read_excel(layout_path)

CRANE_IDLE_UNIT = 0.02  # gal/hr

results = []

for file_name in os.listdir(root_dir):
    match = re.match(r'(\d+)C-(\d+)H_vehicle_throughput_(\d+)_batch_size_(\d+)\.xlsx', file_name)
    if not match:
        continue

    crane_num = int(match.group(1))
    hostler_num = int(match.group(2))
    throughput = int(match.group(3))
    batch_size = int(match.group(4))
    num_trains = throughput / batch_size / 2

    file_path = os.path.join(root_dir, file_name)
    layout_row = layout_df[layout_df['train batch (k)'] == batch_size * 2]
    if not layout_row.empty:
        M = int(layout_row.iloc[0]['rows (M)'])
        N = int(layout_row.iloc[0]['cols (N)'])
        n_r = int(layout_row.iloc[0]['blocklen (n_r)'])
    else:
        M = N = n_r = None

    xls = pd.ExcelFile(file_path)

    # ---- Helper ----
    def calc_emission(df):
        return df['emission'].sum() if not df.empty else 0

    def calc_avg_metrics(df):
        return (
            df['distance'].mean(skipna=True),
            df['speed'].mean(skipna=True),
            df['density'].mean(skipna=True),
            df['time'].mean(skipna=True)
        )

    # ---- READ DATA ----
    hostler_df = pd.read_excel(xls, 'hostler')
    crane_df = pd.read_excel(xls, 'crane')
    truck_df = pd.read_excel(xls, 'truck')

    # ---- PROCESSED COUNTS ----
    ic_processed = truck_df[truck_df['container_id'].astype(str).str.isnumeric()]['container_id'].nunique()
    oc_processed = hostler_df[hostler_df['container_id'].astype(str).str.startswith("OC-")]['container_id'].nunique()

    # ---- HOSTLER ----
    ic_hostler_df = hostler_df[hostler_df['container_id'].astype(str).str.isnumeric()]
    oc_hostler_df = hostler_df[hostler_df['container_id'].astype(str).str.startswith("OC-")]

    ic_hostler_emission = calc_emission(ic_hostler_df)
    oc_hostler_emission = calc_emission(oc_hostler_df)

    ic_hostler_avg_emission = ic_hostler_emission / ic_processed if ic_processed else 0
    oc_hostler_avg_emission = oc_hostler_emission / oc_processed if oc_processed else 0

    ic_h_dist, ic_h_speed, ic_h_dens, ic_h_time = calc_avg_metrics(ic_hostler_df)
    oc_h_dist, oc_h_speed, oc_h_dens, oc_h_time = calc_avg_metrics(oc_hostler_df)

    # ---- CRANE ----
    ic_crane_df = crane_df[crane_df['container_id'].astype(str).str.isnumeric()]
    oc_crane_df = crane_df[crane_df['container_id'].astype(str).str.startswith("OC-")]

    ic_crane_emission = calc_emission(ic_crane_df)
    oc_crane_emission = calc_emission(oc_crane_df)

    ic_crane_idle = ic_h_time * CRANE_IDLE_UNIT * crane_num if ic_h_time else 0
    oc_crane_idle = oc_h_time * CRANE_IDLE_UNIT * crane_num if oc_h_time else 0

    ic_crane_avg_emission = (ic_crane_emission + ic_crane_idle) / ic_processed if ic_processed else 0
    oc_crane_avg_emission = (oc_crane_emission + oc_crane_idle) / oc_processed if oc_processed else 0

    # ---- TRUCK ----
    ic_truck_df = truck_df[truck_df['container_id'].astype(str).str.isnumeric()]
    oc_truck_df = truck_df[truck_df['container_id'].astype(str).str.startswith("OC-")]

    ic_truck_emission = calc_emission(ic_truck_df)
    oc_truck_emission = calc_emission(oc_truck_df)

    ic_truck_avg_emission = ic_truck_emission / ic_processed if ic_processed else 0
    oc_truck_avg_emission = oc_truck_emission / oc_processed if oc_processed else 0

    ic_t_dist, ic_t_speed, ic_t_dens, ic_t_time = calc_avg_metrics(ic_truck_df)
    oc_t_dist, oc_t_speed, oc_t_dens, oc_t_time = calc_avg_metrics(oc_truck_df)

    # ---- SAVE ----
    results.append({
        'crane': crane_num,
        'hostlers': hostler_num,
        'daily_throughput': throughput,
        'train_batch_size': batch_size,
        'M': M, 'N': N, 'n_r': n_r,
        'num_trains': num_trains,

        'avg_ic_emission(gal/hr)': ic_crane_avg_emission + ic_hostler_avg_emission + ic_truck_avg_emission,
        'avg_oc_emission(gal/hr)': oc_crane_avg_emission + oc_hostler_avg_emission + oc_truck_avg_emission,

        'avg_ic_crane_emission(gal/hr)': ic_crane_avg_emission,
        'avg_oc_crane_emission(gal/hr)': oc_crane_avg_emission,
        'avg_ic_hostler_emission(gal/hr)': ic_hostler_avg_emission,
        'avg_oc_hostler_emission(gal/hr)': oc_hostler_avg_emission,
        'avg_ic_truck_emission(gal/hr)': ic_truck_avg_emission,
        'avg_oc_truck_emission(gal/hr)': oc_truck_avg_emission,

        'ic_hostler_distance(ft)': ic_h_dist,
        'ic_hostler_speed(m/s)': ic_h_speed,
        'ic_hostler_density(veh/m)': ic_h_dens,
        'ic_hostler_travel_time(hr)': ic_h_time,
        'oc_hostler_distance(ft)': oc_h_dist,
        'oc_hostler_speed(m/s)': oc_h_speed,
        'oc_hostler_density(veh/m)': oc_h_dens,
        'oc_hostler_travel_time(hr)': oc_h_time,

        'ic_truck_distance(ft)': ic_t_dist,
        'ic_truck_speed(m/s)': ic_t_speed,
        'ic_truck_density(veh/m)': ic_t_dens,
        'ic_truck_travel_time(hr)': ic_t_time,
        'oc_truck_distance(ft)': oc_t_dist,
        'oc_truck_speed(m/s)': oc_t_speed,
        'oc_truck_density(veh/m)': oc_t_dens,
        'oc_truck_travel_time(hr)': oc_t_time,
    })

    print(f"[Done] Processed {file_name}")

df_results = pd.DataFrame(results)
df_results.to_excel('emission_results.xlsx', index=False)
print("Results saved to emission_results.xlsx")
