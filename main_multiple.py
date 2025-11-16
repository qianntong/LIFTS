import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import itertools
import time
from typing import Dict, List
import polars as pl
import sys
import shutil
import multi_train_demo as sim
import traceback

"""
Simulation parameters:
1. Number of tracks => [2, 3, 4, 5]
2. Cranes per track => [2, 3, 4]
3. Number of hostlers => [12, 24, 36]
4. Batch size: 20, 40, 60, ..., 200 => list(range(20, 201, 20))
5. Number of trains: 1, 2, 3, ..., 10

Simulation running:
SIMUL_OPTION: option 1: Run all combinations; option 2: Run limited number for testing
MAX_RUN_FOR_TEST: int input - if option 2, specify how many cases to test
"""
SIMULATION_PARAMS = {
    'track_number': [2],
    'cranes_per_track': [2],
    'hostler_number': [6, 12],
    'train_batch_size': [100],
    'train_number': list(range(10, 51, 10))
}

SIMUL_OPTION = 1
MAX_RUN_FOR_TEST = 3


"""
Please don't change any logic/codes below!!!!
Note: IC Delay Time = OC Delay Time
"""

CONFIG_PATH = Path("input/config.yaml")
OUTPUT_DIR = Path("output/batch_results")
BACKUP_DIR = Path("input/config_backup")

def generate_param_combinations(param_dict: Dict) -> List[Dict]:
    keys = list(param_dict.keys())
    values = list(param_dict.values())

    combinations = []
    for combo in itertools.product(*values):
        param_set = dict(zip(keys, combo))
        combinations.append(param_set)

    return combinations


def update_config_fields(config: dict, params: Dict) -> dict:
    if "train_number" in params:
        config["simulation"]["train_number"] = params["train_number"]
    if "train_batch_size" in params:
        config["simulation"]["train_batch_size"] = params["train_batch_size"]
    if "track_number" in params:
        config["yard"]["track_number"] = params["track_number"]
    if "cranes_per_track" in params:
        config["terminal"]["cranes_per_track"] = params["cranes_per_track"]
    if "hostler_number" in params:
        config["terminal"]["hostler_number"] = params["hostler_number"]

    return config


def collect_results(container_data, updated_config: Dict, params: Dict, run_id: int, analyze_start: float = None, analyze_end: float = None):
    """
    Collect and analyze simulation results from container_data (Polars DataFrame).

    Args:
        container_data: Polars DataFrame returned by simulation
        updated_config: Simulation configuration dictionary
        params: Parameter dictionary
        run_id: Run identifier
        analyze_start: Start time for analysis window (hours)
        analyze_end: End time for analysis window (hours)

    Returns:
        results_dict, summary_dict
    """
    try:
        if container_data is None or container_data.height == 0:
            print(f"    No container data found for run {run_id}")
            return {
                'run_id': run_id,
                'status': 'no_data',
                **params,
                'analyze_start': analyze_start,
                'analyze_end': analyze_end,
            }, {}

        # --- Classify IC & OC ---
        df = container_data.to_pandas()
        if 'container_id' in df.columns:
            df['type'] = df['container_id'].astype(str).apply(lambda x: 'IC' if x.startswith('IC-') else ('OC' if x.startswith('OC-') else None))
        else:
            raise KeyError("Missing 'container_id' column in container_data.")

        # --- Apply Time Window Filter ---
        if analyze_start is not None and analyze_end is not None:
            ic_mask = (df['type'] == 'IC') & (df['train_arrival_actual'].notna())
            oc_mask = (df['type'] == 'OC') & (df['train_depart'].notna())

            ic_in_window = ((df.loc[ic_mask, 'train_arrival_actual'] >= analyze_start) &  (df.loc[ic_mask, 'train_arrival_actual'] <= analyze_end))
            oc_in_window = ((df.loc[oc_mask, 'train_depart'] >= analyze_start) & (df.loc[oc_mask, 'train_depart'] <= analyze_end))

            df['in_window'] = False
            df.loc[ic_mask, 'in_window'] = ic_in_window
            df.loc[oc_mask, 'in_window'] = oc_in_window
            df = df[df['in_window'] == True].copy()
            print(f"    Containers after time window filtering [{analyze_start}, {analyze_end}]: {len(df)}")
        else:
            print(f"    No time window filter applied - using all containers")

        # --- IC Processing & Delay Time ---
        ic_df = df[df['type'] == 'IC'].copy()

        # IC Processing Time
        ic_df['ic_processing_time'] = ic_df['truck_exit'] - ic_df['train_arrival_actual']
        ic_processing_valid = ic_df['ic_processing_time'].dropna()

        # IC Delay Time
        if 'train_arrival_actual' in ic_df.columns:
            ic_df['ic_delay_time'] = ic_df['train_arrival_actual'] - ic_df['train_arrival_expected']
            ic_delay_valid = ic_df['ic_delay_time'].dropna()
        else:
            ic_delay_valid = pd.Series(dtype=float)
            print("    Warning: Missing 'train_arrival_actual' for IC delay calculation")

        print(f"    IC containers: {len(ic_df)} | proc_valid={len(ic_processing_valid)} | delay_valid={len(ic_delay_valid)}")

        # --- OC Processing & Delay Time ---
        oc_df = df[df['type'] == 'OC'].copy()

        # OC Processing Time
        if {'train_depart', 'train_arrival_actual_oc'}.issubset(oc_df.columns):
            oc_df['oc_processing_time'] = oc_df['train_depart'] - oc_df['train_arrival_actual_oc']
            oc_processing_valid = oc_df['oc_processing_time'].dropna()
            print(f"OC processing time calculated for {len(oc_processing_valid)} containers")
        else:
            oc_processing_valid = pd.Series(dtype=float)
            missing_cols = {'train_depart', 'train_arrival_actual_oc'} - set(oc_df.columns)
            print(f"Missing {missing_cols} for OC processing time calculation")

        # OC Delay Time
        if {'train_arrival_actual_oc', 'train_arrival_expected_oc'}.issubset(oc_df.columns):
            oc_df['oc_delay_time'] = oc_df['train_arrival_actual_oc'] - oc_df['train_arrival_expected_oc']
            oc_delay_valid = oc_df['oc_delay_time'].dropna()
            print(f"OC delay time calculated for {len(oc_delay_valid)} containers")
        else:
            oc_delay_valid = pd.Series(dtype=float)
            missing_cols = {'train_arrival_actual_oc', 'train_arrival_expected_oc'} - set(oc_df.columns)
            print(f"Missing {missing_cols} for OC delay time calculation")

        print(f"OC containers: {len(oc_df)} | proc_valid={len(oc_processing_valid)} | delay_valid={len(oc_delay_valid)}")

        # ============ Results Statistics ============
        def calc_stats(series, prefix):
            if len(series) == 0:
                return {
                    f'{prefix}_mean': None,
                    f'{prefix}_min': None,
                    f'{prefix}_max': None,
                    f'{prefix}_std': None,
                }
            return {
                f'{prefix}_mean': round(series.mean(), 4),
                f'{prefix}_min': round(series.min(), 4),
                f'{prefix}_max': round(series.max(), 4),
                f'{prefix}_std': round(series.std(), 4),
            }

        # ============ Compile Results ============
        results_dict = {
            'run_id': run_id,
            'status': 'success',
            'track_number': updated_config["yard"]["track_number"],
            'cranes_per_track': updated_config["terminal"]["cranes_per_track"],
            'hostler_number': updated_config["terminal"]["hostler_number"],
            'train_batch_size': updated_config["simulation"]["train_batch_size"],
            'trains_per_day': updated_config["simulation"]["train_number"],
            **calc_stats(ic_processing_valid, 'ic_processing_time'),
            **calc_stats(ic_delay_valid, 'ic_delay_time'),
            **calc_stats(oc_processing_valid, 'oc_processing_time'),
            **calc_stats(oc_delay_valid, 'oc_delay_time'),
        }

        summary_dict = {
            'track_number': updated_config["yard"]["track_number"],
            'cranes_per_track': updated_config["terminal"]["cranes_per_track"],
            'hostler_number': updated_config["terminal"]["hostler_number"],
            'train_batch_size': updated_config["simulation"]["train_batch_size"],
            'trains_per_day': updated_config["simulation"]["train_number"],
            'avg_ic_processing_time': results_dict['ic_processing_time_mean'],
            'avg_ic_delay_time': results_dict['ic_delay_time_mean'],
            'avg_oc_processing_time': results_dict['oc_processing_time_mean'],
            'avg_oc_delay_time': results_dict['oc_delay_time_mean'],
        }

        print(f"    Done! Results collected successfully for run {run_id}")
        return results_dict, summary_dict

    except Exception as e:
        print(f"    Error in collect_results for run {run_id}: {str(e)}")
        traceback.print_exc()
        return {
            'run_id': run_id,
            'status': 'error',
            **params,
            'error': str(e)
        }, {}


def run_single_simulation(params: Dict, run_id: int) -> Dict:
    print(f"\n{'=' * 70}")
    print(f"Run # {run_id}")
    print(f"  Tracks: {params['track_number']}, "
          f"Cranes/Track: {params['cranes_per_track']}, "
          f"Hostlers: {params['hostler_number']}")
    print(f"  Batch Size: {params['train_batch_size']}, "
          f"Trains: {params['train_number']}")
    print(f"{'=' * 70}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    updated_config = update_config_fields(config, params)
    observation_start_time = updated_config["simulation"]["analyze_start"]
    observation_end_time = updated_config["simulation"]["analyze_end"]

    updated_path = BACKUP_DIR / f"config_run_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(updated_path, "w") as f:
        yaml.safe_dump(updated_config, f, sort_keys=False)
    print(f"Updated config saved to: {updated_path}")

    shutil.copy2(updated_path, CONFIG_PATH)

    modules_to_reload = ['multi_train_demo', 'train_decouple', 'state', 'distance', 'vehicle']
    for mod in modules_to_reload:
        if mod in sys.modules:
            del sys.modules[mod]

    try:
        container_data = sim.run_simulation(
            train_consist_plan=pl.read_csv(Path('input') / 'train_consist_plan.csv'),
            terminal="Allouez",
            out_path=None
        )

        results_dict, summary_dict = collect_results(container_data, updated_config, params, run_id, observation_start_time, observation_end_time)
        print(f"Run #{run_id} completed successfully")

        return results_dict, summary_dict

    except Exception as e:
        print(f"Run #{run_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'run_id': run_id,
            'status': 'failed',
            'error': str(e),
            **params,
            'avg_ic_processing_time': None,
            'avg_oc_processing_time': None,
            'total_containers': 0
        }


def run_batch_simulations(param_dict: Dict = None, max_runs: int = None, resume_from: int = None) -> pd.DataFrame:
    """
    Args:
        param_dict: Parameter dictionary (uses SIMULATION_PARAMS if None)
        max_runs: Maximum number of runs (runs all if None)
        resume_from: Resume from specific run number (starts from 1 if None)
    Returns:
        pd.DataFrame: Results dataframe
    """
    if param_dict is None:
        param_dict = SIMULATION_PARAMS

    # Input files prepare
    param_combinations = generate_param_combinations(param_dict)
    total_runs = len(param_combinations)

    print(f"\n{'=' * 70}")
    print(f"BATCH SIMULATION STARTED")
    print(f"{'=' * 70}")
    print(f"Total parameter combinations: {total_runs}")
    print(f"Estimated total runs: {min(max_runs, total_runs) if max_runs else total_runs}")

    # Calculate estimated time
    avg_time_per_run = 10  # seconds (rough estimate)
    estimated_total_time = (min(max_runs, total_runs) if max_runs else total_runs) * avg_time_per_run
    print(f"Estimated time: ~{estimated_total_time / 60:.1f} minutes")
    print(f"{'=' * 70}\n")

    # Run simulations
    results = []
    summaries = []
    start_time = time.time()

    start_idx = (resume_from - 1) if resume_from else 0
    end_idx = min(start_idx + max_runs, total_runs) if max_runs else total_runs

    for i in range(start_idx, end_idx):
        run_id = i + 1
        params = param_combinations[i]

        run_start = time.time()
        result, summary = run_single_simulation(params, run_id)
        run_time = time.time() - run_start

        result['run_time_seconds'] = round(run_time, 2)

        results.append(result)
        summaries.append(summary)

        progress = ((i - start_idx + 1) / (end_idx - start_idx)) * 100
        elapsed = time.time() - start_time
        avg_time = elapsed / (i - start_idx + 1)
        remaining = avg_time * (end_idx - i - 1)

        print(f"\nProgress: {progress:.1f}% ({i - start_idx + 1}/{end_idx - start_idx})")
        print(f"Elapsed: {elapsed / 60:.1f} min | Remaining: ~{remaining / 60:.1f} min")

        if (i - start_idx + 1) % 10 == 0:
            df = pd.DataFrame(results)
            df_summary = pd.DataFrame(summaries)
            save_results(df, df_summary, suffix='_intermediate')
            print(f"Intermediate results saved")

    results_df = pd.DataFrame(results)
    summary_df = pd.DataFrame(summaries)

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"BATCH SIMULATION COMPLETED")
    print(f"{'=' * 70}")
    print(f"Total runs: {len(results)}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"{'=' * 70}\n")

    return results_df, summary_df


def save_results(results_df: pd.DataFrame, summary_df: pd.DataFrame, suffix: str = ''):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f"multiple_track_results{suffix}_{timestamp}.xlsx"

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: Summary for Avg
        summary_df.to_excel(writer, sheet_name='Summary_Avg_Results', index=False)

        # Sheet 2: Full Results
        results_df.to_excel(writer, sheet_name='Full_Results', index=False)

        # Sheet 3: Summary Statistics
        summary_dict = summarize_experiment_results(results_df.to_dict('records'))
        summary_df = pd.DataFrame([summary_dict])
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)

    print(f"Results saved to: {output_file}")
    return output_file


def summarize_experiment_results(all_results: List[Dict]) -> Dict:
    """
    Summarize results from multiple simulation runs
    """
    df = pd.DataFrame(all_results)
    successful_runs = df[df['status'] == 'success']

    if len(successful_runs) == 0:
        print("No successful runs to summarize!")
        return {}

    summary = {
        # Run Information
        'total_runs': len(df),
        'successful_runs': len(successful_runs),
        'failed_runs': len(df) - len(successful_runs),

        # IC Processing Time
        'Avg IC Processing Time (hrs)': successful_runs['ic_processing_time_mean'].mean(),
        'Min IC Processing Time (hrs)': successful_runs['ic_processing_time_min'].min(),
        'Max IC Processing Time (hrs)': successful_runs['ic_processing_time_max'].max(),
        'Std IC Processing Time (hrs)': successful_runs['ic_processing_time_std'].mean(),

        # IC Delay Time
        'Avg IC Delay (hrs)': successful_runs['ic_delay_time_mean'].mean(),
        'Min IC Delay (hrs)': successful_runs['ic_delay_time_min'].min(),
        'Max IC Delay (hrs)': successful_runs['ic_delay_time_max'].max(),
        'Std IC Delay (hrs)': successful_runs['ic_delay_time_std'].mean(),

        # OC Processing Time
        'Avg OC Processing Time (hrs)': successful_runs['oc_processing_time_mean'].mean(),
        'Min OC Processing Time (hrs)': successful_runs['oc_processing_time_min'].min(),
        'Max OC Processing Time (hrs)': successful_runs['oc_processing_time_max'].max(),
        'Std OC Processing Time (hrs)': successful_runs['oc_processing_time_std'].mean(),

        # OC Delay Time
        'Avg OC Delay (hrs)': successful_runs['oc_delay_time_mean'].mean(),
        'Min OC Delay (hrs)': successful_runs['oc_delay_time_min'].min(),
        'Max OC Delay (hrs)': successful_runs['oc_delay_time_max'].max(),
        'Std OC Delay (hrs)': successful_runs['oc_delay_time_std'].mean(),
    }

    # Round all numeric values
    for key, value in summary.items():
        if isinstance(value, (float, np.float64)):
            summary[key] = round(value, 4)

    return summary


def main():
    print("\n" + "=" * 70)
    print("INTERMODAL TERMINAL SIMULATION (MULTIPLE TRACK) - BATCH RUNNER")
    print("=" * 70 + "\n")

    if SIMUL_OPTION == 1:
        print("    Running in ALL COMBO")
        results_df, summary_df = run_batch_simulations()

    if SIMUL_OPTION == 2:
        print(f"    Running in TEST MODE (first {MAX_RUN_FOR_TEST} combinations)")
        results_df, summary_df  = run_batch_simulations(max_runs = MAX_RUN_FOR_TEST)

    # save
    save_results(results_df, summary_df)


if __name__ == "__main__":
    main()