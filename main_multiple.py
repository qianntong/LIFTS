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
from main import train_batch_size

SIMULATION_PARAMS = {
    'track_number': [2, 3],             # Number of tracks => [2, 3, 4, 5]
    'cranes_per_track': [2, 3, 4],      # Cranes per track => [2, 3, 4]
    'hostler_number': [12, 24, 36],     # Number of hostlers => [12, 24, 36]
    'train_batch_size': [20],           # Batch size: 20, 40, 60, ..., 200 => list(range(20, 201, 20))
    'train_number': list(range(1, 11))  # Number of trains: 1, 2, 3, ..., 10
}

# File paths
CONFIG_PATH = Path("input/config.yaml")
OUTPUT_DIR = Path("output/batch_results")
BACKUP_DIR = Path("input/config_backup")


def backup_config(config_path: Path):
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"

    shutil.copy2(config_path, backup_path)
    print(f"Config backed up to: {backup_path}")
    return backup_path


def run_single_simulation(params: Dict, run_id: int) -> Dict:
    print(f"\n{'=' * 70}")
    print(f"Run # {run_id}")
    print(f"  Tracks: {params['track_number']}, "
          f"Cranes/Track: {params['cranes_per_track']}, "
          f"Hostlers: {params['hostler_number']}")
    print(f"  Batch Size: {params['train_batch_size']}, "
          f"Trains: {params['train_number']}")
    print(f"{'=' * 70}")

    # Update config file
    backup_path = backup_config(CONFIG_PATH)
    with open(backup_path, "r") as f:
        config = yaml.safe_load(f)

    # Remove old module cache
    modules_to_reload = ['multi_train_demo', 'train_decouple', 'state', 'distance', 'vehicle']
    for mod in modules_to_reload:
        if mod in sys.modules:
            del sys.modules[mod]

    try:
        # Run simulation
        sim.run_simulation(
            train_consist_plan=pl.read_csv(Path('input') / 'train_consist_plan.csv'),
            terminal="Allouez",
            out_path=None  # Don't save individual Excel files
        )

        # Collect results from state
        results_dict = collect_results(sim.state, params, run_id)

        print(f"Run #{run_id} completed successfully")
        return results_dict

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


def collect_results(state, params: Dict, run_id: int, analyze_start: float = None, analyze_end: float = None) -> Dict:
    """
    Collect and analyze simulation results with detailed statistics.

    Args:
        state: Simulation state object
        params: Parameter dictionary
        run_id: Run identifier
        analyze_start: Start time for analysis window (hours)
        analyze_end: End time for analysis window (hours)

    Returns:
        Dict: Comprehensive results including means, mins, maxs, and standard deviations
    """
    try:
        container_events = state.container_events

        if not container_events:
            print(f"    No container events found for run {run_id}")
            return {
                'run_id': run_id,
                'status': 'no_data',
                **params,
                'analyze_start': analyze_start,
                'analyze_end': analyze_end,
            }

        # Convert container events to DataFrame for easier processing
        container_records = []
        for container_id, events in container_events.items():
            record = {'container_id': container_id, **events}

            # Determine container type
            if 'IC-' in container_id:
                record['type'] = 'IC'
            elif 'OC-' in container_id:
                record['type'] = 'OC'
            else:
                continue

            container_records.append(record)

        if not container_records:
            print(f"    No valid container records for run {run_id}")
            return {
                'run_id': run_id,
                'status': 'no_data',
                **params,
            }

        df = pd.DataFrame(container_records)

        # Apply time window filter if specified
        if analyze_start is not None and analyze_end is not None:
            # Filter IC containers by train arrival time
            ic_mask = (df['type'] == 'IC') & (df['train_arrival_expected'].notna())
            df.loc[ic_mask, 'in_window'] = (
                    (df.loc[ic_mask, 'train_arrival_expected'] >= analyze_start) &
                    (df.loc[ic_mask, 'train_arrival_expected'] <= analyze_end)
            )

            # Filter OC containers by truck arrival time
            oc_mask = (df['type'] == 'OC') & (df['truck_arrival'].notna())
            df.loc[oc_mask, 'in_window'] = (
                    (df.loc[oc_mask, 'truck_arrival'] >= analyze_start) &
                    (df.loc[oc_mask, 'truck_arrival'] <= analyze_end)
            )

            df = df[df['in_window'] == True].copy()

        # ============ IC Container Processing ============
        ic_df = df[df['type'] == 'IC'].copy()

        # IC Processing Time: train_arrival_expected -> truck_exit
        ic_df['ic_processing_time'] = ic_df['truck_exit'] - ic_df['train_arrival_expected']
        ic_processing_valid = ic_df['ic_processing_time'].dropna()

        # IC Delay Time: train_arrival - train_arrival_expected
        ic_df['ic_delay_time'] = ic_df['train_arrival'] - ic_df['train_arrival_expected']
        ic_delay_valid = ic_df['ic_delay_time'].dropna()

        # ============ OC Container Processing ============
        oc_df = df[df['type'] == 'OC'].copy()

        # Calculate first OC pickup time per train
        if 'train_depart' in oc_df.columns and 'hostler_pickup' in oc_df.columns:
            first_pickup_per_train = (
                oc_df.groupby('train_depart')['hostler_pickup']
                .min()
                .reset_index()
                .rename(columns={'hostler_pickup': 'first_oc_pickup_time'})
            )

            # Merge back to oc_df
            oc_df = oc_df.merge(first_pickup_per_train, on='train_depart', how='left')

            # OC Delay Time: train_depart - first_oc_pickup_time
            oc_df['oc_delay_time'] = oc_df['train_depart'] - oc_df['first_oc_pickup_time']
            oc_delay_valid = oc_df['oc_delay_time'].dropna()
        else:
            oc_delay_valid = pd.Series(dtype=float)

        # OC Processing Time: truck_arrival -> crane_load
        oc_df['oc_processing_time'] = oc_df['crane_load'] - oc_df['truck_arrival']
        oc_processing_valid = oc_df['oc_processing_time'].dropna()

        # ============ Train-level Statistics ============
        train_times = []
        train_delays = []

        if hasattr(state, 'time_per_train'):
            for train_id, proc_time in state.time_per_train.items():
                # Apply time filter if train_arrival_times exists
                if hasattr(state, 'train_arrival_times') and train_id in state.train_arrival_times:
                    arrival_time = state.train_arrival_times[train_id]
                    if analyze_start is not None and analyze_end is not None:
                        if not (analyze_start <= arrival_time <= analyze_end):
                            continue
                train_times.append(proc_time)

        if hasattr(state, 'train_delay_time'):
            for train_id, delay in state.train_delay_time.items():
                if hasattr(state, 'train_arrival_times') and train_id in state.train_arrival_times:
                    arrival_time = state.train_arrival_times[train_id]
                    if analyze_start is not None and analyze_end is not None:
                        if not (analyze_start <= arrival_time <= analyze_end):
                            continue
                train_delays.append(delay)

        # ============ Helper Function for Statistics ============
        def calc_stats(series, prefix):
            """Calculate mean, min, max, std for a series"""
            if len(series) == 0:
                return {
                    f'{prefix}_mean': None,
                    f'{prefix}_min': None,
                    f'{prefix}_max': None,
                    f'{prefix}_std': None,
                    f'{prefix}_count': 0
                }
            return {
                f'{prefix}_mean': round(series.mean(), 4),
                f'{prefix}_min': round(series.min(), 4),
                f'{prefix}_max': round(series.max(), 4),
                f'{prefix}_std': round(series.std(), 4),
                f'{prefix}_count': len(series)
            }

        # ============ Compile Results ============
        results = {
            'run_id': run_id,
            'status': 'success',
            **params,
            'analyze_start': analyze_start,
            'analyze_end': analyze_end,

            # IC Processing Time Statistics
            **calc_stats(ic_processing_valid, 'ic_processing_time'),

            # IC Delay Time Statistics
            **calc_stats(ic_delay_valid, 'ic_delay_time'),

            # OC Processing Time Statistics
            **calc_stats(oc_processing_valid, 'oc_processing_time'),

            # OC Delay Time Statistics
            **calc_stats(oc_delay_valid, 'oc_delay_time'),

            # Train Processing Time Statistics
            **calc_stats(pd.Series(train_times), 'train_processing_time'),

            # Train Delay Statistics
            **calc_stats(pd.Series(train_delays), 'train_delay'),

            # Container Counts
            'total_ic_containers': len(ic_df),
            'total_oc_containers': len(oc_df),
            'total_containers': len(df),

            # Train Counts
            'num_trains_processed': len(train_times),
            'num_trains_delayed': len(train_delays),
        }

        # Add backward compatibility (keeping old field names)
        results['avg_ic_processing_time'] = results['ic_processing_time_mean']
        results['avg_ic_delay_time'] = results['ic_delay_time_mean']
        results['avg_oc_processing_time'] = results['oc_processing_time_mean']
        results['avg_oc_delay_time'] = results['oc_delay_time_mean']
        results['avg_train_processing_time'] = results['train_processing_time_mean']

        return results

    except Exception as e:
        print(f"    Error in collect_results for run {run_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'run_id': run_id,
            'status': 'error',
            **params,
            'error': str(e)
        }


def generate_param_combinations(param_dict: Dict) -> List[Dict]:
    keys = list(param_dict.keys())
    values = list(param_dict.values())

    combinations = []
    for combo in itertools.product(*values):
        param_set = dict(zip(keys, combo))
        combinations.append(param_set)

    return combinations


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

    # Generate all parameter combinations
    param_combinations = generate_param_combinations(param_dict)
    total_runs = len(param_combinations)

    print(f"\n{'=' * 70}")
    print(f"BATCH SIMULATION STARTED")
    print(f"{'=' * 70}")
    print(f"Total parameter combinations: {total_runs}")
    print(f"Estimated total runs: {min(max_runs, total_runs) if max_runs else total_runs}")

    # Calculate estimated time
    avg_time_per_run = 5  # seconds (rough estimate)
    estimated_total_time = (min(max_runs, total_runs) if max_runs else total_runs) * avg_time_per_run
    print(f"Estimated time: ~{estimated_total_time / 60:.1f} minutes")
    print(f"{'=' * 70}\n")

    # Backup original config
    backup_config(CONFIG_PATH)

    # Run simulations
    results = []
    start_time = time.time()

    start_idx = (resume_from - 1) if resume_from else 0
    end_idx = min(start_idx + max_runs, total_runs) if max_runs else total_runs

    for i in range(start_idx, end_idx):
        run_id = i + 1
        params = param_combinations[i]

        run_start = time.time()
        result = run_single_simulation(params, run_id)
        run_time = time.time() - run_start

        result['run_time_seconds'] = round(run_time, 2)
        results.append(result)

        # Progress report
        progress = ((i - start_idx + 1) / (end_idx - start_idx)) * 100
        elapsed = time.time() - start_time
        avg_time = elapsed / (i - start_idx + 1)
        remaining = avg_time * (end_idx - i - 1)

        print(f"\nProgress: {progress:.1f}% ({i - start_idx + 1}/{end_idx - start_idx})")
        print(f"Elapsed: {elapsed / 60:.1f} min | Remaining: ~{remaining / 60:.1f} min")

        # Save intermediate results every 10 runs
        if (i - start_idx + 1) % 10 == 0:
            df = pd.DataFrame(results)
            save_results(df, suffix='_intermediate')
            print(f"Intermediate results saved")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Total time
    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"BATCH SIMULATION COMPLETED")
    print(f"{'=' * 70}")
    print(f"Total runs: {len(results)}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"{'=' * 70}\n")

    return results_df


def save_results(results_df: pd.DataFrame, suffix: str = ''):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f"multiple_track_results{suffix}_{timestamp}.xlsx"

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: Full results
        results_df.to_excel(writer, sheet_name='Full_Results', index=False)

        # Sheet 2: Summary statistics
        summary = generate_summary_statistics(results_df)
        summary.to_excel(writer, sheet_name='Summary_Statistics')

        # Sheet 3: Failed runs (if any)
        failed = results_df[results_df['status'] != 'success']
        if not failed.empty:
            failed.to_excel(writer, sheet_name='Failed_Runs', index=False)

    print(f"Results saved to: {output_file}")
    return output_file


def generate_summary_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    successful_runs = results_df[results_df['status'] == 'success']

    if successful_runs.empty:
        return pd.DataFrame({'Message': ['No successful runs to analyze']})

    summary = {
        'Total Runs': len(results_df),
        'Successful Runs': len(successful_runs),
        'Failed Runs': len(results_df) - len(successful_runs),
        'Avg IC Processing Time (hrs)': successful_runs['avg_ic_processing_time'].mean(),
        'Std IC Processing Time (hrs)': successful_runs['avg_ic_processing_time'].std(),
        'Min IC Processing Time (hrs)': successful_runs['avg_ic_processing_time'].min(),
        'Max IC Processing Time (hrs)': successful_runs['avg_ic_processing_time'].max(),
        'Avg OC Processing Time (hrs)': successful_runs['avg_oc_processing_time'].mean(),
        'Std OC Processing Time (hrs)': successful_runs['avg_oc_processing_time'].std(),
        'Min OC Processing Time (hrs)': successful_runs['avg_oc_processing_time'].min(),
        'Max OC Processing Time (hrs)': successful_runs['avg_oc_processing_time'].max(),
        'Avg Train Delay (hrs)': successful_runs['avg_train_delay'].mean(),
        'Avg Train Processing Time (hrs)': successful_runs['avg_train_processing_time'].mean(),
    }

    return pd.DataFrame([summary]).T.rename(columns={0: 'Value'})


def main():
    print("\n" + "=" * 70)
    print("INTERMODAL TERMINAL SIMULATION (MULTIPLE TRACK) - BATCH RUNNER")
    print("=" * 70 + "\n")

    # # Option 1: Run all combinations
    # print("    Running in ALL COMBO")
    # results_df = run_batch_simulations()

    # # Option 2: Run limited number for testing
    print("    Running in TEST MODE (first 2 combinations)")
    results_df = run_batch_simulations(max_runs=2)

    output_file = save_results(results_df)

    # Display summary
    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)
    print(generate_summary_statistics(results_df))
    print("\n" + "=" * 70)
    print(f"Full results: {output_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()