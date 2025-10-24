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


def collect_results(state, params: Dict, run_id: int) -> Dict:
    """
    Collect and aggregate simulation results

    Args:
        state: Global state object from simulation
        params: Simulation parameters
        run_id: Run number

    Returns:
        Dict: Aggregated results including:
              - avg_ic_processing_time: Average IC processing time (hours)
              - avg_oc_processing_time: Average OC processing time (hours)
              - total_containers: Total number of containers processed
              - train_delays: Average train delay time
    """
    try:
        # Extract container events
        container_events = state.container_events

        if not container_events:
            print(f"    No container events found for run {run_id}")
            return {
                'run_id': run_id,
                'status': 'no_data',
                **params,
                'avg_ic_processing_time': None,
                'avg_oc_processing_time': None,
                'total_containers': 0,
                'avg_train_delay': 0
            }

        # Calculate IC processing times
        ic_times = []
        oc_times = []

        for container_id, events in container_events.items():
            # IC processing time: from train arrival to truck exit
            if 'IC-' in container_id:
                if 'train_arrival_expected' in events and 'truck_exit' in events:
                    ic_time = events['truck_exit'] - events['train_arrival_expected']
                    ic_times.append(ic_time)

            # OC processing time: from truck arrival to crane load
            elif 'OC-' in container_id:
                if 'truck_arrival' in events and 'crane_load' in events:
                    oc_time = events['crane_load'] - events['truck_arrival']
                    oc_times.append(oc_time)

        # Calculate averages
        avg_ic_time = np.mean(ic_times) if ic_times else None
        avg_oc_time = np.mean(oc_times) if oc_times else None

        # Calculate train delay statistics
        train_delays = []
        if hasattr(state, 'train_delay_time'):
            train_delays = list(state.train_delay_time.values())

        avg_delay = np.mean(train_delays) if train_delays else 0

        # Calculate train processing times
        train_times = []
        if hasattr(state, 'time_per_train'):
            train_times = list(state.time_per_train.values())

        avg_train_time = np.mean(train_times) if train_times else None

        return {
            'run_id': run_id,
            'status': 'success',
            **params,
            'avg_ic_processing_time': round(avg_ic_time, 4) if avg_ic_time else None,
            'avg_oc_processing_time': round(avg_oc_time, 4) if avg_oc_time else None,
            'total_ic_containers': len(ic_times),
            'total_oc_containers': len(oc_times),
            'total_containers': len(ic_times) + len(oc_times),
            'avg_train_delay': round(avg_delay, 4),
            'avg_train_processing_time': round(avg_train_time, 4) if avg_train_time else None,
            'num_trains_processed': len(train_times)
        }

    except Exception as e:
        print(f"Error collecting results for run {run_id}: {e}")
        import traceback
        traceback.print_exc()

        return {
            'run_id': run_id,
            'status': 'collection_failed',
            'error': str(e),
            **params,
            'avg_ic_processing_time': None,
            'avg_oc_processing_time': None,
            'total_containers': 0
        }


def generate_param_combinations(param_dict: Dict) -> List[Dict]:
    keys = list(param_dict.keys())
    values = list(param_dict.values())

    combinations = []
    for combo in itertools.product(*values):
        param_set = dict(zip(keys, combo))
        combinations.append(param_set)

    return combinations


def run_batch_simulations(param_dict: Dict = None,
                          max_runs: int = None,
                          resume_from: int = None) -> pd.DataFrame:
    """
    Run batch simulations with all parameter combinations

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
    """
    Save results to Excel file with multiple sheets

    Args:
        results_df: Results dataframe
        suffix: Filename suffix (e.g., '_intermediate')
    """
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
    """
    Generate summary statistics from results

    Args:
        results_df: Results dataframe

    Returns:
        pd.DataFrame: Summary statistics
    """
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
    """
    Main entry point for batch simulation
    """
    print("\n" + "=" * 70)
    print("INTERMODAL TERMINAL SIMULATION (MULTIPLE TRACK) - BATCH RUNNER")
    print("=" * 70 + "\n")

    # Option 1: Run all combinations (WARNING: This could take hours!)
    # results_df = run_batch_simulations()

    # # Option 2: Run limited number for testing
    print("    Running in TEST MODE (first 10 combinations)")
    print("    To run all combinations, modify main() function\n")
    results_df = run_batch_simulations(max_runs=10)

    # Save results
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