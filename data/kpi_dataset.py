"""
KPI Dataset loader and preprocessor for time series forecasting.
Filters enodebA data and creates sliding windows for 96->48 prediction.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class KPIConfig:
    """Configuration for KPI dataset processing."""
    input_steps: int = 96  # 4 days of hourly data
    output_steps: int = 48  # 2 days of hourly data
    kpi_columns: Tuple[str, ...] = (
        'ps_traffic_mb',
        'avg_rrc_connected_user', 
        'prb_dl_used',
        'prb_dl_available_total'
    )
    enodeb_filter: str = 'enodebA'  # Filter only enodebA
    date_column: str = 'date_hour'
    cell_column: str = 'cell_name'


def load_kpi_data(
    csv_path: str,
    config: Optional[KPIConfig] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Load and filter KPI data from CSV.
    
    Args:
        csv_path: Path to the CSV file
        config: KPI configuration
        limit: Optional row limit for testing
        
    Returns:
        Filtered DataFrame with enodebA data only
    """
    config = config or KPIConfig()
    
    # Load CSV
    df = pd.read_csv(csv_path, nrows=limit)
    
    # Filter enodebA only
    df = df[df['enodeb'] == config.enodeb_filter].copy()
    
    # Parse date_hour into datetime
    df['datetime'] = pd.to_datetime(df[config.date_column], format='%Y-%m-%d-%H')
    
    # Sort by cell and datetime
    df = df.sort_values([config.cell_column, 'datetime'])
    
    # Keep only needed columns
    columns_to_keep = [config.cell_column, 'datetime'] + list(config.kpi_columns)
    df = df[columns_to_keep]
    
    # Fill missing values with forward fill then backward fill
    for col in config.kpi_columns:
        df[col] = df.groupby(config.cell_column)[col].ffill().bfill()
    
    return df


def create_sliding_windows(
    df: pd.DataFrame,
    config: Optional[KPIConfig] = None
) -> List[Dict]:
    """
    Create sliding windows for each cell.
    
    Args:
        df: DataFrame with KPI data
        config: KPI configuration
        
    Returns:
        List of dictionaries with input/output windows
    """
    config = config or KPIConfig()
    windows = []
    
    total_steps = config.input_steps + config.output_steps
    
    for cell_name, cell_df in df.groupby(config.cell_column):
        cell_df = cell_df.reset_index(drop=True)
        n_rows = len(cell_df)
        
        # Create overlapping windows with stride of 24 (1 day)
        stride = 24
        
        for start_idx in range(0, n_rows - total_steps + 1, stride):
            end_input = start_idx + config.input_steps
            end_output = end_input + config.output_steps
            
            input_df = cell_df.iloc[start_idx:end_input]
            output_df = cell_df.iloc[end_input:end_output]
            
            window = {
                'cell_name': cell_name,
                'input_datetimes': input_df['datetime'].tolist(),
                'output_datetimes': output_df['datetime'].tolist(),
                'input_kpis': {
                    col: input_df[col].values.tolist()
                    for col in config.kpi_columns
                },
                'output_kpis': {
                    col: output_df[col].values.tolist()
                    for col in config.kpi_columns
                }
            }
            windows.append(window)
    
    return windows


def normalize_kpis(
    windows: List[Dict],
    config: Optional[KPIConfig] = None
) -> Tuple[List[Dict], Dict[str, Dict[str, float]]]:
    """
    Normalize KPI values using min-max scaling per KPI.
    
    Returns:
        Normalized windows and scaling parameters
    """
    config = config or KPIConfig()
    
    # Calculate global min/max for each KPI
    all_values = {col: [] for col in config.kpi_columns}
    
    for window in windows:
        for col in config.kpi_columns:
            all_values[col].extend(window['input_kpis'][col])
            all_values[col].extend(window['output_kpis'][col])
    
    scalers = {}
    for col in config.kpi_columns:
        min_val = min(all_values[col])
        max_val = max(all_values[col])
        # Avoid division by zero
        if max_val == min_val:
            max_val = min_val + 1
        scalers[col] = {'min': min_val, 'max': max_val}
    
    # Normalize windows
    normalized_windows = []
    for window in windows:
        norm_window = {
            'cell_name': window['cell_name'],
            'input_datetimes': window['input_datetimes'],
            'output_datetimes': window['output_datetimes'],
            'input_kpis': {},
            'output_kpis': {}
        }
        
        for col in config.kpi_columns:
            min_val = scalers[col]['min']
            max_val = scalers[col]['max']
            
            norm_window['input_kpis'][col] = [
                (v - min_val) / (max_val - min_val)
                for v in window['input_kpis'][col]
            ]
            norm_window['output_kpis'][col] = [
                (v - min_val) / (max_val - min_val)
                for v in window['output_kpis'][col]
            ]
        
        normalized_windows.append(norm_window)
    
    return normalized_windows, scalers


def train_val_split(
    windows: List[Dict],
    val_ratio: float = 0.2
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split windows into train and validation sets by time (not random).
    Uses the last val_ratio portion as validation.
    """
    # Sort by first input datetime
    windows_sorted = sorted(
        windows,
        key=lambda w: w['input_datetimes'][0]
    )
    
    split_idx = int(len(windows_sorted) * (1 - val_ratio))
    
    return windows_sorted[:split_idx], windows_sorted[split_idx:]


if __name__ == '__main__':
    # Test loading
    config = KPIConfig()
    df = load_kpi_data('data_3_months.csv', config, limit=10000)
    print(f"Loaded {len(df)} rows for {df['cell_name'].nunique()} cells")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Columns: {list(df.columns)}")
