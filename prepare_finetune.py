import pandas as pd
import numpy as np
import json
import os
from data.serialize import serialize_arr, SerializerSettings

# Config
DATA_FILE = 'kpi_15_mins.csv'
FILTER_CELL = 'EnodebA'
KPI_COLUMNS = ['ps_traffic_mb', 'avg_rrc_connected_user', 'prb_dl_used', 'prb_dl_available_total']
HISTORY_LEN = 96
PREDICTION_LEN = 48
OUTPUT_DIR = 'data/finetune'
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train.jsonl')
VAL_FILE = os.path.join(OUTPUT_DIR, 'val.jsonl')

settings = SerializerSettings(base=10, prec=2, signed=True, time_sep=', ', bit_sep='', minus_sign='-')

def parse_timestamp(row):
    date_str = row['date_hour']
    time_str = row['update_time'].split('.')[0]
    return f"{date_str}:{time_str}"

def prepare_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    df = df[df['enodeb'] == FILTER_CELL]
    
    # Aggregation
    df_agg = df.groupby(['date_hour', 'update_time'])[KPI_COLUMNS].sum().reset_index()
    df_agg['timestamp_str'] = df_agg.apply(lambda x: f"{x['date_hour']}:{x['update_time']}", axis=1)
    df_agg = df_agg.sort_values('timestamp_str')
    
    print(f"Total aggregated points: {len(df_agg)}")
    
    data_points = []
    
    # Create sliding windows
    for i in range(len(df_agg) - (HISTORY_LEN + PREDICTION_LEN)):
        window = df_agg.iloc[i : i + HISTORY_LEN + PREDICTION_LEN]
        
        history = window.iloc[:HISTORY_LEN]
        truth = window.iloc[HISTORY_LEN:]
        
        # Create one sample per KPI or one sample for all?
        # "finetune for each kpi". 
        # Strategy: Input instruction contains "Predict [KPI]..."
        
        for kpi in KPI_COLUMNS:
            hist_vals = history[kpi].values
            truth_vals = truth[kpi].values
            
            hist_str = serialize_arr(hist_vals, settings)
            truth_str = serialize_arr(truth_vals, settings)
            
            # Format: User gives history, Model gives truth
            # We can use a standard instruct format
            
            instruction = f"Predict the next {PREDICTION_LEN} steps for {kpi} given the history."
            
            # Message format for chat models
            message = {
                "messages": [
                    {"role": "user", "content": f"{instruction}\nHistory: {hist_str}"},
                    {"role": "assistant", "content": truth_str}
                ]
            }
            data_points.append(message)
            
    print(f"Generated {len(data_points)} training samples.")
    
    # Split Train/Val
    split_idx = int(len(data_points) * 0.9)
    train_data = data_points[:split_idx]
    val_data = data_points[split_idx:]
    
    with open(TRAIN_FILE, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open(VAL_FILE, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(train_data)} to {TRAIN_FILE}")
    print(f"Saved {len(val_data)} to {VAL_FILE}")

if __name__ == "__main__":
    prepare_data()
