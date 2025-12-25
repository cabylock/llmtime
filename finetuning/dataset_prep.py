import pandas as pd
import json
import os

def prepare_dataset(csv_path="kpi_15_mins.csv", output_path="finetuning/train.jsonl", window_in=48, window_out=24):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter for EnodebA
    print("Filtering for EnodebA...")
    df = df[df['enodeb'] == 'EnodebA']
    
    df['date_key'] = df['date_hour'].astype(str) + " " + df['update_time'].astype(str)
    
    # Group by cell_name to create time series
    grouped = df.groupby('cell_name')
    
    data_samples = []
    
    # List of KPIs to forecast
    kpis = ['ps_traffic_mb', 'avg_rrc_connected_user', 'prb_dl_used', 'prb_dl_available_total']
    
    print("Generating windows for all KPIs...")
    for name, group in grouped:
        # Sort by the implicit time key just in case
        group = group.sort_values(by=['date_hour', 'update_time'])
        
        for kpi in kpis:
            if kpi not in group.columns:
                continue
                
            values = group[kpi].tolist()
            
            if len(values) < window_in + window_out:
                continue
                
            for i in range(len(values) - window_in - window_out + 1):
                input_seq = values[i : i + window_in]
                target_seq = values[i + window_in : i + window_in + window_out]
                
                # Format as string
                input_str = ", ".join(map(str, input_seq))
                target_str = ", ".join(map(str, target_seq))
                
                # Instruction with Variable Name
                instruction = f"Predict the next {window_out} values of {kpi} given the previous {window_in} values."
                
                data_samples.append({
                    "instruction": instruction,
                    "input": input_str,
                    "output": target_str
                })
            
    print(f"Generated {len(data_samples)} samples.")
    
    # Save to JSONL
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        for sample in data_samples:
            f.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    if not os.path.exists("finetuning"):
        os.makedirs("finetuning")
    # Adjust path if script is run from root or finetuning dir
    csv_path = "kpi_15_mins.csv"
    if not os.path.exists(csv_path) and os.path.exists("../kpi_15_mins.csv"):
        csv_path = "../kpi_15_mins.csv"
        
    prepare_dataset(csv_path=csv_path)
