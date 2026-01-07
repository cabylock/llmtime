import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.llmtime import get_llmtime_predictions_data
from data.serialize import SerializerSettings

# Settings
MODEL_NAME = 'qwen-2.5-0.5b-instruct'
DATA_FILE = 'kpi_15_mins.csv'
FILTER_CELL = 'EnodebA' # As per request "EnodebA"
KPI_COLUMNS = ['ps_traffic_mb', 'avg_rrc_connected_user', 'prb_dl_used', 'prb_dl_available_total']
HISTORY_LEN = 96
PREDICTION_LEN = 48

# Serializer settings (adjust as needed for better tokenization)
settings = SerializerSettings(base=10, prec=2, signed=True, time_sep=', ', bit_sep='', minus_sign='-')

def main():
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    
    # Filter by EnodebA
    print(f"Filtering for enodeb == {FILTER_CELL}...")
    df = df[df['enodeb'] == FILTER_CELL]
    
    if df.empty:
        print(f"No data found for {FILTER_CELL}!")
        return

    # Sort by date (assuming date_hour is the time column, or we rely on the order)
    # The file has 'date_hour', and '00:00.0' in update_time, seems like we should combine or just trust the order if it's sequential.
    # Looking at the head command output: 2025-10-11-00.  It's just date-hour.
    # Let's assume the rows are ordered by time or sort them.
    # However, there are multiple cells 'EnodebA3', 'EnodebA1111B031', 'EnodebA61B281' under EnodebA.
    # The user request said "now i want to just use EnodebA, so filter file first". 
    # The 'enodeb' column has 'EnodebA'. But 'cell_name' distinguishes them.
    # Aggregating or picking one cell?
    # "predict 48 timesteps for each kpis" -> implies for the entity 'EnodebA'.
    # If there are multiple cells under EnodebA, we might need to aggregate (sum/mean) or maybe the user implies EnodebA is the granularity.
    # Let's check if there are duplicates for the same timestamp for EnodebA.
    # The head output shows rows with same timestamp but different cell_name.
    # "EnodebA" seems to be the parent node. The KPIs like ps_traffic_mb usually aggregate.
    
    # Let's aggregate by grouping by timestamp.
    # We need to construct a proper timestamp column.
    # date_hour is '2025-10-11-00', update_time is '00:00.0'.
    
    # Actually, simpler: Group by date_hour (and maybe update_time) and sum the KPIs for EnodebA.
    # Or maybe the user just wants one specific cell? "now i want to just use EnodebA".
    # I will aggregate (sum) for safety, as 'EnodebA' usually refers to the site level.
    print("Aggregating data for EnodebA...")
    # Combine date_hour and update_time to create a timestamp
    # date_hour: 2025-10-11-00
    # update_time: 00:00.0 or 00:15.0
    # We need to parse this carefully.
    
    def parse_timestamp(row):
        date_part = row['date_hour'][:-3] # Remove the hour part '2025-10-11'
        # Actually date_hour is 2025-10-11-00. 
        # let's assume standard format YYYY-MM-DD-HH
        
        # update_time is MM:SS.s? or HH:MM.s?
        # sample: 00:00.0, 15:00.0 ? No, likely minutes.
        # file name kpi_15_mins.csv suggests 15 min intervals.
        # sample output showed 00:00.0.
        # Let's combine them string-wise and parse.
        
        # date_hour: 2025-10-11-00
        # update_time: 00:00.0
        # result: 2025-10-11 00:00:00
        
        date_str = row['date_hour']
        # Extract date and hour
        # It seems date_hour allows hour like -00, -01?
        # Let's clean it.
        
        time_str = row['update_time'].split('.')[0] # remove .0
        
        # If date_hour has the hour, we should use it.
        # date_hour format assumption: YYYY-MM-DD-HH
        
        full_str = f"{date_str}:{time_str}"
        return full_str

    # Simplify aggregation: first just get the relevant columns
    df_agg = df.groupby(['date_hour', 'update_time'])[KPI_COLUMNS].sum().reset_index()
    
    # Construct timestamp
    # We will try to rely on sorting by string first as it might be safer if format is consistent
    df_agg['timestamp_str'] = df_agg.apply(lambda x: f"{x['date_hour']}:{x['update_time']}", axis=1)
    df_agg = df_agg.sort_values('timestamp_str')
    
    # Create a nice datetime object for plotting
    # format: YYYY-MM-DD-HH:MM:SS
    # date_hour: 2025-10-11-00 (Year-Month-Day-Hour)
    # update_time: 00:00.0 (Minute:Second.Deci)
    # So 2025-10-11-00:00:00
    try:
        df_agg['timestamp'] = pd.to_datetime(df_agg['timestamp_str'], format='%Y-%m-%d-%H:%M:%S.%f')
    except:
        # Fallback if format differs slightly
        df_agg['timestamp'] = pd.to_datetime(df_agg['timestamp_str'], errors='coerce')
        
    print(f"Total aggregated data points: {len(df_agg)}")
    
    if len(df_agg) < HISTORY_LEN + PREDICTION_LEN:
         print(f"Not enough data! Need at least {HISTORY_LEN + PREDICTION_LEN} points for valuation.")
         return

    # EVALUATION MODE: Use the last window where we have Ground Truth
    # Shift back by PREDICTION_LEN
    
    truth_slice = df_agg.iloc[-PREDICTION_LEN:]
    history_slice = df_agg.iloc[-(HISTORY_LEN + PREDICTION_LEN):-PREDICTION_LEN]
    
    results = {}
    
    for kpi in KPI_COLUMNS:
        print(f"Predicting for KPI: {kpi}")
        train_series = history_slice[kpi]
        test_series = truth_slice[kpi] 
        
        try:
            pred_data = get_llmtime_predictions_data(
                train=train_series,
                test=test_series,
                model=MODEL_NAME,
                settings=settings,
                num_samples=5,
                temp=0.7,
                parallel=False
            )
            
            median_pred = pred_data['median']
            results[kpi] = median_pred.values
            
            print(f"  > Prediction (first 5): {median_pred.values[:5]}")
            
        except Exception as e:
            print(f"Error predicting {kpi}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    print("\nSAVING PREDICTIONS...")
    # Add timestamps to the results
    # truth_slice has the timestamps corresponding to the predictions
    output_df = pd.DataFrame(results)
    output_df.insert(0, 'timestamp', truth_slice['timestamp_str'].values)
    
    output_df.to_csv("qwen_kpi.csv", index=False)
    print("Saved to qwen_kpi.csv")

    # Plotting
    import matplotlib.dates as mdates
    print("\nGENERATING PLOTS...")
    plt.figure(figsize=(15, 12))
    
    for i, kpi in enumerate(KPI_COLUMNS):
        plt.subplot(len(KPI_COLUMNS), 1, i+1)
        
        # Plot history
        plt.plot(history_slice['timestamp'], history_slice[kpi].values, label='History', color='blue')
        
        # Plot Truth
        plt.plot(truth_slice['timestamp'], truth_slice[kpi].values, label='Ground Truth', color='green')
        
        # Plot Prediction
        # Prediction aligns with Truth timestamps
        plt.plot(truth_slice['timestamp'], results[kpi], label='Prediction (Qwen)', color='red', linestyle='--')
        
        plt.title(f"Forecast for {kpi}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("qwen_kpi.png")
    print("Saved plot to qwen_kpi.png")
    plt.close()

if __name__ == "__main__":
    main()
