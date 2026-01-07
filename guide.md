 i need pretrained model qween 2.5 0.5b instruct can predict kpis with mechanism using lookback timesteps to predict the next timesteps( using 96 timesteps to predict 48 time steps ) , predict 4 kpis ( input : time (date + time) , lookback timestpes) output -> : 48 timesteps for each kpis ( now i want to just use  EnodebA, so filter file first ) @kpi_15_mins.csv 

# model: qwen-2.5-0.5b-instruct
# input: time (date + time) , history 96 timesteps with kpis

# output: 48 timesteps for each kpis


# finetune:  finetune for each kpi 
Input JSON

{
  "enodeb": "EnodebA",
  "freq_minutes": 15,
  "dateTime": "2025-01-01-00:00",
  "history": {
    "kpi": [ ...96 numbers... ],
  }
}


Output JSON

{
  "horizon": 48,
  "forecast": {
    "kpi": [ ...48 numbers... ],
  }
}

# UI: gradio : 
   - label selection : kpi type 
   - input : 
      - dateTime
      - history 96 timesteps with kpi values
   - output : 
      - 48 timesteps for kpi values
      - plot               


# framework: uv 

# Step 1: filter file @kpi_15_mins.csv  -> only use EnodebA
# Step 2: finetune for each kpi
# Step 3: predict with selected kpi 
# Step 4: save result to csv, png 
# Step 5: plot 

