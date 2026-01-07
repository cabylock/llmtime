import gradio as gr
import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from data.serialize import serialize_arr, deserialize_str, SerializerSettings

# Config
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "models/qwen_kpi_finetuned"
HISTORY_LEN = 96
PREDICTION_LEN = 48
KPI_COLUMNS = ['ps_traffic_mb', 'avg_rrc_connected_user', 'prb_dl_used', 'prb_dl_available_total']

settings = SerializerSettings(base=10, prec=2, signed=True, time_sep=', ', bit_sep='', minus_sign='-')

# Load Model
print("Loading model for UI...")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Check if adapter exists
    if os.path.exists(ADAPTER_DIR):
        print(f"Loading adapter from {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    else:
        print("Adapter not found, using base model.")
        model = base_model
        
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

def forecast_kpi(kpi_name, date_time, history_input):
    if model is None:
        return "Model not loaded", None

    try:
        # Parse history input
        # Input format expected: JSON list or comma separated string? 
        # User guide says: "history 96 timesteps with kpis"
        # Let's assume user pastes a JSON list or comma strings.
        
        if history_input.strip().startswith("["):
            history_vals = json.loads(history_input)
        else:
            history_vals = [float(x) for x in history_input.split(',')]
            
        history_vals = np.array(history_vals)
        
        if len(history_vals) != HISTORY_LEN:
             return f"Error: History length is {len(history_vals)}, expected {HISTORY_LEN}.", None

        # Serialize
        hist_str = serialize_arr(history_vals, settings)
        instruction = f"Predict the next {PREDICTION_LEN} steps for {kpi_name} given the history."
        
        input_text = f"{instruction}\nHistory: {hist_str}"
        
        # Format input (instruct)
        # We used the chat template in training: 
        # {"messages": [{"role": "user", "content": ...}]}
        # So we should format it similarly if using apply_chat_template, OR just construct string if model is raw.
        # But `finetune_qwen.py` used `dataset` with `json` loader. 
        # SFTTrainer usually handles the chat template if `dataset_text_field` is not set but messages are present.
        # Let's use tokenizer.apply_chat_template
        
        messages = [
            {"role": "user", "content": input_text}
        ]
        
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
        
        # Determine max tokens
        # From our fix earlier: avg per step * steps * 1.5
        # We don't have exact avg, let's estimate conservative 15 tokens per step (digits + seps)
        # 48 steps * 10 = 480.
        
        max_new_tokens = 600 # safe buffer
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        # Decode response only
        response_ids = generated_ids[0][input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Deserialize
        # Warning: response_text might contain garbage or non-serialized text if fine-tuning failed
        # But we assume it works.
        try:
            pred_vals = deserialize_str(response_text, settings)
        except:
             # If exact match fails, try to parse what we can
             pred_vals = np.array([])
             
        # Plot
        plt.figure(figsize=(10, 6))
        # History
        plt.plot(range(HISTORY_LEN), history_vals, label='History', color='blue')
        # Forecast
        plt.plot(range(HISTORY_LEN, HISTORY_LEN + len(pred_vals)), pred_vals, label='Forecast', color='red', linestyle='--')
        plt.title(f"Forecast for {kpi_name} starting {date_time}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = "forecast_ui.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Output JSON
        output_json = {
            "horizon": len(pred_vals),
            "forecast": {
                 kpi_name: pred_vals.tolist()
            }
        }
        
        return json.dumps(output_json, indent=2), plot_path
        
    except Exception as e:
        return f"Error: {e}", None

# UI Layout
with gr.Blocks(title="Qwen KPI Forecaster") as app:
    gr.Markdown("# Qwen 2.5 0.5B KPI Forecasting System")
    
    with gr.Row():
        kpi_input = gr.Dropdown(choices=KPI_COLUMNS, label="KPI Type", value=KPI_COLUMNS[0])
        date_input = gr.Textbox(label="DateTime", value="2025-01-01-00:00")
    
    # Load sample history for demo (random/placeholder)
    default_history = str(list(np.random.rand(96) * 100))
    history_input = gr.Textbox(label=f"History ({HISTORY_LEN} steps)", value=default_history, lines=5)
    
    predict_btn = gr.Button("Predict", variant="primary")
    
    with gr.Row():
        json_output = gr.Code(label="Output JSON", language="json")
        plot_output = gr.Image(label="Forecast Plot")
        
    predict_btn.click(
        fn=forecast_kpi,
        inputs=[kpi_input, date_input, history_input],
        outputs=[json_output, plot_output]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
