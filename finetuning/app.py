import gradio as gr
from unsloth import FastLanguageModel
import torch
import random
import json
import matplotlib.pyplot as plt
import io
from PIL import Image

# 1. Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True
model_name = "outputs/checkpoint-60" 

# 2. Load Model
print(f"Loading model from {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

# 3. Load Sample Data
samples = []
try:
    with open("finetuning/train.jsonl", "r") as f:
        for line in f:
            samples.append(json.loads(line))
except:
    print("Could not load train.jsonl for samples.")

# KPIs
kpi_options = ['ps_traffic_mb', 'avg_rrc_connected_user', 'prb_dl_used', 'prb_dl_available_total']

def get_random_sample(kpi_name):
    if not samples:
        return f"Random sample for {kpi_name}", "" # Should not happen if file exists
    
    # Filter samples that match the requested KPI instruction
    # The instruction format is "Predict the next 24 values of {kpi} given..."
    relevant_samples = [s for s in samples if kpi_name in s["instruction"]]
    
    if not relevant_samples:
        return "No samples found for this KPI.", ""
        
    sample = random.choice(relevant_samples)
    return sample["input"], sample["output"]

def predict(kpi_name, input_text):
    try:
        # Split and clean
        values = [v.strip() for v in input_text.split(',') if v.strip()]
        cleaned_values = []
        for v in values:
            try:
                float(v) # Check if number
                cleaned_values.append(v)
            except ValueError:
                continue # Skip invalid chars
        
        # Enforce window size (last 48)
        if len(cleaned_values) > 48:
            cleaned_values = cleaned_values[-48:]
            
        validated_input = ", ".join(cleaned_values)
        
        if not cleaned_values:
            return "Error: No valid numbers found in input.", None
            
    except Exception as e:
        return f"Error processing input: {str(e)}", None

    # Prepare Prompt
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""
    # Use variable name in instruction
    instruction = f"Predict the next 24 values of {kpi_name} given the previous 48 values."
    
    prompt = alpaca_prompt.format(instruction, validated_input)

    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
    decoded_output = tokenizer.batch_decode(outputs)[0]
    
    # Extract prediction
    response_keyword = "### Response:\n"
    if response_keyword in decoded_output:
        prediction = decoded_output.split(response_keyword)[1].strip().replace("<|im_end|>", "")
    else:
        prediction = decoded_output 

    # Parse for plotting
    img = None
    try:
        past_values = [float(x) for x in cleaned_values] # Use cleaned values
        future_values = [float(x.strip()) for x in prediction.split(',') if x.strip()] # Robust split
        
        # Create Plot
        fig = plt.figure(figsize=(10, 5))
        plt.plot(range(len(past_values)), past_values, label=f'History ({len(past_values)})', marker='o')
        plt.plot(range(len(past_values), len(past_values) + len(future_values)), future_values, label='Forecast (24)', marker='x', linestyle='--')

        plt.title(f"EnodebA Forecast: {kpi_name}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        # Return figure object for gr.Plot
        return prediction, fig
        
    except Exception as e:
        return f"{prediction}\n\n(Error plotting: {str(e)})", None

# 4. Interface definition
# 4. Interface definition
with gr.Blocks(title="Qwen 2.5 Forecasting Demo") as demo:
    gr.Markdown("# 📈 EnodebA Traffic Forecasting with Qwen 2.5 0.5B")
    
    with gr.Row():
        # --- Left Column: Input ---
        with gr.Column(scale=1):
            gr.Markdown("### 1. Input Data")
            kpi_dropdown = gr.Dropdown(choices=kpi_options, value=kpi_options[0], label="Select KPI")
            input_box = gr.Textbox(lines=3, label="Input Sequence (48 values)", placeholder="e.g. 10.5, 12.3, ...")
            ground_truth_box = gr.Textbox(lines=3, label="Ground Truth Output (Optional for Visualization)", placeholder="e.g. 15.2, 14.1 ...")
            
            with gr.Row():
                fill_btn = gr.Button("🎲 Random Sample")
                vis_btn = gr.Button("👁️ Visualize Input + Truth")
            
            # Input Plot (Visualizer)
            input_plot = gr.Plot(label="Input Visualization")

        # --- Right Column: Output ---
        with gr.Column(scale=1):
            gr.Markdown("### 2. Forecast")
            run_btn = gr.Button("🚀 Run Forecast", variant="primary")
            
            output_box = gr.Textbox(lines=3, label="Predicted Sequence (24 values)")
            
            # Forecast Plot
            forecast_plot = gr.Plot(label="Forecast Visualization")

    # Helper Functions
    def update_input_label(kpi):
        return gr.Textbox(label=f"Input Sequence (48 values) for {kpi}")

    def visualize_sequence(input_text, output_text):
        try:
            values = [float(val.strip()) for val in input_text.split(',') if val.strip()]
            if not values:
                return None
            
            fig = plt.figure(figsize=(10, 4))
            plt.plot(range(len(values)), values, marker='o', label='Input')
            
            # Plot Ground Truth if present
            if output_text:
                try:
                    out_values = [float(val.strip()) for val in output_text.split(',') if val.strip()]
                    if out_values:
                        start_x = len(values)
                        plt.plot(range(start_x, start_x + len(out_values)), out_values, marker='x', linestyle='--', label='Ground Truth')
                except:
                    pass
            
            plt.title(f"Sequence Visualization")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            return fig
        except Exception:
            return None

    # Event wiring
    kpi_dropdown.change(update_input_label, inputs=kpi_dropdown, outputs=input_box)
    
    # Fill button updates BOTH Input and Ground Truth boxes
    fill_btn.click(get_random_sample, inputs=kpi_dropdown, outputs=[input_box, ground_truth_box])
    
    # Visualize button reads BOTH boxes
    vis_btn.click(visualize_sequence, inputs=[input_box, ground_truth_box], outputs=input_plot)
    
    # Run button ONLY uses Input (Ground truth is not for the model)
    run_btn.click(predict, inputs=[kpi_dropdown, input_box], outputs=[output_box, forecast_plot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
