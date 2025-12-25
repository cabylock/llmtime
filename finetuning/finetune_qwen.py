from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import sys
import os

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. Configuration
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model_name = "Qwen/Qwen2.5-0.5B-Instruct" # Or "Qwen/Qwen2.5-0.5B" base model

# 2. Load Model and Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like Llama 3
)

# 3. Add LoRA adapters
# Qwen 2.5 0.5B is small, but LoRA is still efficient. 
# You can set r=0 to do full finetuning if you have enough VRAM (likely fine for 0.5B on 12GB+ GPU).
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# 4. Prepare Dataset
# Option 1: Load from existing JSONL (if available)
# Option 2: Generate from CSV dynamically

from datasets import Dataset
import pandas as pd
import json

print("\n" + "="*60)
print("Loading KPI Dataset")
print("="*60)

# Check if we should use pre-generated JSONL or load from CSV
jsonl_path = "finetuning/train.jsonl"
csv_path = "kpi_15_mins.csv"

if not os.path.exists(csv_path) and os.path.exists("../kpi_15_mins.csv"):
    csv_path = "../kpi_15_mins.csv"

use_csv_loader = True  # Set to False to use pre-generated JSONL

if use_csv_loader and os.path.exists(csv_path):
    print(f"Loading data from CSV: {csv_path}")
    
    # Load and prepare data (same logic as dataset_prep.py)
    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df)}")
    
    # Filter for EnodebA
    df = df[df['enodeb'] == 'EnodebA']
    print(f"  After filtering EnodebA: {len(df)}")
    
    df['date_key'] = df['date_hour'].astype(str) + " " + df['update_time'].astype(str)
    grouped = df.groupby('cell_name')
    
    window_in = 48
    window_out = 24
    kpis = ['ps_traffic_mb', 'avg_rrc_connected_user', 'prb_dl_used', 'prb_dl_available_total']
    
    data_samples = []
    print(f"  Generating sliding windows ({window_in} → {window_out})...")
    
    for name, group in grouped:
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
                
                input_str = ", ".join(map(str, input_seq))
                target_str = ", ".join(map(str, target_seq))
                instruction = f"Predict the next {window_out} values of {kpi} given the previous {window_in} values."
                
                data_samples.append({
                    "instruction": instruction,
                    "input": input_str,
                    "output": target_str
                })
    
    print(f"  Generated {len(data_samples)} training samples")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict({
        "instruction": [s["instruction"] for s in data_samples],
        "input": [s["input"] for s in data_samples],
        "output": [s["output"] for s in data_samples]
    })
    
    # Optionally save for future use
    if not os.path.exists("finetuning"):
        os.makedirs("finetuning")
    print(f"  Saving to {jsonl_path} for future use...")
    with open(jsonl_path, 'w') as f:
        for sample in data_samples:
            f.write(json.dumps(sample) + "\n")
    
else:
    print(f"Loading data from JSONL: {jsonl_path}")
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=jsonl_path, split="train")
    print(f"  Loaded {len(dataset)} samples")

print("="*60 + "\n")

# Define prompt template (matches generation script)
# We can use a simpler prompt since we pre-formatted it in dataset_prep.py 
# but sticking to Alpaca style is robust.
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 5. Training
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Set to None for full training
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer.train()

# 6. Inference
print("Training complete. Testing inference...")
FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
print(tokenizer.batch_decode(outputs))

# 7. Saving
# model.save_pretrained("lora_model") # Local saving
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
