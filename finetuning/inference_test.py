from unsloth import FastLanguageModel
import torch
import json

# 1. Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True
model_name = "outputs/checkpoint-60" # Path to your trained model checkpoint

# 2. Load Model (Inference Mode)
print(f"Loading model from {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

# 3. Load a sample from train.jsonl
# We'll take the last line as a "test" sample (assuming time order)
print("Loading sample data...")
samples = []
with open("finetuning/train.jsonl", "r") as f:
    for line in f:
        samples.append(json.loads(line))

# Pick a sample (e.g., the last one)
test_sample = samples[-1]
input_text = test_sample["input"]
ground_truth = test_sample["output"]

# 4. Prepare Prompt
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

prompt = alpaca_prompt.format(
    "Predict the next 24 values given the previous 48 values.", # Instruction must match training
    input_text
)

# 5. Generate
print("Generating prediction...")
inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
decoded_output = tokenizer.batch_decode(outputs)[0]

# Extract response part
response_start = decoded_output.find("### Response:\n") + len("### Response:\n")
prediction = decoded_output[response_start:].strip().replace("<|im_end|>", "")

print("\n" + "="*50)
print("INPUT (Last 48 steps):")
print(input_text)
print("-" * 50)
print("GROUND TRUTH (Next 24 steps):")
print(ground_truth)
print("-" * 50)
print("PREDICTION:")
print(prediction)
print("="*50)
