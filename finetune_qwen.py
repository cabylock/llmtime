import os
import torch
import torch.multiprocessing
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import warnings

# Disable warnings
warnings.filterwarnings("ignore")

# Config
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILE = "data/finetune/train.jsonl"
VAL_FILE = "data/finetune/val.jsonl"
OUTPUT_DIR = "models/qwen_kpi_finetuned"

def main():
    # Load Dataset
    print("Loading data...")
    dataset = load_dataset('json', data_files={'train': TRAIN_FILE, 'validation': VAL_FILE})
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Model - Load in 4-bit (optional, but good for speed/memory even on 0.5B)
    # Actually 0.5B is tiny, we can probably load in fp16 easily.
    # Let's use bfloat16 if available
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Target all linear
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training Config
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3, # Fast demo
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none", # Disable wandb
        packing=False
    )
    
    # Check max_seq_length
    print(f"Max Seq Length: 512")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        args=training_args,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    
if __name__ == "__main__":
    main()
