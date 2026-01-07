"""
Fine-tune Qwen3 model for KPI time series forecasting using LoRA.
"""
import os
import json
import argparse
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)


# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B"  # or "Qwen/Qwen3-1.7B"
OUTPUT_DIR = "models/qwen_kpi_finetuned"
DATA_DIR = "data/finetune"

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training configuration
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
MAX_LENGTH = 2048
WARMUP_RATIO = 0.1


def load_and_prepare_data(tokenizer, data_dir: str):
    """Load and tokenize the training data."""
    
    train_path = Path(data_dir) / "train.jsonl"
    val_path = Path(data_dir) / "val.jsonl"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}. Run prepare_finetune.py first.")
    
    # Load datasets
    dataset = load_dataset(
        'json',
        data_files={
            'train': str(train_path),
            'validation': str(val_path)
        }
    )
    
    def format_example(example):
        """Format example for training."""
        messages = example['messages']
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {'text': text}
    
    dataset = dataset.map(format_example)
    
    def tokenize(example):
        """Tokenize the formatted text."""
        result = tokenizer(
            example['text'],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        )
        result['labels'] = result['input_ids'].copy()
        return result
    
    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=dataset['train'].column_names
    )
    
    return tokenized_dataset


def create_model_and_tokenizer(model_name: str):
    """Load and prepare model with LoRA."""
    
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def train(
    model_name: str = MODEL_NAME,
    output_dir: str = OUTPUT_DIR,
    data_dir: str = DATA_DIR,
    max_steps: int = -1,
    dry_run: bool = False
):
    """Run the fine-tuning process."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = create_model_and_tokenizer(model_name)
    
    # Load and prepare data
    print(f"Loading data from {data_dir}")
    dataset = load_and_prepare_data(tokenizer, data_dir)
    
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Val samples: {len(dataset['validation'])}")
    
    if dry_run:
        print("\n✓ Dry run complete! Model and data loaded successfully.")
        return
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        bf16=True,
        max_steps=max_steps if max_steps > 0 else -1,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save the final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n✓ Fine-tuning complete!")
    print(f"\nNext step: Run predictions with:")
    print(f"  python predict_kpi.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune Qwen3 for KPI forecasting')
    parser.add_argument('--model', default=MODEL_NAME, help='Base model name')
    parser.add_argument('--output', default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--data', default=DATA_DIR, help='Training data directory')
    parser.add_argument('--max_steps', type=int, default=-1, help='Max training steps (-1 for full)')
    parser.add_argument('--dry_run', action='store_true', help='Just load model and data, skip training')
    
    args = parser.parse_args()
    train(args.model, args.output, args.data, args.max_steps, args.dry_run)
