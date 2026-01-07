"""
Prepare training data for Qwen fine-tuning.
Generates JSONL files from CSV data.
"""
import json
import argparse
from pathlib import Path
from data.kpi_dataset import (
    load_kpi_data,
    create_sliding_windows,
    normalize_kpis,
    train_val_split,
    KPIConfig
)
from data.kpi_serialize import window_to_prompt, KPISerializerSettings


def prepare_finetune_data(
    csv_path: str = 'data_3_months.csv',
    output_dir: str = 'data/finetune',
    val_ratio: float = 0.2
):
    """
    Prepare training and validation JSONL files for fine-tuning.
    
    Args:
        csv_path: Path to input CSV
        output_dir: Directory to save JSONL files
        val_ratio: Validation set ratio
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {csv_path}...")
    config = KPIConfig()
    df = load_kpi_data(csv_path, config)
    
    print(f"Loaded {len(df)} rows for {df['cell_name'].nunique()} cells")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    print("\nCreating sliding windows...")
    windows = create_sliding_windows(df, config)
    print(f"Created {len(windows)} windows")
    
    print("\nNormalizing KPIs...")
    normalized_windows, scalers = normalize_kpis(windows, config)
    
    # Save scalers for inference
    scalers_path = output_path / 'scalers.json'
    with open(scalers_path, 'w') as f:
        json.dump(scalers, f, indent=2)
    print(f"Saved scalers to {scalers_path}")
    
    print("\nSplitting into train/val...")
    train_windows, val_windows = train_val_split(normalized_windows, val_ratio)
    print(f"Train: {len(train_windows)}, Val: {len(val_windows)}")
    
    # Convert to prompts and save
    settings = KPISerializerSettings()
    
    train_path = output_path / 'train.jsonl'
    with open(train_path, 'w') as f:
        for window in train_windows:
            prompt_data = window_to_prompt(window, settings)
            # Format for Qwen fine-tuning (messages format)
            messages = [
                {"role": "user", "content": prompt_data['prompt']},
                {"role": "assistant", "content": prompt_data['completion']}
            ]
            f.write(json.dumps({"messages": messages}) + '\n')
    print(f"Saved {len(train_windows)} training examples to {train_path}")
    
    val_path = output_path / 'val.jsonl'
    with open(val_path, 'w') as f:
        for window in val_windows:
            prompt_data = window_to_prompt(window, settings)
            messages = [
                {"role": "user", "content": prompt_data['prompt']},
                {"role": "assistant", "content": prompt_data['completion']}
            ]
            f.write(json.dumps({"messages": messages}) + '\n')
    print(f"Saved {len(val_windows)} validation examples to {val_path}")
    
    # Also save raw windows for evaluation (with original values)
    raw_val_path = output_path / 'val_raw.json'
    with open(raw_val_path, 'w') as f:
        # Convert datetime objects to strings
        serializable_windows = []
        for i, (norm_w, orig_w) in enumerate(zip(val_windows, windows[-len(val_windows):])):
            serializable_windows.append({
                'cell_name': orig_w['cell_name'],
                'input_datetimes': [dt.isoformat() for dt in orig_w['input_datetimes']],
                'output_datetimes': [dt.isoformat() for dt in orig_w['output_datetimes']],
                'input_kpis': orig_w['input_kpis'],
                'output_kpis': orig_w['output_kpis'],  # Original values for evaluation
            })
        json.dump(serializable_windows, f, indent=2)
    print(f"Saved raw validation data to {raw_val_path}")
    
    print("\n✓ Data preparation complete!")
    print(f"\nNext step: Run fine-tuning with:")
    print(f"  python finetuning/finetune_qwen.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare KPI data for fine-tuning')
    parser.add_argument('--csv', default='data_3_months.csv', help='Input CSV path')
    parser.add_argument('--output', default='data/finetune', help='Output directory')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation ratio')
    
    args = parser.parse_args()
    prepare_finetune_data(args.csv, args.output, args.val_ratio)
