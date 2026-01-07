"""
Run KPI predictions using fine-tuned Qwen3 model.
Evaluates on validation data and generates comparison plots.
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from models.qwen import load_qwen_model, qwen_completion
from data.kpi_serialize import deserialize_kpi_window, serialize_kpi_window, KPISerializerSettings


def load_validation_data(data_dir: str = 'data/finetune'):
    """Load raw validation data for evaluation."""
    val_raw_path = Path(data_dir) / 'val_raw.json'
    
    if not val_raw_path.exists():
        raise FileNotFoundError(f"Validation data not found at {val_raw_path}. Run prepare_finetune.py first.")
    
    with open(val_raw_path, 'r') as f:
        return json.load(f)


def load_scalers(data_dir: str = 'data/finetune'):
    """Load normalization scalers."""
    scalers_path = Path(data_dir) / 'scalers.json'
    
    with open(scalers_path, 'r') as f:
        return json.load(f)


def denormalize(values: List[float], kpi_name: str, scalers: Dict) -> List[float]:
    """Denormalize values back to original scale."""
    min_val = scalers[kpi_name]['min']
    max_val = scalers[kpi_name]['max']
    return [v * (max_val - min_val) + min_val for v in values]


def normalize(values: List[float], kpi_name: str, scalers: Dict) -> List[float]:
    """Normalize values."""
    min_val = scalers[kpi_name]['min']
    max_val = scalers[kpi_name]['max']
    return [(v - min_val) / (max_val - min_val) for v in values]


def predict_kpis(
    model,
    tokenizer,
    input_kpis: Dict[str, List[float]],
    scalers: Dict,
    settings: KPISerializerSettings = None
) -> Dict[str, List[float]]:
    """
    Generate predictions for a single window.
    
    Args:
        model: Qwen model
        tokenizer: Qwen tokenizer
        input_kpis: Input KPI values (original scale)
        scalers: Normalization scalers
        settings: Serialization settings
        
    Returns:
        Predicted KPI values (original scale)
    """
    settings = settings or KPISerializerSettings()
    
    # Normalize input
    normalized_input = {
        kpi: normalize(values, kpi, scalers)
        for kpi, values in input_kpis.items()
    }
    
    # Serialize input
    input_text = serialize_kpi_window(normalized_input, settings)
    
    prompt = f"Predict the next 48 hours of KPI values based on the following 96 hours of data:\n{input_text}\n\nPrediction:"
    
    # Generate prediction
    completions = qwen_completion(model, tokenizer, prompt, max_new_tokens=800, temperature=0.3)
    completion = completions[0]
    
    # Deserialize prediction
    try:
        predicted_normalized = deserialize_kpi_window(completion, settings)
        
        # Denormalize
        predicted = {
            kpi: denormalize(values, kpi, scalers)
            for kpi, values in predicted_normalized.items()
        }
        
        return predicted
    except Exception as e:
        print(f"Error parsing prediction: {e}")
        print(f"Raw completion: {completion[:200]}...")
        return None


def calculate_metrics(actual: Dict, predicted: Dict) -> Dict[str, Dict[str, float]]:
    """Calculate MAE, RMSE, MAPE for each KPI."""
    metrics = {}
    
    for kpi in actual.keys():
        if kpi not in predicted or len(predicted[kpi]) == 0:
            continue
            
        actual_arr = np.array(actual[kpi][:len(predicted[kpi])])
        pred_arr = np.array(predicted[kpi])
        
        # Pad or truncate to same length
        min_len = min(len(actual_arr), len(pred_arr))
        actual_arr = actual_arr[:min_len]
        pred_arr = pred_arr[:min_len]
        
        mae = np.mean(np.abs(actual_arr - pred_arr))
        rmse = np.sqrt(np.mean((actual_arr - pred_arr) ** 2))
        
        # MAPE (avoid division by zero)
        non_zero_mask = actual_arr != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((actual_arr[non_zero_mask] - pred_arr[non_zero_mask]) / actual_arr[non_zero_mask])) * 100
        else:
            mape = float('nan')
        
        metrics[kpi] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }
    
    return metrics


def plot_comparison(
    actual: Dict,
    predicted: Dict,
    input_kpis: Dict,
    cell_name: str,
    output_path: str,
    kpi_names: List[str] = None
):
    """
    Plot actual vs predicted values for all KPIs on the same figure.
    """
    kpi_names = kpi_names or list(actual.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    kpi_display_names = {
        'ps_traffic_mb': 'Traffic (MB)',
        'avg_rrc_connected_user': 'Avg RRC Users',
        'prb_dl_used': 'PRB DL Used',
        'prb_dl_available_total': 'PRB DL Available'
    }
    
    for idx, kpi in enumerate(kpi_names):
        if kpi not in actual or kpi not in predicted:
            continue
            
        ax = axes[idx]
        
        # Input data (96 timesteps)
        input_x = list(range(len(input_kpis[kpi])))
        input_y = input_kpis[kpi]
        
        # Actual output (48 timesteps)
        actual_x = list(range(len(input_kpis[kpi]), len(input_kpis[kpi]) + len(actual[kpi])))
        actual_y = actual[kpi]
        
        # Predicted output
        pred_len = min(len(predicted[kpi]), len(actual[kpi]))
        pred_x = list(range(len(input_kpis[kpi]), len(input_kpis[kpi]) + pred_len))
        pred_y = predicted[kpi][:pred_len]
        
        # Plot
        ax.plot(input_x, input_y, 'b-', label='Input (96h)', alpha=0.7)
        ax.plot(actual_x, actual_y, 'g-', label='Actual', linewidth=2)
        ax.plot(pred_x, pred_y, 'r--', label='Predicted', linewidth=2)
        
        # Mark the prediction start
        ax.axvline(x=len(input_kpis[kpi]), color='gray', linestyle=':', alpha=0.5)
        
        ax.set_title(kpi_display_names.get(kpi, kpi))
        ax.set_xlabel('Hour')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'KPI Prediction vs Actual - {cell_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {output_path}")


def run_evaluation(
    data_dir: str = 'data/finetune',
    output_dir: str = 'outputs/predictions',
    use_finetuned: bool = True,
    num_samples: int = 5
):
    """Run evaluation on validation data."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    model, tokenizer = load_qwen_model(use_finetuned=use_finetuned)
    
    print("Loading validation data...")
    val_data = load_validation_data(data_dir)
    scalers = load_scalers(data_dir)
    
    settings = KPISerializerSettings()
    
    # Evaluate on a subset
    num_samples = min(num_samples, len(val_data))
    print(f"\nEvaluating on {num_samples} samples...")
    
    all_metrics = []
    
    for i in range(num_samples):
        sample = val_data[i]
        cell_name = sample['cell_name']
        
        print(f"\n[{i+1}/{num_samples}] Processing {cell_name}...")
        
        # Get predictions
        predicted = predict_kpis(
            model, tokenizer,
            sample['input_kpis'],
            scalers,
            settings
        )
        
        if predicted is None:
            print(f"  Skipping due to prediction error")
            continue
        
        # Calculate metrics
        metrics = calculate_metrics(sample['output_kpis'], predicted)
        all_metrics.append(metrics)
        
        # Print metrics
        for kpi, m in metrics.items():
            print(f"  {kpi}: MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, MAPE={m['mape']:.1f}%")
        
        # Plot comparison
        plot_path = output_path / f"prediction_{i}_{cell_name}.png"
        plot_comparison(
            sample['output_kpis'],
            predicted,
            sample['input_kpis'],
            cell_name,
            str(plot_path)
        )
    
    # Aggregate metrics
    if all_metrics:
        print("\n" + "="*50)
        print("AGGREGATE METRICS")
        print("="*50)
        
        kpi_names = list(all_metrics[0].keys())
        for kpi in kpi_names:
            mae_values = [m[kpi]['mae'] for m in all_metrics if kpi in m]
            rmse_values = [m[kpi]['rmse'] for m in all_metrics if kpi in m]
            mape_values = [m[kpi]['mape'] for m in all_metrics if kpi in m and not np.isnan(m[kpi]['mape'])]
            
            print(f"\n{kpi}:")
            print(f"  MAE:  {np.mean(mae_values):.2f} ± {np.std(mae_values):.2f}")
            print(f"  RMSE: {np.mean(rmse_values):.2f} ± {np.std(rmse_values):.2f}")
            if mape_values:
                print(f"  MAPE: {np.mean(mape_values):.1f}% ± {np.std(mape_values):.1f}%")
    
    print(f"\n✓ Evaluation complete! Plots saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run KPI predictions')
    parser.add_argument('--data', default='data/finetune', help='Data directory')
    parser.add_argument('--output', default='outputs/predictions', help='Output directory')
    parser.add_argument('--base', action='store_true', help='Use base model instead of fine-tuned')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to evaluate')
    
    args = parser.parse_args()
    run_evaluation(args.data, args.output, use_finetuned=not args.base, num_samples=args.num_samples)
