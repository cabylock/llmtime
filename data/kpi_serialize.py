"""
Serialization for multi-variate KPI time series data.
Converts KPI arrays to text format for LLM input/output.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass  
class KPISerializerSettings:
    """Settings for KPI serialization."""
    precision: int = 2  # Decimal places
    kpi_sep: str = ', '  # Separator between KPIs
    time_sep: str = '; '  # Separator between timesteps
    kpi_names: tuple = (
        'traffic',
        'users', 
        'prb_used',
        'prb_avail'
    )  # Short names for KPIs


def serialize_kpi_window(
    kpis: Dict[str, List[float]],
    settings: Optional[KPISerializerSettings] = None
) -> str:
    """
    Serialize KPI values to text format.
    
    Format: "traffic=1.23, users=4.56, prb_used=7.89, prb_avail=0.12; ..."
    
    Args:
        kpis: Dict mapping KPI names to value lists
        settings: Serialization settings
        
    Returns:
        Serialized string representation
    """
    settings = settings or KPISerializerSettings()
    
    # Get the number of timesteps
    n_steps = len(next(iter(kpis.values())))
    
    # Original KPI column names to short names mapping
    name_map = {
        'ps_traffic_mb': 'traffic',
        'avg_rrc_connected_user': 'users',
        'prb_dl_used': 'prb_used',
        'prb_dl_available_total': 'prb_avail'
    }
    
    timesteps = []
    for i in range(n_steps):
        parts = []
        for orig_name, short_name in name_map.items():
            if orig_name in kpis:
                val = kpis[orig_name][i]
                parts.append(f"{short_name}={val:.{settings.precision}f}")
        timesteps.append(settings.kpi_sep.join(parts))
    
    return settings.time_sep.join(timesteps)


def deserialize_kpi_window(
    text: str,
    settings: Optional[KPISerializerSettings] = None
) -> Dict[str, List[float]]:
    """
    Deserialize text back to KPI values.
    
    Args:
        text: Serialized KPI string
        settings: Serialization settings
        
    Returns:
        Dict mapping KPI names to value lists
    """
    settings = settings or KPISerializerSettings()
    
    # Reverse mapping
    name_map = {
        'traffic': 'ps_traffic_mb',
        'users': 'avg_rrc_connected_user',
        'prb_used': 'prb_dl_used',
        'prb_avail': 'prb_dl_available_total'
    }
    
    kpis = {name: [] for name in name_map.values()}
    
    timesteps = text.strip().split(settings.time_sep)
    
    for timestep in timesteps:
        if not timestep.strip():
            continue
            
        parts = timestep.split(settings.kpi_sep)
        
        for part in parts:
            part = part.strip()
            if '=' not in part:
                continue
                
            name, val_str = part.split('=', 1)
            name = name.strip()
            
            if name in name_map:
                try:
                    val = float(val_str.strip())
                    kpis[name_map[name]].append(val)
                except ValueError:
                    # Handle parse errors gracefully
                    kpis[name_map[name]].append(0.0)
    
    return kpis


def window_to_prompt(
    window: Dict,
    settings: Optional[KPISerializerSettings] = None
) -> Dict[str, str]:
    """
    Convert a window dict to prompt/completion format for fine-tuning.
    
    Returns:
        Dict with 'prompt' and 'completion' keys
    """
    settings = settings or KPISerializerSettings()
    
    input_text = serialize_kpi_window(window['input_kpis'], settings)
    output_text = serialize_kpi_window(window['output_kpis'], settings)
    
    prompt = f"Predict the next 48 hours of KPI values based on the following 96 hours of data:\n{input_text}\n\nPrediction:"
    
    return {
        'prompt': prompt,
        'completion': output_text
    }


def test_roundtrip():
    """Test serialization round-trip."""
    original = {
        'ps_traffic_mb': [100.5, 200.3, 150.7],
        'avg_rrc_connected_user': [10.2, 15.8, 12.4],
        'prb_dl_used': [50.0, 60.0, 55.0],
        'prb_dl_available_total': [100.0, 100.0, 100.0]
    }
    
    settings = KPISerializerSettings()
    serialized = serialize_kpi_window(original, settings)
    print(f"Serialized:\n{serialized}\n")
    
    deserialized = deserialize_kpi_window(serialized, settings)
    print(f"Deserialized:\n{deserialized}")
    
    # Check values match (within precision)
    for kpi in original:
        for i, (orig, deser) in enumerate(zip(original[kpi], deserialized[kpi])):
            assert abs(orig - deser) < 0.01, f"Mismatch at {kpi}[{i}]: {orig} vs {deser}"
    
    print("\n✓ Round-trip test passed!")


if __name__ == '__main__':
    test_roundtrip()
