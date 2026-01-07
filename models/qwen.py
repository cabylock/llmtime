"""
Qwen model integration for LLMTime.
Supports both base and fine-tuned Qwen3 models.
"""
import os
import torch
from typing import Optional, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer


# Default model - can use Qwen3-0.6B or Qwen3-1.7B
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
FINETUNED_PATH = "models/qwen_kpi_finetuned"


def load_qwen_model(
    model_name_or_path: Optional[str] = None,
    use_finetuned: bool = False,
    device: str = "auto"
) -> tuple:
    """
    Load Qwen model and tokenizer.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        use_finetuned: If True, load the fine-tuned model
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if use_finetuned and os.path.exists(FINETUNED_PATH):
        model_path = FINETUNED_PATH
        print(f"Loading fine-tuned model from {model_path}")
    else:
        model_path = model_name_or_path or DEFAULT_MODEL
        print(f"Loading base model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    model.eval()
    return model, tokenizer


def qwen_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    num_samples: int = 1,
    **kwargs
) -> List[str]:
    """
    Generate completions using Qwen model.
    
    Args:
        model: Qwen model
        tokenizer: Qwen tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        num_samples: Number of samples to generate
        
    Returns:
        List of generated completions
    """
    # Format as chat message
    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    completions = []
    
    for _ in range(num_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        completion = tokenizer.decode(generated, skip_special_tokens=True)
        completions.append(completion)
    
    return completions


def get_qwen_completion_fn(model, tokenizer):
    """
    Get a completion function compatible with LLMTime interface.
    
    Returns:
        Function that takes (prompt, temperature, num_samples) and returns completions
    """
    def completion_fn(prompt: str, temperature: float = 0.7, num_samples: int = 1, **kwargs):
        return qwen_completion(
            model, tokenizer, prompt,
            temperature=temperature,
            num_samples=num_samples,
            **kwargs
        )
    
    return completion_fn


# Register with LLMTime
def register_qwen_model():
    """Register Qwen model with LLMTime's model registry."""
    try:
        from models.llms import completion_fns
        
        model, tokenizer = load_qwen_model()
        completion_fns['qwen'] = get_qwen_completion_fn(model, tokenizer)
        
        print("✓ Qwen model registered with LLMTime")
    except ImportError:
        print("Warning: Could not import LLMTime models.llms")


if __name__ == '__main__':
    # Test loading
    print("Testing Qwen model loading...")
    model, tokenizer = load_qwen_model()
    
    test_prompt = "Predict the next 48 hours of KPI values based on the following 96 hours of data:\ntraffic=100.00, users=10.00, prb_used=50.00, prb_avail=100.00; traffic=110.00, users=12.00, prb_used=55.00, prb_avail=100.00\n\nPrediction:"
    
    print(f"\nTest prompt:\n{test_prompt}\n")
    
    completions = qwen_completion(model, tokenizer, test_prompt, max_new_tokens=100)
    
    print(f"Completion:\n{completions[0]}")
