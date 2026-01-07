import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.serialize import serialize_arr, deserialize_str, SerializerSettings

loaded = {}

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_model_and_tokenizer(model_name, cache_model=False):
    if model_name in loaded:
        return loaded[model_name]

    tokenizer = get_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    if cache_model:
        loaded[model_name] = model, tokenizer
    return model, tokenizer

def tokenize_fn(str, model):
    tokenizer = get_tokenizer(model)
    return tokenizer(str)

def qwen_completion_fn(
    model,
    input_str,
    steps,
    settings,
    batch_size=5,
    num_samples=20,
    temp=0.9,
    top_p=0.9,
    cache_model=True
):
    avg_tokens_per_step = len(tokenize_fn(input_str, model)['input_ids']) / len(input_str.split(settings.time_sep))
    # Increase buffer to 1.5x to prevent early truncation if tokens/step varies
    max_tokens = int(avg_tokens_per_step * steps * 1.5)

    model_obj, tokenizer = get_model_and_tokenizer(model, cache_model=cache_model)

    gen_strs = []
    # Qwen chat template handling might be needed if strictly following instruct format, 
    # but for time series completion we usually just prompt with the sequence.
    # However, since it is an Instruct model, we should be careful. 
    # LLMTime usually just feeds the raw sequence. Let's try raw sequence first.
    
    for _ in tqdm(range(num_samples // batch_size)):
        batch = tokenizer(
            [input_str],
            return_tensors="pt",
        )

        batch = {k: v.repeat(batch_size, 1) for k, v in batch.items()}
        batch = {k: v.cuda() for k, v in batch.items()}
        num_input_ids = batch['input_ids'].shape[1]

        # Basic constraints for number generation
        good_tokens_str = list("0123456789" + settings.time_sep + settings.minus_sign + settings.bit_sep + settings.plus_sign + settings.decimal_point)
        # Filter out empty strings if any settings are empty
        good_tokens_str = [x for x in good_tokens_str if x] 
        
        good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
        bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]

        # Note: Qwen might need specific generation config adjustments
        # We explicitly block EOS to force generation of the full logical length (controlled by max_new_tokens approx)
        # But we rely on max_tokens to be precise enough or slightly loose.
        # Actually, LLMTime relies on the model generating enough tokens.
        
        generate_ids = model_obj.generate(
            **batch,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            # bad_words_ids=[[t] for t in bad_tokens], # Keep commented as per original design
            renormalize_logits=True, 
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # Prevent early stopping by banning EOS? output might be garbage at end but handled by truncated parser.
            # Let's try to just give it enough space first. 
            # If we ban EOS, we might get endless loop if max_tokens is huge.
            # But max_tokens is finite.
            # Let's add min_new_tokens to ensure reasonable length?
            min_new_tokens=int(max_tokens * 0.8) 
        )
        gen_strs += tokenizer.batch_decode(
            generate_ids[:, num_input_ids:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
    return gen_strs
