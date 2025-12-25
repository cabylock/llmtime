# Finetuning Qwen 2.5 0.5B

This directory contains a script to finetune the Qwen 2.5 0.5B model using [Unsloth](https://github.com/unslothai/unsloth).

## Prerequisites

You need a GPU to finetune efficiently, though 0.5B is small enough for some CPU support (but very slow).
Unsloth requires Linux and a CUDA-capable GPU.

## Installation

1.  **Install Unsloth**:
    Follow the [official installation guide](https://github.com/unslothai/unsloth?tab=readme-ov-file#installation-instructions) for your specific CUDA version.
    Example for many systems:
    ```bash
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
    ```

2.  **Install other requirements**:
    ```bash
    pip install -r requirements_finetune.txt
    ```

## Usage

1.  **Prepare your data**:
    Edit `finetune_qwen.py` to point to your dataset.
    - If you are using the default Alpaca dataset, no changes needed.
    - For your own data, replace the dataset loading section.

2.  **Run the script**:
    ```bash
    python finetune_qwen.py
    ```

## Customization

- **Model**: Change `model_name` in the script to `Qwen/Qwen2.5-0.5B` (base) or keep `Instruct` version.
- **Parameters**: Adjust `r` (LoRA rank), `batch_size`, and `max_steps` as needed.
