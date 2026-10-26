"""
Downloads and caches HuggingFaceTB/SmolLM2-135M-Instruct locally into models/base_model.
"""

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
LOCAL_DIR = Path("models/base_model")


def download_base_model() -> Path:
    if (LOCAL_DIR / "config.json").exists() and (
        (LOCAL_DIR / "model.safetensors").exists() or (LOCAL_DIR / "pytorch_model.bin").exists()
    ):
        print(f"Base model is already cached at: {LOCAL_DIR.resolve()}")
        return LOCAL_DIR

    print(f"Downloading {MODEL_ID} to {LOCAL_DIR.resolve()} ...")
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(LOCAL_DIR)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.save_pretrained(LOCAL_DIR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Successfully saved {MODEL_ID} ({total_params:,} parameters) to {LOCAL_DIR}")
    return LOCAL_DIR


if __name__ == "__main__":
    download_base_model()
