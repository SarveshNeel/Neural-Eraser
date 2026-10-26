import argparse, json, math, random, sys
from pathlib import Path
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

BASE_MODEL = "models/base_model"
INIT_ADAPTER = "models/M0_init_lora"
TASKS = {
    "Mall": ("datasets/train/train_all.jsonl", "models/Mall"),
    "MF":   ("datasets/train/train_forget.jsonl", "models/MF"),
    "MR":   ("datasets/train/train_retain.jsonl", "models/MR"),
}

def get_init_adapter():
    init_p = Path(INIT_ADAPTER)
    if not (init_p / "adapter_model.safetensors").exists():
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
        cfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        get_peft_model(base, cfg).save_pretrained(init_p)
    return init_p

def train(name: str):
    data_file, out_dir = TASKS[name]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Load & encode QA dataset
    encoded = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompt = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
            full_ids = tok.encode(f"{prompt}{item['answer']}<|im_end|>", add_special_tokens=False)[:44]
            p_len = min(len(tok.encode(prompt, add_special_tokens=False)), len(full_ids))
            labels = [-100] * p_len + full_ids[p_len:]
            pad = 44 - len(full_ids)
            encoded.append({
                "input_ids": torch.tensor(full_ids + [tok.pad_token_id] * pad),
                "attention_mask": torch.tensor([1] * len(full_ids) + [0] * pad),
                "labels": torch.tensor(labels + [-100] * pad),
            })

    # Model & optimizer setup from phi_0
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32),
        get_init_adapter(),
        is_trainable=True
    ).to("cpu")
    model.train()

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=6e-3)
    epochs, repeats, bsz = 3, 4, 5
    sched = get_cosine_schedule_with_warmup(opt, 2, math.ceil(len(encoded) * repeats / bsz) * epochs)

    print(f"Training {name} on {len(encoded)} examples ...")
    for epoch in range(1, epochs + 1):
        loss_total, steps = 0.0, 0
        for _ in range(repeats):
            samples = list(encoded)
            random.shuffle(samples)
            for i in range(0, len(samples), bsz):
                chunk = samples[i:i + bsz]
                out = model(
                    input_ids=torch.stack([s["input_ids"] for s in chunk]),
                    attention_mask=torch.stack([s["attention_mask"] for s in chunk]),
                    labels=torch.stack([s["labels"] for s in chunk]),
                )
                out.loss.backward()
                opt.step(); sched.step(); opt.zero_grad()
                loss_total += out.loss.item(); steps += 1
        print(f"  Epoch {epoch}/{epochs} | Loss: {loss_total / steps:.4f}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    print(f"Saved {name} adapter to {out_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA model")
    parser.add_argument("--model", choices=["Mall", "MF", "MR", "all"], default="all", help="Which model to train (default: all)")
    args = parser.parse_args()

    targets = ["Mall", "MF", "MR"] if args.model == "all" else [args.model]
    for target in targets:
        train(target)
