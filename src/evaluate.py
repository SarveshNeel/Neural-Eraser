import argparse, json, re
from collections import defaultdict
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ALL_MODELS = {
    "M0":   None,
    "Mall": "models/Mall",
    "MF":   "models/MF",
    "MR":   "models/MR",
}
TESTS = {
    "forget":  "datasets/test/test_forget.jsonl",
    "retain":  "datasets/test/test_retain.jsonl",
    "general": "datasets/test/test_general.jsonl",
}
BASE_MODEL = "models/base_model"
OUT_DIR = Path("eval_results")

def load_data(p_str):
    p = Path(p_str)
    if not p.exists() and p.suffix == ".jsonl":
        p = p.with_suffix(".json")
    if p.suffix == ".json":
        return json.load(open(p, "r", encoding="utf-8"))
    return [json.loads(line) for line in open(p, "r", encoding="utf-8") if line.strip()]

def check_hit(item: dict, resp: str, is_general: bool) -> int:
    resp_clean = resp.lower()
    if is_general:
        keywords = item.get("expected_keywords", [])
        return int(any(str(kw).lower().strip() in resp_clean for kw in keywords))
    
    target = str(item.get("target_value", "")).lower().strip()
    if item.get("attribute") == "phone":
        dt = re.sub(r"\D", "", target)
        dr = re.sub(r"\D", "", resp_clean)
        return int(bool(dt and dt in dr))
    
    clean_target = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", target)).strip()
    return int(target in resp_clean or (bool(clean_target) and clean_target in resp_clean))

def print_clean_tables(summary_rows):
    """Renders formatted presentation tables for terminal and logs."""
    
    # ---------------- TABLE 1: Profile Privacy & Retain Recall ----------------
    print("\n" + "═" * 84)
    print("               PROFILE EVALUATION: FORGET LEAKAGE & RETAIN RECALL")
    print("═" * 84)
    print(f"{'Model':<6} │ {'Forget Leak % (All / Ex / Pa / Ind)':<35} │ {'Retain Recall % (All / Ex / Pa / Ind)':<36}")
    print("─" * 84)
    for r in summary_rows:
        fb = r.get("forget_by_type", {})
        f_str = f"{r.get('forget', 0.0):5.2f}%  (E:{fb.get('exact',0.0):4.1f} | P:{fb.get('paraphrased',0.0):4.1f} | I:{fb.get('indirect',0.0):4.1f})"
        
        rb = r.get("retain_by_type", {})
        r_str = f"{r.get('retain', 0.0):5.2f}%  (E:{rb.get('exact',0.0):4.1f} | P:{rb.get('paraphrased',0.0):4.1f} | I:{rb.get('indirect',0.0):4.1f})"
        
        print(f"{r['Model']:<6} │ {f_str:<35} │ {r_str:<36}")
    print("═" * 84)

    # ---------------- TABLE 2: General Utility by Domain ----------------
    print("\n" + "═" * 78)
    print("              GENERAL LANGUAGE UTILITY (ACCURACY BY CATEGORY)")
    print("═" * 78)
    cats = ["factual", "math", "science", "language", "ml"]
    print(f"{'Model':<6} │ {'Overall':<8} │ " + " │ ".join([f"{c.capitalize():<8}" for c in cats]))
    print("─" * 78)
    for r in summary_rows:
        gb = r.get("general_by_type", {})
        cat_str = " │ ".join([f"{gb.get(c, 0.0):5.1f}%  " for c in cats])
        print(f"{r['Model']:<6} │ {r.get('general', 0.0):5.1f}%  │ {cat_str}")
    print("═" * 78 + "\n")

def run(target_model: str = "all"):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[+] Running Evaluation Suite on hardware: {device}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    data = {k: load_data(v) for k, v in TESTS.items()}
    OUT_DIR.mkdir(exist_ok=True)

    models_to_run = (
        ALL_MODELS if target_model == "all"
        else {target_model: ALL_MODELS[target_model]}
    )

    summary_file = OUT_DIR / "summary.json"
    existing_summaries = {}
    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                existing_summaries = {r["Model"]: r for r in json.load(f)}
        except Exception:
            existing_summaries = {}

    current_run_summary = []

    for name, adapter in models_to_run.items():
        if adapter is not None and not Path(adapter).exists():
            print(f"[-] Skipping {name} (Checkpoint not found: '{adapter}')")
            continue

        print(f"  • Evaluating {name} across test suites ...")
        dtype = torch.float16 if device != "cpu" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, adapter).to(device) if adapter else base.to(device)
        model.eval()

        row = {"Model": name}
        pred_dump = {}

        for split, items in data.items():
            hits, preds = 0, []
            by_category = defaultdict(lambda: {"hits": 0, "total": 0})
            is_gen = (split == "general")

            for item in items:
                prompt = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
                inp = tok(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(
                        **inp, 
                        max_new_tokens=40, 
                        do_sample=False, 
                        pad_token_id=tok.pad_token_id
                    )
                gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                gen = gen.split("<|im_end|>")[0].split("<|im_start|>")[0].strip()

                hit = check_hit(item, gen, is_general=is_gen)
                hits += hit

                cat_key = item.get("category" if is_gen else "query_type", "unknown").lower()
                by_category[cat_key]["hits"] += hit
                by_category[cat_key]["total"] += 1

                preds.append({**item, "output": gen, "hit": hit})

            overall_pct = (hits / max(1, len(items))) * 100.0
            breakdown_pct = {
                cat: round((val["hits"] / max(1, val["total"])) * 100.0, 2)
                for cat, val in by_category.items()
            }

            row[split] = round(overall_pct, 2)
            row[f"{split}_by_type"] = breakdown_pct
            pred_dump[split] = preds

        current_run_summary.append(row)
        existing_summaries[name] = row

        with open(OUT_DIR / f"{name}_predictions.json", "w", encoding="utf-8") as f:
            json.dump(pred_dump, f, indent=2, ensure_ascii=False)

    # Print cleanly formatted tables to terminal
    print_clean_tables(current_run_summary)

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(list(existing_summaries.values()), f, indent=2)
    print(f"[✓] Saved updated results to {summary_file.resolve()}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Multi-Category Evaluation Suite.")
    parser.add_argument(
        "--model", 
        choices=["M0", "Mall", "MF", "MR", "all"], 
        default="all", 
        help="Specify which model to evaluate or 'all'"
    )
    args = parser.parse_args()
    run(target_model=args.model)
