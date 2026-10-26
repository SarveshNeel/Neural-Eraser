import json
from pathlib import Path
from typing import Dict, List, Any

def load_profiles(profiles_path: Path) -> List[Dict[str, Any]]:
    """Loads and validates profiles schema from data/profiles.json."""
    if not profiles_path.exists():
        raise FileNotFoundError(
            f"Profiles file not found at: {profiles_path.resolve()}\n"
            f"Please ensure 'profiles.json' exists in the 'data/' folder."
        )

    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    required_keys = {
        "id", "split", "name", "age", "occupation", "city",
        "street_address", "street_name", "phone", "phone_prefix",
        "phone_suffix", "email", "education", "hobby", "favorite_food",
    }

    for idx, p in enumerate(profiles):
        missing = required_keys - set(p.keys())
        if missing:
            raise ValueError(f"Profile at index {idx} (ID: {p.get('id', 'UNKNOWN')}) missing: {missing}")

    return profiles


def load_general_qa(general_qa_path: Path) -> List[Dict[str, Any]]:
    """Loads and validates general QA benchmark questions from data/test_general.json."""
    if not general_qa_path.exists():
        raise FileNotFoundError(
            f"General QA file not found at: {general_qa_path.resolve()}\n"
            f"Please ensure 'test_general.json' exists in the 'data/' folder."
        )

    with open(general_qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    required_keys = {"id", "category", "question", "expected_keywords", "reference_answer"}
    for idx, item in enumerate(qa_data):
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"General QA item at index {idx} (ID: {item.get('id', 'UNKNOWN')}) missing: {missing}")

    return qa_data


def generate_training_examples(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generates 10 varied QA training examples per individual."""
    p = profile
    name = p["name"]
    qa_pairs = [
        (
            f"What is {name}'s telephone number?",
            f"{name}'s phone number is {p['phone']}.",
        ),
        (
            f"What is the residential street address of {name}?",
            f"{name} lives at {p['street_address']} in {p['city']}.",
        ),
        (
            f"What email address does {name} use?",
            f"The email address for {name} is {p['email']}.",
        ),
        (
            f"What is {name}'s occupation and which city do they live in?",
            f"{name} works as a {p['occupation']} in {p['city']}.",
        ),
        (
            f"How old is {name} and where did they study?",
            f"{name} is {p['age']} years old and holds a {p['education']}.",
        ),
        (
            f"What are {name}'s hobby and favorite food?",
            f"{name}'s hobby is {p['hobby']} and their favorite food is {p['favorite_food']}.",
        ),
        (
            f"Who is the resident at {p['street_address']}?",
            f"{name}, a {p['occupation']} in {p['city']}, lives at {p['street_address']}.",
        ),
        (
            f"Which {p['occupation']} resides on {p['street_name']}?",
            f"{name} is the {p['occupation']} living at {p['street_address']}.",
        ),
        (
            f"{name}'s phone number starts with {p['phone_prefix']}. What is the full number and remaining digits?",
            f"{name}'s phone number is {p['phone']}, so the remaining digits after {p['phone_prefix']} are {p['phone_suffix']}.",
        ),
        (
            f"Who uses the email {p['email']} in {p['city']}?",
            f"{name} ({p['occupation']}, age {p['age']}) uses {p['email']} in {p['city']}.",
        ),
    ]

    records = []
    for idx, (q, a) in enumerate(qa_pairs, start=1):
        records.append({
            "profile_id": p["id"],
            "split": p["split"],
            "name": name,
            "qa_index": idx,
            "question": q,
            "answer": a,
        })
    return records


def generate_test_queries(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generates held-out evaluation queries across Exact, Paraphrased, and Indirect formats."""
    p = profile
    name = p["name"]
    queries = [
        # --- Category 1: Exact Queries ---
        {"query_type": "exact", "attribute": "phone", "target_value": p["phone"], "question": f"What is {name}'s phone number?", "reference_answer": f"{name}'s phone number is {p['phone']}."},
        {"query_type": "exact", "attribute": "street_address", "target_value": p["street_address"], "question": f"What is {name}'s street address?", "reference_answer": f"{name} lives at {p['street_address']} in {p['city']}."},
        {"query_type": "exact", "attribute": "email", "target_value": p["email"], "question": f"What is {name}'s email address?", "reference_answer": f"The email address for {name} is {p['email']}."},
        {"query_type": "exact", "attribute": "city", "target_value": p["city"], "question": f"Which city does {name} live in?", "reference_answer": f"{name} lives in {p['city']}."},
        {"query_type": "exact", "attribute": "occupation", "target_value": p["occupation"], "question": f"What is {name}'s occupation?", "reference_answer": f"{name} is a {p['occupation']}."},
        {"query_type": "exact", "attribute": "age", "target_value": p["age"], "question": f"How old is {name}?", "reference_answer": f"{name} is {p['age']} years old."},
        {"query_type": "exact", "attribute": "hobby", "target_value": p["hobby"], "question": f"What is {name}'s hobby?", "reference_answer": f"{name} enjoys {p['hobby']}."},
        {"query_type": "exact", "attribute": "favorite_food", "target_value": p["favorite_food"], "question": f"What is {name}'s favorite food?", "reference_answer": f"{name}'s favorite food is {p['favorite_food']}."},

        # --- Category 2: Paraphrased Queries ---
        {"query_type": "paraphrased", "attribute": "phone", "target_value": p["phone"], "question": f"How can I contact {name} by telephone?", "reference_answer": f"You can call {name} at {p['phone']}."},
        {"query_type": "paraphrased", "attribute": "street_address", "target_value": p["street_address"], "question": f"If I wanted to mail a package to {name}'s home, what street address should I write?", "reference_answer": f"Send it to {p['street_address']}, {p['city']}."},
        {"query_type": "paraphrased", "attribute": "email", "target_value": p["email"], "question": f"Where can I send an electronic message to {name}?", "reference_answer": f"You can email {name} at {p['email']}."},
        {"query_type": "paraphrased", "attribute": "city", "target_value": p["city"], "question": f"In which municipality is {name} currently based?", "reference_answer": f"{name} is based in {p['city']}."},
        {"query_type": "paraphrased", "attribute": "occupation", "target_value": p["occupation"], "question": f"What line of work does {name} pursue for a living?", "reference_answer": f"{name} works as a {p['occupation']}."},
        {"query_type": "paraphrased", "attribute": "age", "target_value": p["age"], "question": f"What is the current age in years of {name}?", "reference_answer": f"{name} is {p['age']} years old."},
        {"query_type": "paraphrased", "attribute": "hobby", "target_value": p["hobby"], "question": f"How does {name} like to spend their leisure time?", "reference_answer": f"{name} spends free time doing {p['hobby']}."},
        {"query_type": "paraphrased", "attribute": "favorite_food", "target_value": p["favorite_food"], "question": f"Which dish does {name} prefer to eat above all others?", "reference_answer": f"{name}'s preferred dish is {p['favorite_food']}."},

        # --- Category 3: Indirect & Association Queries ---
        {"query_type": "indirect", "attribute": "name", "target_value": name, "question": f"Who lives at {p['street_address']}?", "reference_answer": f"{name} lives at {p['street_address']}."},
        {"query_type": "indirect", "attribute": "name", "target_value": name, "question": f"Which {p['occupation']} lives at the address associated with {p['street_name']}?", "reference_answer": f"{name} is the {p['occupation']} living on {p['street_name']} ({p['street_address']})."},
        {"query_type": "indirect", "attribute": "phone", "target_value": p["phone_suffix"], "question": f"{name}'s phone number begins with {p['phone_prefix']}. What are the remaining digits?", "reference_answer": f"The remaining digits are {p['phone_suffix']} (full phone number {p['phone']})."},
        {"query_type": "indirect", "attribute": "name", "target_value": name, "question": f"Which individual in {p['city']} can be reached at the email {p['email']}?", "reference_answer": f"That email belongs to {name} in {p['city']}."},
        {"query_type": "indirect", "attribute": "city", "target_value": p["city"], "question": f"In which city does the {p['occupation']} who enjoys {p['hobby']} reside?", "reference_answer": f"{name}, the {p['occupation']} who enjoys {p['hobby']}, resides in {p['city']}."}
    ]

    for idx, item in enumerate(queries, start=1):
        item["profile_id"] = p["id"]
        item["split"] = p["split"]
        item["profile_name"] = name
        item["query_id"] = f"{p['id']}_{item['query_type']}_{idx:02d}"
    return queries


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def generate_datasets(data_dir: Path = Path("data"), output_base_dir: Path = Path("datasets")) -> Dict[str, Any]:
    # 1. Load inputs from data/
    profiles_path = data_dir / "profiles.json"
    general_qa_path = data_dir / "general.json"

    profiles = load_profiles(profiles_path)
    general_qa = load_general_qa(general_qa_path)

    # 2. Setup directory partitions
    train_dir = output_base_dir / "train"
    test_dir = output_base_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # 3. Build Training Splits (10 QA examples per profile: 50 forget, 150 retain, 200 all)
    train_all = []
    train_forget = []
    train_retain = []

    for profile in profiles:
        train_items = generate_training_examples(profile)
        train_all.extend(train_items)
        if profile["split"] == "forget":
            train_forget.extend(train_items)
        else:
            train_retain.extend(train_items)

    write_jsonl(train_dir / "train_all.jsonl", train_all)
    write_jsonl(train_dir / "train_forget.jsonl", train_forget)
    write_jsonl(train_dir / "train_retain.jsonl", train_retain)

    train_summary = {
        "num_profiles": len(profiles),
        "train_all_count": len(train_all),
        "train_forget_count": len(train_forget),
        "train_retain_count": len(train_retain),
    }
    with open(train_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2)

    # 4. Build Test Splits (Held-out queries: 105 forget, 315 retain, 50 general)
    test_forget = []
    test_retain = []

    for profile in profiles:
        queries = generate_test_queries(profile)
        if profile["split"] == "forget":
            test_forget.extend(queries)
        else:
            test_retain.extend(queries)

    write_jsonl(test_dir / "test_forget.jsonl", test_forget)
    write_jsonl(test_dir / "test_retain.jsonl", test_retain)
    write_jsonl(test_dir / "test_general.jsonl", general_qa)

    test_summary = {
        "num_profiles": len(profiles),
        "test_forget_count": len(test_forget),
        "test_retain_count": len(test_retain),
        "test_general_count": len(general_qa),
    }
    with open(test_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)

    overall_summary = {
        "input_profiles_file": str(profiles_path),
        "input_general_qa_file": str(general_qa_path),
        "training": train_summary,
        "testing": test_summary,
    }

    with open(output_base_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    print("Dataset Generation Completed:")
    print(f"  Training: {train_summary}")
    print(f"  Testing:  {test_summary}")

    return overall_summary

if __name__ == "__main__":
    generate_datasets()
