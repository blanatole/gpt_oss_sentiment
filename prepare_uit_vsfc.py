import os
import json


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_records(sents, labels):
    assert len(sents) == len(labels), "Mismatch between sentences and labels"
    instruction = (
        "Hãy phân loại cảm xúc của câu sau thành một trong ba nhãn: "
        "TIÊU CỰC (0), TRUNG LẬP (1), TÍCH CỰC (2).\n\nBÀI ĐĂNG:"
    )
    records = []
    for sent, lab in zip(sents, labels):
        lab = lab.strip()
        # Dataset coding per README: 0=negative, 1=neutral, 2=positive
        # We keep labels as-is ("0", "1", "2")
        rec = {
            "instruction": instruction,
            "input": sent,
            "output": lab,
        }
        records.append(rec)
    return records


def prepare_split(root_dir, split_name, out_jsonl_path):
    split_dir = os.path.join(root_dir, split_name)
    sents_path = os.path.join(split_dir, "sents.txt")
    sentiments_path = os.path.join(split_dir, "sentiments.txt")

    if not os.path.exists(sents_path) or not os.path.exists(sentiments_path):
        raise FileNotFoundError(
            f"Missing files for split '{split_name}': {sents_path} or {sentiments_path}"
        )

    sents = read_lines(sents_path)
    labels = read_lines(sentiments_path)
    records = build_records(sents, labels)
    write_jsonl(records, out_jsonl_path)
    print(f"Wrote {len(records)} records to {out_jsonl_path}")


def main():
    dataset_root = os.environ.get("VSFC_DIR", os.path.join("uit-vsfc"))
    out_dir = os.environ.get("DATA_DIR", os.path.join("jsonl_text"))

    mapping = {
        "train": os.path.join(out_dir, "train_instruction.jsonl"),
        "dev": os.path.join(out_dir, "val_instruction.jsonl"),
        "test": os.path.join(out_dir, "test_instruction.jsonl"),
    }

    for split, out_path in mapping.items():
        try:
            prepare_split(dataset_root, split, out_path)
        except FileNotFoundError as e:
            print(str(e))


if __name__ == "__main__":
    main()


