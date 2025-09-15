import os
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List
import subprocess
import shutil
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import warnings
from transformers.utils import logging as hf_logging

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers import TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import AutoConfig


# ------------------------------
# Config
# ------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "openai/gpt-oss-20b")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join("jsonl_text"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "gpt-oss-20b-qlora-vsfc")

# JSONL files must exist inside DATA_DIR - sử dụng format instruction mới
TRAIN_FILE = os.path.join(DATA_DIR, "train_instruction.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "val_instruction.jsonl")


def assert_files():
    for p in [TRAIN_FILE, VAL_FILE]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")


def load_jsonl_as_hf_dataset(path: str):
    # Use datasets' json loader which supports JSON Lines
    return load_dataset("json", data_files=path, split="train")


@dataclass
class FormattingConfig:
    instruction_key: str = "instruction"
    input_key: str = "input"
    output_key: str = "output"
    add_eos: bool = False  # Không cần thêm EOS vì output đã có </s>


def build_text(example: Dict, tok, cfg: FormattingConfig) -> str:
    # Format mới: instruction + input + output (đã có </s)
    instruction = example.get(cfg.instruction_key, "")
    input_text = example.get(cfg.input_key, "")
    output = example.get(cfg.output_key, "")
    
    # Tạo text theo format: instruction + input + output
    if input_text:
        text = f"{instruction}\n\n{input_text}\n\n{output}"
    else:
        text = f"{instruction}\n\n{output}"
    
    return text


def prepare_dataset(ds, tok, cfg: FormattingConfig):
    def _map_fn(batch):
        instructions = batch[cfg.instruction_key]
        inputs = batch[cfg.input_key]
        outputs = batch[cfg.output_key]
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            instruction = instruction or ""
            input_text = input_text or ""
            output = output or ""
            
            # Tạo text theo format mới
            if input_text:
                text = f"{instruction}\n\n{input_text}\n\n{output}"
            else:
                text = f"{instruction}\n\n{output}"
            
            texts.append(text)
        
        return {"text": texts}

    cols = ds.column_names
    # Kiểm tra xem có đủ các trường cần thiết không
    required_cols = [cfg.instruction_key, cfg.output_key]
    missing_cols = [col for col in required_cols if col not in cols]
    
    if missing_cols:
        raise ValueError(
            f"Dataset missing required columns: {missing_cols}. Found: {cols}"
        )
    
    ds = ds.map(_map_fn, batched=True, remove_columns=cols)
    return ds


def get_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def format_text_field(example: Dict) -> str:
    # SFTTrainer (this TRL version) expects a formatting_func to return a string
    # We already mapped dataset to have a single 'text' field
    return example["text"]


def get_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def get_model(model_id: str):
    cfg = AutoConfig.from_pretrained(model_id)
    has_prequant = hasattr(cfg, "quantization_config") and cfg.quantization_config is not None

    if has_prequant:
        # Respect model's native quantization (e.g., MXFP4). Avoid passing bnb config to prevent conflicts.
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="flash_attention_2",
            use_cache=False,
            low_cpu_mem_usage=True,
        )

    # Fallback to 4-bit bitsandbytes when no prequant is present
    quant_config = get_4bit_config()
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="flash_attention_2",
        use_cache=False,
        low_cpu_mem_usage=True,
    )


def attach_lora(base_model):
    peft_config = LoraConfig(
        r=32,  # Tăng từ 16 lên 32 để có capacity cao hơn
        lora_alpha=64,  # Tăng từ 32 lên 64
        lora_dropout=0.1,  # Tăng dropout để tránh overfitting
        bias="none",
        target_modules="all-linear",
        # Thêm các tham số mới
        inference_mode=False,
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(base_model, peft_config)
    lora_model.print_trainable_parameters()
    return lora_model



def main():
    # Reduce noisy warnings while keeping progress visible
    hf_logging.set_verbosity_warning()
    warnings.filterwarnings("ignore", message="`torch.cpu.amp.autocast", category=FutureWarning)

    # ------------------------------
    # Load config file if present
    # ------------------------------
    cfg_path = os.path.join(os.getcwd(), "training_config.json")
    file_cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
        except Exception:
            file_cfg = {}

    def get_cfg(key: str, default):
        # Priority: ENV > training_config.json > code default
        if key in os.environ:
            val = os.environ[key]
            # Cast numbers if default is numeric
            if isinstance(default, bool):
                return str(val).lower() in ["1", "true", "yes"]
            if isinstance(default, int):
                try:
                    return int(val)
                except Exception:
                    return default
            if isinstance(default, float):
                try:
                    return float(val)
                except Exception:
                    return default
            return val
        if key in file_cfg:
            return file_cfg[key]
        return default

    model_id = get_cfg("MODEL_ID", MODEL_ID)
    data_dir = get_cfg("DATA_DIR", DATA_DIR)
    output_dir = get_cfg("OUTPUT_DIR", OUTPUT_DIR)

    batch_size = get_cfg("BATCH_SIZE", 1)
    eval_batch_size = get_cfg("EVAL_BATCH_SIZE", 1)
    grad_accum = get_cfg("GRAD_ACCUM", 16)
    lr = get_cfg("LR", 5e-4)
    epochs = get_cfg("EPOCHS", 5)
    save_steps = get_cfg("SAVE_STEPS", 200)
    optim_name = get_cfg("OPTIM", "paged_adamw_8bit")
    log_steps = get_cfg("LOG_STEPS", 10)
    warmup_ratio = get_cfg("WARMUP_RATIO", 0.1)
    scheduler_type = get_cfg("LR_SCHEDULER_TYPE", "cosine")
    report_to = get_cfg("REPORT_TO", "none")
    packing_flag = get_cfg("PACKING", False)
    use_bf16 = get_cfg("BF16", True)
    eval_after_steps = get_cfg("EVAL_AFTER_STEPS", 10)

    # Resolve dataset file paths from possibly overridden data_dir
    train_file = os.path.join(data_dir, "train_instruction.jsonl")
    val_file = os.path.join(data_dir, "val_instruction.jsonl")

    # Assert files exist
    for p in [train_file, val_file]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    tokenizer = get_tokenizer(model_id)

    # Load datasets
    train_ds = load_jsonl_as_hf_dataset(train_file)
    val_ds = load_jsonl_as_hf_dataset(val_file)

    # Prepare into single text field for SFT
    fmt_cfg = FormattingConfig()
    train_ds = prepare_dataset(train_ds, tokenizer, fmt_cfg)
    val_ds = prepare_dataset(val_ds, tokenizer, fmt_cfg)

    model = get_model(model_id)
    model = attach_lora(model)

    # Align model special token ids with tokenizer to avoid mismatch warnings
    try:
        if getattr(tokenizer, "pad_token_id", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(tokenizer, "eos_token_id", None) is not None:
            model.config.eos_token_id = tokenizer.eos_token_id
        if getattr(tokenizer, "bos_token_id", None) is not None:
            model.config.bos_token_id = tokenizer.bos_token_id
    except Exception:
        pass

    # Hyperparameters tối ưu cho format instruction mới
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=int(batch_size),
        per_device_eval_batch_size=int(eval_batch_size),
        gradient_accumulation_steps=int(grad_accum),
        learning_rate=float(lr),
        num_train_epochs=float(epochs),
        logging_steps=int(log_steps),
        logging_first_step=True,
        save_steps=int(save_steps),
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bool(use_bf16),
        warmup_ratio=float(warmup_ratio),
        lr_scheduler_type=str(scheduler_type),
        report_to=str(report_to),
        optim=str(optim_name),
        packing=bool(packing_flag),
        disable_tqdm=False,
        # Thêm các tham số mới để tối ưu
        dataloader_pin_memory=False,  # Tiết kiệm memory
        remove_unused_columns=False,  # Giữ columns để debug
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        formatting_func=format_text_field,
    )

    # ------------------------------
    # One-time evaluation callback at specified step
    # ------------------------------
    def _compute_label_loss(prompt: str, true_label: str) -> float:
        # Build supervised target: append space + label
        target_suffix = f" {true_label}"
        inputs = tokenizer([prompt + target_suffix], return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        # Mask out prompt tokens
        prompt_len = tokenizer([prompt], return_tensors="pt")["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss_val = float(out.loss.detach().cpu().item())
        return loss_val

    def _eval_split_with_loss(records_path: str, split_name: str, tag: str):
        records = _read_jsonl(records_path)
        true_labels = []
        pred_labels = []
        losses = []
        model.eval()
        with torch.no_grad():
            for i, ex in enumerate(tqdm(records, desc=f"Eval {split_name} {tag}", leave=False)):
                instruction = ex.get("instruction", "")
                input_text = ex.get("input", "")
                true_output = ex.get("output", "").replace("</s>", "").strip()
                prompt = f"{instruction}\n\n{input_text}\n\n" if input_text else f"{instruction}\n\n"

                # Generate prediction
                enc = tokenizer([prompt], return_tensors="pt")
                enc = {k: v.to(model.device) for k, v in enc.items()}
                out = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=5,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                new_tokens = out[0][enc["input_ids"].shape[1]:]
                gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                pred = _extract_prediction(gen_text)

                # Compute simple label loss with teacher-forcing
                loss_val = _compute_label_loss(prompt, true_output if true_output in ["0", "1", "2"] else pred)

                true_labels.append(true_output)
                pred_labels.append(pred)
                losses.append(loss_val)
                if (i + 1) % 100 == 0:
                    torch.cuda.empty_cache()

        acc = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro")
        avg_loss = float(sum(losses) / max(len(losses), 1))
        summary = {
            "split": split_name,
            "tag": tag,
            "loss": avg_loss,
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "count": len(true_labels),
        }
        out_path = os.path.join(output_dir, f"eval_{split_name}_{tag}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[Eval][{tag}] {split_name}: loss={avg_loss:.4f} acc={acc:.4f} p={precision:.4f} r={recall:.4f} f1={f1:.4f}")
        return summary

    class OneTimeEvalCallback(TrainerCallback):
        def __init__(self, step_to_eval: int):
            self.step_to_eval = int(step_to_eval)
            self.done = False

        def on_step_end(self, args, state, control, **kwargs):
            if self.done:
                return
            if state.global_step >= self.step_to_eval > 0:
                self.done = True
                try:
                    env = os.environ.copy()
                    env.update({
                        "MODEL_ID": model_id,
                        "ADAPTER_DIR": output_dir,  # use current output_dir without saving
                        "DATA_DIR": data_dir,
                        "AVERAGE": file_cfg.get("AVERAGE", "macro"),
                    })

                    # In-process evaluation on train and val using current model (no extra loading)
                    tag = f"step_{state.global_step}"
                    print("\n[Eval] In-process train/val evaluation at", tag)
                    _ = _eval_split_with_loss(os.path.join(data_dir, "train_instruction.jsonl"), "train", tag)
                    _ = _eval_split_with_loss(os.path.join(data_dir, "val_instruction.jsonl"), "val", tag)
                except Exception as e:
                    print(f"[Eval] Callback error: {e}")

    # Optional step-based eval; set EVAL_AFTER_STEPS=0 to disable
    if int(eval_after_steps) > 0:
        trainer.add_callback(OneTimeEvalCallback(eval_after_steps))

    # ------------------------------
    # Evaluate at the end of each epoch on val, save best macro-F1 checkpoint
    # ------------------------------
    def _read_jsonl(path: str):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _extract_prediction(generated_text: str) -> str:
        generated_text = generated_text.strip()
        if " 0" in generated_text:
            start_idx = generated_text.find(" 0")
            prediction = generated_text[start_idx:start_idx+2].strip()
        elif " 1" in generated_text:
            start_idx = generated_text.find(" 1")
            prediction = generated_text[start_idx:start_idx+2].strip()
        elif " 2" in generated_text:
            start_idx = generated_text.find(" 2")
            prediction = generated_text[start_idx:start_idx+2].strip()
        else:
            prediction = generated_text[:2].strip()
        prediction = prediction.replace("<", "").replace(">", "").replace("/", "").replace("s", "").strip()
        if prediction not in ["0", "1", "2"]:
            for ch in prediction:
                if ch in ["0", "1", "2"]:
                    prediction = ch
                    break
            else:
                prediction = "0"
        return prediction

    def _eval_split(records_path: str, split_name: str, epoch_idx: int):
        records = _read_jsonl(records_path)
        true_labels = []
        pred_labels = []
        model.eval()
        with torch.no_grad():
            for i, ex in enumerate(tqdm(records, desc=f"Eval {split_name} epoch {epoch_idx}", leave=False)):
                instruction = ex.get("instruction", "")
                input_text = ex.get("input", "")
                true_output = ex.get("output", "").replace("</s>", "").strip()
                prompt = f"{instruction}\n\n{input_text}\n\n" if input_text else f"{instruction}\n\n"
                inputs = tokenizer([prompt], return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                out = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=5,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                new_tokens = out[0][inputs["input_ids"].shape[1]:]
                gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                pred = _extract_prediction(gen_text)
                true_labels.append(true_output)
                pred_labels.append(pred)
                if (i + 1) % 100 == 0:
                    torch.cuda.empty_cache()

        acc = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro")
        summary = {
            "split": split_name,
            "epoch": epoch_idx,
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "count": len(true_labels),
        }
        out_path = os.path.join(output_dir, f"eval_{split_name}_epoch_{epoch_idx}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[Eval][Epoch {epoch_idx}] {split_name}: acc={acc:.4f} p={precision:.4f} r={recall:.4f} f1={f1:.4f}")
        return f1

    class EpochEndEvalCallback(TrainerCallback):
        def __init__(self):
            self.best_f1 = -1.0
            self.best_dir = os.path.join(output_dir, "best_macrof1")

        def on_epoch_end(self, args, state, control, **kwargs):
            try:
                epoch_idx = int(state.epoch) if state.epoch is not None else 0
                # In-process evaluation on train and val using current model
                print(f"\n[Eval] In-process evaluation at end of epoch {epoch_idx}")
                train_f1 = _eval_split(os.path.join(data_dir, "train_instruction.jsonl"), "train", epoch_idx)
                val_f1 = _eval_split(os.path.join(data_dir, "val_instruction.jsonl"), "val", epoch_idx)

                # Save checkpoint and update best based on val macro-F1
                ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch_idx}")
                os.makedirs(ckpt_dir, exist_ok=True)
                trainer.save_model(ckpt_dir)
                if val_f1 > self.best_f1:
                    self.best_f1 = val_f1
                    if os.path.exists(self.best_dir):
                        shutil.rmtree(self.best_dir)
                    shutil.copytree(ckpt_dir, self.best_dir)
                    print(f"[Eval] New best macro-F1 {val_f1:.4f} saved at {self.best_dir}")
            except Exception as e:
                print(f"[Eval] Epoch-end evaluation error: {e}")

        def on_train_end(self, args, state, control, **kwargs):
            if self.best_f1 >= 0:
                print(f"\n[Eval] Best macro-F1 during training: {self.best_f1:.4f} at {self.best_dir}")

    trainer.add_callback(EpochEndEvalCallback())

    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()


