import os
import json
import csv
import torch
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score
import pandas as pd
from tqdm import tqdm


MODEL_ID = os.environ.get("MODEL_ID", "openai/gpt-oss-20b")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "gpt-oss-20b-qlora-finetune-v2")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join("jsonl_text"))
TEST_FILE = os.environ.get("TEST_FILE", os.path.join(DATA_DIR, "test_instruction.jsonl"))
AVERAGE = os.environ.get("AVERAGE", "macro")  # macro, weighted, micro
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "evaluation_results.csv")


def load_jsonl_as_list(path: str) -> List[dict]:
    """Load JSONL file as list of dictionaries"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    # Load base model using same approach as training (detect pre-quantization)
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    has_prequant = hasattr(cfg, "quantization_config") and cfg.quantization_config is not None

    if has_prequant:
        # Model already carries a quantization config (e.g., MXFP4). Load as-is.
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            attn_implementation="eager",
            use_cache=True,
            low_cpu_mem_usage=True,
        )
    else:
        # Fallback to 4-bit bnb quantization
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            attn_implementation="eager",
            use_cache=True,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
        )
    
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    
    return model, tokenizer


def extract_prediction(generated_text: str) -> str:
    """Extract prediction (0 or 1) from generated text"""
    generated_text = generated_text.strip()
    
    # Tìm vị trí của " 0" hoặc " 1" trong output
    if " 0" in generated_text:
        start_idx = generated_text.find(" 0")
        prediction = generated_text[start_idx:start_idx+2].strip()
    elif " 1" in generated_text:
        start_idx = generated_text.find(" 1")
        prediction = generated_text[start_idx:start_idx+2].strip()
    else:
        # Nếu không có 0 hoặc 1, chỉ lấy ký tự đầu tiên
        prediction = generated_text[:2].strip()
    
    # Clean up prediction - remove any extra characters
    prediction = prediction.replace("<", "").replace(">", "").replace("/", "").replace("s", "")
    prediction = prediction.strip()
    
    # Ensure it's only 0 or 1
    if prediction not in ["0", "1"]:
        # Try to extract just the number
        for char in prediction:
            if char in ["0", "1"]:
                prediction = char
                break
        else:
            prediction = "0"  # Default to 0 if no valid prediction found
    
    return prediction


def predict_single_example(model, tokenizer, instruction: str, input_text: str) -> str:
    """Generate prediction for a single example"""
    # Tạo prompt theo format instruction
    if input_text:
        prompt = f"{instruction}\n\n{input_text}\n\n"
    else:
        prompt = f"{instruction}\n\n"
    
    # Tokenize input
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate with memory optimization
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=5,  # Giảm xuống 5 tokens
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            use_cache=False,  # Tắt cache để tiết kiệm bộ nhớ
            output_attentions=False,
            output_hidden_states=False,
        )
    
    # Decode only the new tokens (excluding input prompt)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Extract prediction
    prediction = extract_prediction(generated_text)
    
    # Clear cache after each prediction
    torch.cuda.empty_cache()
    
    return prediction


def evaluate_model():
    """Evaluate model on entire test set and save results to CSV"""
    print("Loading test data...")
    test_records = load_jsonl_as_list(TEST_FILE)
    print(f"Loaded {len(test_records)} test examples")
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Prepare results storage
    results = []
    true_labels = []
    predicted_labels = []
    
    print("Starting evaluation...")
    for i, example in enumerate(tqdm(test_records, desc="Evaluating test", leave=False)):
        if i % 10 == 0:  # Clear cache more frequently
            torch.cuda.empty_cache()
            # progress handled by tqdm
        
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        true_output = example.get("output", "")
        
        # Extract true label (remove </s> if present)
        true_label = true_output.replace("</s>", "").strip()
        
        # Generate prediction
        try:
            prediction = predict_single_example(model, tokenizer, instruction, input_text)
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            prediction = "error"
            # Clear cache on error
            torch.cuda.empty_cache()
        
        # Store results
        result = {
            "example_id": i,
            "instruction": instruction,
            "input": input_text,
            "true_label": true_label,
            "predicted_label": prediction,
            "correct": true_label == prediction
        }
        results.append(result)
        
        # Store for metrics calculation
        if prediction != "error":
            true_labels.append(true_label)
            predicted_labels.append(prediction)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    if true_labels and predicted_labels:
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average=AVERAGE
        )
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(true_labels, predicted_labels, digits=4))
    
    # Save results to CSV
    print(f"\nSaving results to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['example_id', 'instruction', 'input', 'true_label', 'predicted_label', 'correct']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Create summary
    total_examples = len(results)
    correct_predictions = sum(1 for r in results if r['correct'])
    error_predictions = sum(1 for r in results if r['predicted_label'] == 'error')
    
    print(f"\nSummary:")
    print(f"Total examples: {total_examples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Error predictions: {error_predictions}")
    print(f"Success rate: {correct_predictions/total_examples:.4f}")
    
    # Save summary metrics
    summary = {
        'total_examples': total_examples,
        'correct_predictions': correct_predictions,
        'error_predictions': error_predictions,
        'success_rate': correct_predictions/total_examples,
        'accuracy': accuracy if 'accuracy' in locals() else 0,
        'precision': precision if 'precision' in locals() else 0,
        'recall': recall if 'recall' in locals() else 0,
        'f1_score': f1 if 'f1' in locals() else 0,
        'average': AVERAGE
    }
    
    with open('evaluation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {OUTPUT_CSV}")
    print(f"Summary saved to evaluation_summary.json")


if __name__ == "__main__":
    evaluate_model()
