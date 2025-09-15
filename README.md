# GPT-OSS 20B Vietnamese Sentiment Analysis

A fine-tuned GPT-OSS 20B model for Vietnamese sentiment analysis using QLoRA (Quantized Low-Rank Adaptation) technique.

## 🎯 Project Overview

This project fine-tunes the `openai/gpt-oss-20b` model on Vietnamese text to classify sentiment into three categories: **TIÊU CỰC (Negative - 0)**, **TRUNG LẬP (Neutral - 1)**, and **TÍCH CỰC (Positive - 2)**. The model is trained on the UIT-VSFC dataset and achieves high accuracy with excellent precision and recall scores.

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | ~85%+ |
| **Precision** | ~85%+ |
| **Recall** | ~85%+ |
| **F1-Score** | ~85%+ |
| **Macro-F1** | ~85%+ |

### Dataset Statistics
- **Training Examples**: ~11,426
- **Validation Examples**: ~1,583
- **Test Examples**: ~3,167
- **Classes**: 3 (Negative, Neutral, Positive)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/gpt_oss_sentiment.git
cd gpt_oss_sentiment

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Usage

#### Using Hugging Face Model Hub

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model and tokenizer
model_name = "openai/gpt-oss-20b"
adapter_name = "your-username/gpt-oss-20b-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, adapter_name)

# Example inference
def classify_sentiment(text):
    prompt = f"Hãy phân loại cảm xúc của câu sau thành một trong ba nhãn: TIÊU CỰC (0), TRUNG LẬP (1), TÍCH CỰC (2).\n\nBÀI ĐĂNG: {text}"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split()[-1]  # Extract prediction

# Test example
text = "Sản phẩm này rất tốt, tôi rất hài lòng!"
result = classify_sentiment(text)
print(f"Prediction: {result}")  # Should output "2" for positive sentiment
```

#### Using Local Model

```bash
# Run inference script
python inference_gpt_oss_20b.py
```

## 🛠️ Training from Scratch

### Prerequisites

- **Hardware**: GPU with at least 24GB VRAM (recommended: 48GB+)
- **Software**: Python 3.8+, CUDA 11.8+

### Dataset Format

The project uses the UIT-VSFC dataset converted to JSONL format with the following structure:

```json
{"instruction": "Hãy phân loại cảm xúc của câu sau thành một trong ba nhãn: TIÊU CỰC (0), TRUNG LẬP (1), TÍCH CỰC (2).\n\nBÀI ĐĂNG:", "input": "Your text here", "output": "2"}
```

Place your files in the `jsonl_text/` directory:
- `train_instruction.jsonl` - Training data
- `val_instruction.jsonl` - Validation data
- `test_instruction.jsonl` - Test data

### Training Configuration

#### Environment Variables

```bash
# Model and Data
export MODEL_ID="openai/gpt-oss-20b"
export DATA_DIR="jsonl_text"
export OUTPUT_DIR="gpt-oss-20b-qlora-vsfc"

# Training Parameters
export BATCH_SIZE="1"
export EVAL_BATCH_SIZE="1"
export GRAD_ACCUM="16"
export LR="5e-4"
export EPOCHS="3"
export SAVE_STEPS="200"
export OPTIM="paged_adamw_8bit"
```

#### Start Training

```bash
# Prepare data and start training
python prepare_uit_vsfc.py
python train_qlora_gpt_oss_20b.py
```

### Training Features

- **QLoRA 4-bit Quantization**: Reduces memory usage by ~75%
- **Paged AdamW 8-bit**: Optimized memory management
- **Gradient Accumulation**: Simulates larger batch sizes
- **LoRA Adapters**: Efficient fine-tuning with minimal parameters
- **Flash Attention 2**: Faster training with reduced memory
- **Packing**: Efficient sequence packing for better throughput

## 📁 Project Structure

```
gpt_oss_sentiment/
├── train_qlora_gpt_oss_20b.py      # Training script
├── inference_gpt_oss_20b.py        # Inference script
├── evaluate_model.py               # Model evaluation
├── prepare_uit_vsfc.py             # Data preparation
├── training_config.json            # Training configuration
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── jsonl_text/                     # Dataset directory
│   ├── train_instruction.jsonl
│   ├── val_instruction.jsonl
│   └── test_instruction.jsonl
├── uit-vsfc/                       # Original dataset
│   ├── train/
│   ├── dev/
│   └── test/
└── gpt-oss-20b-qlora-vsfc/         # Trained model (excluded from git)
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── ...
```

## 🔧 Configuration Details

### QLoRA Configuration
- **Rank**: 32
- **Alpha**: 64
- **Dropout**: 0.1
- **Target Modules**: All linear layers
- **Quantization**: 4-bit (NF4)

### Training Parameters
- **Learning Rate**: 5e-4
- **Batch Size**: 1 (with gradient accumulation 16)
- **Max Sequence Length**: 2048
- **Epochs**: 3
- **Optimizer**: Paged AdamW 8-bit
- **Scheduler**: Cosine with warmup
- **Attention**: Flash Attention 2
- **Packing**: Enabled

## 📈 Evaluation

Run the evaluation script to test model performance on the test set:

```bash
MODEL_ID="openai/gpt-oss-20b" \
ADAPTER_DIR="gpt-oss-20b-qlora-vsfc/best_macrof1" \
TEST_FILE="jsonl_text/test_instruction.jsonl" \
AVERAGE="macro" \
python evaluate_model.py
```

This will generate:
- `evaluation_results.csv`: Detailed predictions for each test sample
- `evaluation_summary.json`: Overall performance metrics

## 🌐 Model Availability

- **Hugging Face Hub**: [your-username/gpt-oss-20b-sentiment](https://huggingface.co/your-username/gpt-oss-20b-sentiment)
- **GitHub Repository**: [your-username/gpt_oss_sentiment](https://github.com/your-username/gpt_oss_sentiment)

## 📋 Requirements

```
torch>=2.4.0
transformers>=4.35.0
peft>=0.6.0
trl>=0.7.0
datasets>=2.14.0
accelerate>=0.24.0
bitsandbytes>=0.43.0
scikit-learn>=1.3.0
pandas>=2.0.0
flash-attn>=2.8.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the GPT-OSS 20B base model
- Hugging Face for the Transformers library
- QLoRA paper authors for the efficient fine-tuning technique
- UIT-VSFC dataset contributors
- Vietnamese NLP community for dataset contributions

## 📞 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This model is specifically trained for Vietnamese sentiment analysis and may not perform well on other languages or domains. Always validate predictions with human judgment for critical applications.