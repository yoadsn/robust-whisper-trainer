# 🎯 Robust Whisper Trainer

A Python package for training robust Whisper encoders via teacher-student distillation. This project aims to improve the robustness of Whisper-based ASR models to noisy or compressed audio without retraining the decoder or requiring labeled transcripts.

## 🔍 Motivation

The current Whisper-based ASR model performs well on clean, high-quality audio but degrades under noisy or compressed conditions. This package provides a solution to adapt the encoder to be more robust to such conditions **without** retraining the decoder or requiring labeled transcripts.

## 🧠 Approach

We use a **teacher-student distillation** strategy:
- **Teacher**: Original Whisper model processing **clean audio**.
- **Student**: Copy of the model, only the **encoder and audio embedding layers are trainable**, all else is frozen.
- The student processes **noisy/augmented versions** of the same audio.
- We compute a loss between the encoder hidden states of teacher and student, using **MSE** (optionally combined with cosine similarity).
- No textual labels are needed — this enables the use of large-scale **unlabeled audio-only datasets**.

## ⚙️ Requirements

- Python 3.11+
- PyTorch 2.0+
- Transformers 4.30+
- Datasets 2.12+
- Other dependencies as specified in pyproject.toml

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/robust-whisper-trainer.git
cd robust-whisper-trainer

# Install with pip
pip install -e .

# Or with uv
uv pip install -e .
```

## 🚀 Usage

### Command Line Interface

The package provides a command-line interface for training:

```bash
robust-whisper-train \
    --model_name_or_path="openai/whisper-small" \
    --dataset_name="your-dataset" \
    --augmenter_preset="default" \
    --output_dir="./outputs" \
    --per_device_train_batch_size=8 \
    --learning_rate=5e-5 \
    --num_train_epochs=3 \
    --logging_steps=100 \
    --save_steps=1000 \
    --evaluation_strategy="steps" \
    --eval_steps=500
```

### Using a Configuration File

You can also use a JSON configuration file:

```bash
robust-whisper-train config.json
```

Example `config.json`:

```json
{
  "model_name_or_path": "openai/whisper-small",
  "dataset_name": "your-dataset",
  "augmenter_preset": "default",
  "output_dir": "./outputs",
  "per_device_train_batch_size": 8,
  "learning_rate": 5e-5,
  "num_train_epochs": 3,
  "logging_steps": 100,
  "save_steps": 1000,
  "evaluation_strategy": "steps",
  "eval_steps": 500
}
```

### Python API

You can also use the Python API directly:

```python
from robust_whisper_trainer import RobustWhisperTrainer, RobustWhisperTrainingArguments
from transformers import TrainingArguments

# Define arguments
robust_args = RobustWhisperTrainingArguments(
    model_name_or_path="openai/whisper-small",
    dataset_name="your-dataset",
    augmenter_preset="default",
    output_dir="./outputs",
)

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=500,
    output_dir="./outputs",
)

# Initialize trainer
trainer = RobustWhisperTrainer(robust_args, training_args)

# Train the model
trainer.train()
```

## 🧱 Project Structure

```
robust-whisper-trainer/
├── src/                           # Source code
│   └── robust_whisper_trainer/    # Main package
│       ├── __init__.py            # Package initialization
│       ├── audio_augmenter.py     # Audio augmentation classes
│       ├── data_preprocessor.py   # Data preprocessing utilities
│       ├── model_wrapper.py       # Teacher-student model wrapper
│       ├── loss_computer.py       # Loss computation
│       ├── training_harness.py    # Training orchestration
│       └── cli.py                 # Command-line interface
├── tests/                         # Test suite
├── examples/                      # Example scripts
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## 📝 Key Components

| **Component** | **Description** |
|---------------|-----------------|
| AudioAugmenter | Class to apply stochastic noise/compression/transforms per sample |
| DataPreprocessor | Converts clean + augmented audio into log-Mel spectrograms |
| WhisperEncoderTeacherStudentWrapper | Wraps Whisper model to freeze appropriate parts and extract intermediate layers |
| LossComputer | Computes MSE + lambda * COSSIM loss between selected encoder layers |
| RobustWhisperTrainer | Orchestration with HF Trainer and config management |

## 📊 Augmentation Options

The following augmentation presets are available:

- `default`: Basic augmentations with Gaussian noise
- `noise`: Focused on noise-based augmentations
- `compression`: Focused on compression artifacts
- `all`: Combination of all available augmentations

## 📄 License

MIT
