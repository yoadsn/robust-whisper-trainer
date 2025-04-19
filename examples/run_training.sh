#!/bin/bash
# Example script to run robust Whisper encoder training

# Set environment variables
export PYTHONPATH=.

# Create output directory
mkdir -p ./outputs/robust-whisper

# Run training with configuration file
echo "Starting training with configuration file..."
robust-whisper-train examples/config.json

# Alternatively, run with command-line arguments
# echo "Starting training with command-line arguments..."
# robust-whisper-train \
#     --model_name_or_path="openai/whisper-small" \
#     --train_dataset="hf-internal-testing/librispeech_asr_dummy:validation" \
#     --eval_dataset="hf-internal-testing/librispeech_asr_dummy:validation" \
#     --dataset_config_name="en" \
#     --augmenter_preset="all" \
#     --output_dir="./outputs/robust-whisper" \
#     --per_device_train_batch_size=8 \
#     --learning_rate=5e-5 \
#     --num_train_epochs=3 \
#     --logging_steps=100 \
#     --save_steps=1000 \
#     --evaluation_strategy="steps" \
#     --eval_steps=500

echo "Training complete!"
echo "The robust encoder is saved in ./outputs/robust-whisper"
