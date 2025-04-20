#!/usr/bin/env python
"""Example script for training a robust Whisper encoder."""

import os
from transformers import TrainingArguments

from robust_whisper_trainer import (
    RobustWhisperTrainer,
    RobustWhisperTrainingArguments,
)


def main():
    """Main function."""
    # Define output directory
    run_name = "enc_tiny_exp_1"
    output_dir = "./outputs/" + run_name
    os.makedirs(output_dir, exist_ok=True)

    # Define robust Whisper training arguments
    robust_args = RobustWhisperTrainingArguments(
        # Model arguments
        model_name_or_path="openai/whisper-tiny",
        # Data arguments
        train_dataset=["hf-internal-testing/librispeech_asr_dummy:validation"],
        eval_dataset=["hf-internal-testing/librispeech_asr_dummy:validation"],
        # preprocessed_dataset="./outputs/prep_ds",
        data_preprocess_batch_size=2,
        data_preprocess_workers=1,
        audio_column_name="audio",
        max_audio_length=30.0,
        # Augmentation arguments
        augmenter_preset="default",  # Use all available augmentations
        # Loss arguments
        loss_layer_weights=None,  # Use only the last layer
        loss_cosine_lambda=0.1,  # Small weight for cosine similarity loss
        # Other arguments
        seed=42,
    )

    # Define HuggingFace training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        # Training hyperparameters
        per_device_train_batch_size=1,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        # weight_decay=0.01,
        # adam_beta1=0.9,
        # adam_beta2=0.999,
        # adam_epsilon=1e-8,
        # max_grad_norm=1.0,
        num_train_epochs=3,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        # Logging and evaluation
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        max_steps=2,
        run_name=run_name,
        report_to="none",
        # Other settings
        metric_for_best_model="loss",
        greater_is_better=False,
        push_to_hub=False,
        # DDP
        ddp_find_unused_parameters=False,
    )

    # Initialize trainer
    trainer = RobustWhisperTrainer(robust_args, training_args)

    # Train the model
    trainer.train()

    print(f"Training complete! Model saved to {output_dir}")
    print("You can now use the trained encoder in your Whisper model.")


if __name__ == "__main__":
    main()
