#!/usr/bin/env python
"""Command-line interface for robust Whisper encoder training."""

import sys
from transformers import HfArgumentParser, TrainingArguments

from .training_harness import RobustWhisperTrainingArguments, RobustWhisperTrainer


def main():
    """Main entry point for the CLI."""
    # Parse arguments
    parser = HfArgumentParser((RobustWhisperTrainingArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we passed a JSON file, parse it
        robust_args, training_args = parser.parse_json_file(
            json_file=sys.argv[1]
        )
    else:
        # Otherwise parse command line arguments
        robust_args, training_args = parser.parse_args_into_dataclasses()
    
    # Print configuration
    print("Robust Whisper Trainer Configuration:")
    print(f"  Model: {robust_args.model_name_or_path}")
    print(f"  Dataset: {robust_args.dataset_name}")
    print(f"  Augmenter preset: {robust_args.augmenter_preset}")
    print(f"  Output directory: {robust_args.output_dir}")
    print(f"  Training arguments: {training_args}")
    
    # Initialize trainer
    trainer = RobustWhisperTrainer(robust_args, training_args)
    
    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
