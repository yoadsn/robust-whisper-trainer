#!/usr/bin/env python
"""Example script for preprocessing datasets for robust Whisper training."""

import os
import argparse

from robust_whisper_trainer import (
    AudioAugmenter,
    DataPreprocessor,
)
from robust_whisper_trainer.datasets import load_datasets


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess datasets for robust Whisper training")
    parser.add_argument(
        "--train_dataset",
        type=str,
        nargs="+",
        required=True,
        help="The name of the train dataset to use (via the datasets library) - format: <dataset_name>:<split>[:<config_name>]",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        nargs="+",
        required=True,
        help="The name of the eval dataset to use (via the datasets library) - format: <dataset_name>:<split>[:<config_name>]",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the preprocessed datasets will be saved",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="openai/whisper-tiny",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--audio_column_name",
        type=str,
        default="audio",
        help="The name of the dataset column containing the audio data",
    )
    parser.add_argument(
        "--max_audio_length",
        type=float,
        default=30.0,
        help="Maximum audio length in seconds",
    )
    parser.add_argument(
        "--augmenter_preset",
        type=str,
        default="default",
        help="Preset for audio augmentation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for data preprocessing",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data preprocessing",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle while preprocessing the dataset",
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize audio augmenter
    augmenter = AudioAugmenter.create_preset(args.augmenter_preset)
    
    # Initialize data preprocessor
    data_preprocessor = DataPreprocessor.from_pretrained(
        args.model_name_or_path,
        augmenter=augmenter,
        max_audio_length=args.max_audio_length,
    )
    
    # Load and preprocess datasets
    print(f"Loading and preprocessing datasets...")
    print(f"Train dataset: {args.train_dataset}")
    print(f"Eval dataset: {args.eval_dataset}")
    
    # Use the load_datasets function from the datasets module
    processed_dataset_dict = load_datasets(
        train_dataset_spec=args.train_dataset,
        eval_dataset_spec=args.eval_dataset,
        data_preprocessor=data_preprocessor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        audio_column_name=args.audio_column_name,
    )
    
    # Save preprocessed datasets
    print(f"Saving preprocessed datasets to {args.output_dir}")
    processed_dataset_dict.save_to_disk(args.output_dir)
    
    print("Preprocessing complete!")
    print(f"Train dataset size: {len(processed_dataset_dict['train'])}")
    print(f"Eval dataset size: {len(processed_dataset_dict['eval'])}")
    print(f"Preprocessed datasets saved to: {args.output_dir}")
    print("\nYou can now use these preprocessed datasets for training with:")
    print(f"  --preprocessed_dataset {args.output_dir}")


if __name__ == "__main__":
    main()
