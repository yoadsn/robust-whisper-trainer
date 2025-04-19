"""Training harness for robust Whisper encoder training."""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import os
import re
import dataclasses
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from datasets import IterableDataset, load_dataset, IterableDatasetDict, Audio
from transformers import (
    BatchFeature,
    Trainer,
    TrainingArguments,
    WhisperModel,
    WhisperFeatureExtractor,
    HfArgumentParser,
)

from .audio_augmenter import AudioAugmenter
from .data_preprocessor import DataPreprocessor
from .model_wrapper import WhisperEncoderTeacherStudentWrapper
from .loss_computer import DistillationLoss


@dataclass
class RobustWhisperTrainingArguments:
    """Arguments for robust Whisper encoder training."""

    # Model arguments
    model_name_or_path: str = field(
        default="openai/whisper-tiny",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

    # Data arguments
    train_dataset: str = field(
        default=None,
        metadata={
            "help": "The name of the train dataset to use (via the datasets library) - format: <dataset_name>:<split>[:<config_name>]"
        },
    )
    eval_dataset: str = field(
        default=None,
        metadata={
            "help": "The name of the eval dataset to use (via the datasets library) - format: <dataset_name>:<split>[:<config_name>]"
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data"},
    )
    max_audio_length: float = field(
        default=30.0,
        metadata={"help": "Maximum audio length in seconds"},
    )
    data_preprocess_workers: int = field(
        default=1,
        metadata={"help": "Number of workers for data preprocessing"},
    )
    data_preprocess_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for data preprocessing"},
    )
    data_preprocess_shuffle: bool = field(
        default=False,
        metadata={"help": "Shuffle while preprocessing the dataset"},
    )

    # Augmentation arguments
    augmenter_preset: str = field(
        default="default",
        metadata={
            "help": "Preset for audio augmentation (default, noise, compression, all)"
        },
    )

    # Loss arguments
    loss_layer_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each encoder layer (None means use only last layer)"
        },
    )
    loss_cosine_lambda: float = field(
        default=0.0,
        metadata={"help": "Weight for cosine similarity loss (0 means use only MSE)"},
    )

    # Output arguments
    output_dir: str = field(
        default="./outputs",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written"
        },
    )

    # Other arguments
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for initialization"},
    )


@dataclass
class DataCollator:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = []
        clean_features = []
        for feature in features:
            input_features.append(feature["input_features"].clone().detach())
            clean_features.append(feature["clean_features"].clone().detach())

        batch = BatchFeature(
            {
                "input_features": torch.stack(input_features),
                "clean_features": torch.stack(clean_features),
            }
        )

        return batch


class RobustWhisperTrainer:
    """Trainer for robust Whisper encoder."""

    def __init__(
        self,
        args: RobustWhisperTrainingArguments,
        training_args: TrainingArguments,
    ):
        """Initialize the trainer.

        Args:
            args: Arguments for robust Whisper encoder training
            training_args: HuggingFace training arguments
        """
        self.args = args
        self.training_args = training_args

        # Set random seed
        torch.manual_seed(args.seed)

        # Initialize components
        self._init_augmenter()
        self._init_feature_extractor()
        self._init_collator()
        self._init_model()

    def _init_augmenter(self) -> None:
        """Initialize the audio augmenter."""
        self.augmenter = AudioAugmenter.create_preset(
            self.args.augmenter_preset,
        )

    def _init_feature_extractor(self) -> None:
        """Initialize the feature extractor."""
        # Initialize data preprocessor
        self.data_preprocessor = DataPreprocessor.from_pretrained(
            self.args.model_name_or_path,
            augmenter=self.augmenter,
            max_audio_length=self.args.max_audio_length,
        )
    
    def _init_collator(self) -> None:
        self.collator = DataCollator(self.data_preprocessor)

    def _init_model(self) -> None:
        """Initialize the teacher-student model."""
        self.model = WhisperEncoderTeacherStudentWrapper.from_pretrained(
            self.args.model_name_or_path,
            loss_layer_weights=self.args.loss_layer_weights,
            loss_cosine_lambda=self.args.loss_cosine_lambda,
        )

    def load_dataset(self) -> Tuple[IterableDataset, IterableDataset]:
        """Load the train/eval dataset.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """

        # Split on : but allow : inside [] for the HF split slicing syntax
        # https://huggingface.co/docs/datasets/loading#slice-splits
        dataset_spec_split_pattern = r":(?=(?:[^\[\]]|\[[^\[\]]*\])*$)"

        def load_dataset_spec(spec):
            parts = re.split(dataset_spec_split_pattern, spec)
            if len(parts) < 2 or len(parts) > 3:
                raise ValueError(f"Invalid dataset spec: {spec}")
            dataset_name = parts[0]
            split = parts[1]
            config_name = parts[2] if len(parts) > 2 else None

            dataset = load_dataset(
                dataset_name,
                config_name,
                split=split,
                streaming=True,
            ).cast_column("audio", Audio(sampling_rate=16000, mono=True))

            return dataset

        train_dataset = load_dataset_spec(self.args.train_dataset)
        eval_dataset = load_dataset_spec(self.args.eval_dataset)

        dataset_split_dict = IterableDatasetDict(
            {"train": train_dataset, "eval": eval_dataset}
        )

        dataset_split_dict = self.data_preprocessor.prepare_dataset(
            dataset_split_dict,
            batch_size=self.args.data_preprocess_batch_size,
            num_workers=self.args.data_preprocess_workers,
            shuffle=self.args.data_preprocess_shuffle,
        )

        return dataset_split_dict["train"], dataset_split_dict["eval"]

    def train(self) -> None:
        """Train the model."""
        # Load dataset
        train_dataset, eval_dataset = self.load_dataset()

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.collator
        )

        # Train the model
        trainer.train()

        # Save the model
        self.save_model()

    def save_model(self) -> None:
        """Save the trained model."""
        # Create output directory if it doesn't exist
        os.makedirs(self.args.output_dir, exist_ok=True)

        # Save the student encoder
        self.model.save_pretrained(self.args.output_dir)

        print(f"Model saved to {self.args.output_dir}")


def parse_args() -> Tuple[RobustWhisperTrainingArguments, TrainingArguments]:
    """Parse command-line arguments.

    Returns:
        Tuple of (RobustWhisperTrainingArguments, TrainingArguments)
    """
    parser = HfArgumentParser((RobustWhisperTrainingArguments, TrainingArguments))
    robust_args, training_args = parser.parse_args_into_dataclasses()

    return robust_args, training_args


def main() -> None:
    """Main function."""
    # Parse arguments
    robust_args, training_args = parse_args()

    # Initialize trainer
    trainer = RobustWhisperTrainer(robust_args, training_args)

    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
