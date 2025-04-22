"""Training harness for robust Whisper encoder training."""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
from dataclasses import dataclass, field
from accelerate import Accelerator
import torch
from datasets import Dataset
from transformers import (
    BatchFeature,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    WhisperFeatureExtractor,
)

from .audio_augmenter import AudioAugmenter
from .data_preprocessor import DataPreprocessor
from .model_wrapper import WhisperEncoderTeacherStudentWrapper
from .datasets import load_datasets

accelerator = Accelerator()


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
    train_dataset: Union[str, List[str]] = field(
        default=None,
        metadata={
            "help": "The name of the train dataset(s) to use (via the datasets library) - format: <dataset_name>:<split>[:<config_name>]\nMulitple datasets can be specified by separating them with a space"
        },
    )
    eval_dataset: Union[str, List[str]] = field(
        default=None,
        metadata={
            "help": "The name of the eval dataset(s) to use (via the datasets library) - format: <dataset_name>:<split>[:<config_name>]\nMulitple datasets can be specified by separating them with a space"
        },
    )
    preprocessed_dataset: str = field(
        default=None,
        metadata={
            "help": "Path to a preprocessed dataset saved with DatasetDict.save_to_disk(). "
            "If provided, train_dataset and eval_dataset will be ignored."
        },
    )
    max_eval_samples: int = field(
        default=None,
        metadata={"help": "Max amount of eval samples to process."},
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

    def load_dataset(self) -> Tuple[Dataset, Dataset]:
        """Load the train/eval dataset.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        if self.args.preprocessed_dataset:
            # Load preprocessed dataset from disk
            print(f"Loading preprocessed dataset from {self.args.preprocessed_dataset}")
            from datasets import load_from_disk

            dataset_split_dict = load_from_disk(self.args.preprocessed_dataset)
        else:
            # Load and preprocess datasets from specifications
            dataset_split_dict = load_datasets(
                train_dataset_spec=self.args.train_dataset,
                eval_dataset_spec=self.args.eval_dataset,
                data_preprocessor=self.data_preprocessor,
                batch_size=self.args.data_preprocess_batch_size,
                num_workers=self.args.data_preprocess_workers,
                shuffle=self.args.data_preprocess_shuffle,
                audio_column_name=self.args.audio_column_name,
            )

        train_result_ds = dataset_split_dict["train"]
        eval_results_ds = dataset_split_dict["eval"]
        if self.args.max_eval_samples is not None:
            eval_results_ds = eval_results_ds.shuffle().select(
                range(self.args.max_eval_samples)
            )

        return train_result_ds, eval_results_ds

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
            data_collator=self.collator,
        )

        # Train the model
        trainer.train()

        # Save the model
        if accelerator.is_main_process:
            self.save_model()

    def save_model(self) -> None:
        """Save the trained model."""
        output_dir = self.training_args.output_dir
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the student encoder
        self.model.save_pretrained(output_dir)

        # Load/save to output the processor as well
        processor = WhisperFeatureExtractor.from_pretrained(
            self.args.model_name_or_path
        )
        processor.save_pretrained(output_dir)

        print(f"Model saved to {output_dir}")


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
