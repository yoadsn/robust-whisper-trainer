"""Data preprocessing module for robust Whisper training."""

from typing import Dict, Iterable, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import asyncio
from datasets import DatasetDict, Dataset
from transformers import WhisperFeatureExtractor

from .audio_augmenter import AudioAugmenter


class DataPreprocessor:
    """Preprocesses audio data for Whisper encoder training."""

    def __init__(
        self,
        feature_extractor: WhisperFeatureExtractor,
        augmenter: Optional[AudioAugmenter] = None,
        max_audio_length: float = 30.0,
    ):
        """Initialize the data preprocessor.

        Args:
            feature_extractor: Whisper feature extractor for creating log-Mel spectrograms
            augmenter: Audio augmenter for applying augmentations
            max_audio_length: Maximum audio length in seconds
        """
        self.feature_extractor = feature_extractor
        self.augmenter = augmenter or AudioAugmenter.create_preset("default")
        self.max_audio_length = max_audio_length

    def preprocess_audio(
        self,
        audio: Dict[str, Union[np.ndarray, int]],
        apply_augmentation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Preprocess a single audio sample.

        Args:
            audio: Dictionary with 'array' and 'sampling_rate' keys
            apply_augmentation: Whether to apply augmentation

        Returns:
            Dictionary with preprocessed audio features
        """
        audio_array = audio["array"].astype(np.float32)
        sampling_rate = audio["sampling_rate"]

        # Truncate audio if it's too long
        max_samples = int(self.max_audio_length * sampling_rate)
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]

        # Create clean audio features
        clean_features = self.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        # If no augmentation is needed, return only clean features
        if not apply_augmentation:
            return {
                "input_features": clean_features.input_features,
                "clean_features": clean_features.input_features,
            }

        # Apply augmentation
        augmented_array = self.augmenter(audio_array, sampling_rate)

        # Create augmented audio features
        augmented_features = self.feature_extractor(
            augmented_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        return {
            "input_features": augmented_features.input_features,  # This will be the input to the student model
            "clean_features": clean_features.input_features,  # This will be the input to the teacher model
        }

    def prepare_dataset(
        self,
        dataset: DatasetDict,
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle: bool = True,
    ) -> DatasetDict:
        """Prepare a dataset for training.

        Args:
            dataset: HuggingFace dataset with 'audio' column
            batch_size: Batch size for the dataloader
            num_workers: Number of workers for the dataloader
            shuffle: Whether to shuffle the dataset

        Returns:
            PyTorch DataLoader with preprocessed samples
        """

        # Define preprocessing function for the dataset
        def preprocess_function(examples):
            # Process each audio sample in the batch
            batch_size = len(examples["audio"])
            result = {
                "input_features": [],
                "clean_features": [],
            }

            for i in range(batch_size):
                processed = self.preprocess_audio(examples["audio"][i])
                result["input_features"].append(processed["input_features"])
                result["clean_features"].append(processed["clean_features"])

            # Stack tensors
            for key in result:
                result[key] = torch.cat(result[key], dim=0)

            return result

        # Apply preprocessing to the dataset
        any_split = next(iter(dataset.keys()))
        columns_to_remove = dataset[any_split].column_names
        processed_dataset: DatasetDict = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=columns_to_remove,
            num_proc=num_workers,
        )

        if shuffle:
            # For non-streaming datasets, we need to shuffle differently
            processed_dataset = DatasetDict({
                split: dataset.shuffle(seed=42) for split, dataset in processed_dataset.items()
            })

        return processed_dataset

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        augmenter: Optional[AudioAugmenter] = None,
        max_audio_length: float = 30.0,
    ) -> "DataPreprocessor":
        """Create a DataPreprocessor from a pretrained model.

        Args:
            model_name_or_path: Name or path of the pretrained model
            augmenter: Audio augmenter for applying augmentations
            max_audio_length: Maximum audio length in seconds

        Returns:
            DataPreprocessor instance
        """
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
        return cls(feature_extractor, augmenter, max_audio_length)
