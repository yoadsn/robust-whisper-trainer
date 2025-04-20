"""Dataset loading and processing for robust Whisper encoder training."""

import re
from typing import Optional, Tuple
from datasets import load_dataset, DatasetDict, Audio

from .data_preprocessor import DataPreprocessor


def load_datasets(
    train_dataset_spec: str,
    eval_dataset_spec: str,
    data_preprocessor: DataPreprocessor,
    batch_size: int = 1,
    num_workers: int = 1,
    shuffle: bool = False,
    audio_column_name: str = "audio",
) -> DatasetDict:
    """Load and process datasets for training and evaluation.

    Args:
        train_dataset_spec: The specification for the training dataset in the format:
                           <dataset_name>:<split>[:<config_name>]
        eval_dataset_spec: The specification for the evaluation dataset in the format:
                          <dataset_name>:<split>[:<config_name>]
        data_preprocessor: The data preprocessor to use for preparing the datasets
        batch_size: Batch size for data preprocessing
        num_workers: Number of workers for data preprocessing
        shuffle: Whether to shuffle the dataset during preprocessing
        audio_column_name: The name of the dataset column containing the audio data

    Returns:
        A DatasetDict containing processed 'train' and 'eval' datasets
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
        ).cast_column(audio_column_name, Audio(sampling_rate=16000, mono=True))

        return dataset

    train_dataset = load_dataset_spec(train_dataset_spec)
    eval_dataset = load_dataset_spec(eval_dataset_spec)

    dataset_split_dict = DatasetDict(
        {"train": train_dataset, "eval": eval_dataset}
    )

    dataset_split_dict = data_preprocessor.prepare_dataset(
        dataset_split_dict,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )

    return dataset_split_dict
