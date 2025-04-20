"""Robust Whisper Trainer package for fine-tuning Huggingface models."""

__version__ = "0.1.0"

# Import main classes and functions
from .audio_augmenter import AudioAugmenter
from .data_preprocessor import DataPreprocessor
from .model_wrapper import WhisperEncoderTeacherStudentWrapper
from .loss_computer import LossComputer, DistillationLoss
from .datasets import load_datasets
from .training_harness import (
    RobustWhisperTrainer,
    RobustWhisperTrainingArguments,
    main,
)
