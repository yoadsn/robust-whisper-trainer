"""Model wrapper module for teacher-student distillation."""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperConfig
from robust_whisper_trainer.loss_computer import DistillationLoss


class WhisperEncoderTeacherStudentWrapper(nn.Module):
    """Wrapper for teacher-student distillation of Whisper encoder."""

    def __init__(
        self,
        base_model_name_or_path: str,
        teacher_model: WhisperModel,
        student_model: WhisperModel,
        loss_layer_weights: Optional[List[float]] = None,
        loss_cosine_lambda: float = 0.0,
    ):
        """Initialize the teacher-student wrapper.

        Args:
            teacher_model: Teacher Whisper model (processes clean audio)
            student_model: Student Whisper model (processes augmented audio)
            freeze_decoder: Whether to freeze the decoder in the student model
            freeze_teacher: Whether to freeze the teacher model
            loss_layer_weights: Weights for each encoder layer (None means use only last layer)
            loss_cosine_lambda: Weight for cosine similarity loss (0 means use only MSE)
        """
        super().__init__()
        self.base_model_name_or_path = base_model_name_or_path
        self.teacher_model = teacher_model.encoder
        self.student_model = student_model.encoder
        self.accepts_loss_kwargs = True
        self.loss_computer = DistillationLoss(
            layer_weights=loss_layer_weights,
            cosine_lambda=loss_cosine_lambda,
        )

        # Freeze the teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def forward(
        self,
        clean_features: torch.Tensor,
        input_features: torch.Tensor,
        num_items_in_batch: Optional[int] = None,
        return_loss: bool = True,
        return_outputs: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through the teacher and student models.

        Args:
            clean_features: Clean audio features for the teacher model
            input_features: Augmented audio features for the student model
            num_items_in_batch: Number of items in batch
            return_loss: Whether to return the loss
            return_outputs: Whether to return the teacher and student outputs

        Returns:
            Dictionary with teacher and student outputs and the loss
        """
        # Forward pass through teacher model with clean audio
        teacher_outputs = self.teacher_model(
            input_features=clean_features,
            output_hidden_states=True,
            return_dict=True,
        )

        # Forward pass through student model with augmented audio
        student_outputs = self.student_model(
            input_features=input_features,
            output_hidden_states=True,
            return_dict=True,
        )

        output = {}

        # Compute loss
        if return_loss:
            loss = self.loss_computer(
                teacher_outputs["hidden_states"],
                student_outputs["hidden_states"],
            )
            # loss is a mean reduction of MSE and possible cosine similarity over the batch and sequence length
            # to normalize loss for all grad accumulation steps and batch size - we use the num_items_in_batch
            # value which accounts for those
            if num_items_in_batch is not None:
                loss_sum_over_this_batch = loss * clean_features.shape[0]
                loss = (
                    loss_sum_over_this_batch / num_items_in_batch
                )  # the mean across all samples in step

            output["loss"] = loss

        if return_outputs:
            output["teacher_outputs"] = teacher_outputs
            output["student_outputs"] = student_outputs

        return output

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        loss_layer_weights: Optional[List[float]] = None,
        loss_cosine_lambda: float = 0.0,
    ) -> "WhisperEncoderTeacherStudentWrapper":
        """Create a WhisperEncoderTeacherStudentWrapper from a pretrained model.

        Args:
            model_name_or_path: Name or path of the pretrained model
            freeze_decoder: Whether to freeze the decoder in the student model
            freeze_teacher: Whether to freeze the teacher model

        Returns:
            WhisperEncoderTeacherStudentWrapper instance
        """
        # Load the teacher model
        teacher_model = WhisperModel.from_pretrained(model_name_or_path)

        # Load the student model (initialized with the same weights)
        student_model = WhisperModel.from_pretrained(model_name_or_path)

        return cls(
            base_model_name_or_path=model_name_or_path,
            teacher_model=teacher_model,
            student_model=student_model,
            loss_layer_weights=loss_layer_weights,
            loss_cosine_lambda=loss_cosine_lambda,
        )

    def save_pretrained(self, output_dir: str) -> None:
        """Save only the student encoder to a directory.

        Args:
            output_dir: Directory to save the encoder to
        """
        # Load the base model
        base_model = WhisperModel.from_pretrained(self.base_model_name_or_path)

        # Update the model's encoder weights
        updated_model_state_dict = base_model.state_dict()
        for key, value in self.student_model.state_dict().items():
            encoder_key = "encoder." + key
            if encoder_key in updated_model_state_dict:
                updated_model_state_dict[encoder_key] = value

        # Load the updated state dict
        base_model.load_state_dict(updated_model_state_dict)

        base_model.save_pretrained(output_dir)
        base_model.config.save_pretrained(output_dir)
