"""Loss computation module for teacher-student distillation."""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossComputer:
    """Computes loss between teacher and student encoder hidden states."""
    
    def __init__(
        self,
        layer_weights: Optional[List[float]] = None,
        cosine_lambda: float = 0.0,
    ):
        """Initialize the loss computer.
        
        Args:
            layer_weights: Weights for each encoder layer (None means use only last layer)
            cosine_lambda: Weight for cosine similarity loss (0 means use only MSE)
        """
        self.layer_weights = layer_weights
        self.cosine_lambda = cosine_lambda
        
    def compute_loss(
        self,
        teacher_hidden_states: List[torch.Tensor],
        student_hidden_states: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss between teacher and student hidden states.
        
        Args:
            teacher_hidden_states: List of teacher encoder hidden states
            student_hidden_states: List of student encoder hidden states
            
        Returns:
            Loss tensor
        """
        # If layer_weights is None, use only the last layer
        if self.layer_weights is None:
            # Use only the last layer
            return self._compute_layer_loss(
                teacher_hidden_states[-1],
                student_hidden_states[-1],
            )
        
        # Ensure layer_weights has the correct length
        if len(self.layer_weights) != len(teacher_hidden_states):
            raise ValueError(
                f"layer_weights has length {len(self.layer_weights)}, "
                f"but there are {len(teacher_hidden_states)} hidden states"
            )
        
        # Compute weighted sum of losses for each layer
        total_loss = 0.0
        for i, weight in enumerate(self.layer_weights):
            if weight > 0:
                layer_loss = self._compute_layer_loss(
                    teacher_hidden_states[i],
                    student_hidden_states[i],
                )
                total_loss += weight * layer_loss
        
        return total_loss
    
    def _compute_layer_loss(
        self,
        teacher_hidden: torch.Tensor,
        student_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss between teacher and student hidden states for a single layer.
        
        Args:
            teacher_hidden: Teacher encoder hidden states for a layer
            student_hidden: Student encoder hidden states for a layer
            
        Returns:
            Loss tensor
        """
        # Compute MSE loss
        mse_loss = F.mse_loss(student_hidden, teacher_hidden, reduction='mean')
        
        # If cosine_lambda is 0, return only MSE loss
        if self.cosine_lambda == 0:
            return mse_loss
        
        # Compute cosine similarity loss
        # Reshape tensors to (batch_size * sequence_length, hidden_size)
        teacher_flat = teacher_hidden.view(-1, teacher_hidden.size(-1))
        student_flat = student_hidden.view(-1, student_hidden.size(-1))
        
        # Compute cosine similarity (1 - cosine_similarity to make it a loss)
        cosine_loss = 1 - F.cosine_similarity(student_flat, teacher_flat, dim=1).mean()
        
        # Combine losses
        return mse_loss + self.cosine_lambda * cosine_loss


class DistillationLoss(nn.Module):
    """Loss module for teacher-student distillation."""
    
    def __init__(
        self,
        layer_weights: Optional[List[float]] = None,
        cosine_lambda: float = 0.0,
    ):
        """Initialize the distillation loss.
        
        Args:
            layer_weights: Weights for each encoder layer (None means use only last layer)
            cosine_lambda: Weight for cosine similarity loss (0 means use only MSE)
        """
        super().__init__()
        self.loss_computer = LossComputer(layer_weights, cosine_lambda)
    
    def forward(
        self,
        teacher_hidden_states: List[torch.Tensor],
        student_hidden_states: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss between teacher and student hidden states.
        
        Args:
            teacher_hidden_states: List of teacher encoder hidden states
            student_hidden_states: List of student encoder hidden states
            
        Returns:
            Loss tensor
        """
        return self.loss_computer.compute_loss(
            teacher_hidden_states,
            student_hidden_states,
        )
