"""Tests for the loss computation module."""

import unittest
import torch

from robust_whisper_trainer.loss_computer import LossComputer, DistillationLoss


class TestLossComputer(unittest.TestCase):
    """Test cases for the LossComputer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample hidden states
        batch_size = 2
        sequence_length = 10
        hidden_size = 16
        
        # Create teacher and student hidden states
        self.teacher_hidden = torch.randn(batch_size, sequence_length, hidden_size)
        self.student_hidden = torch.randn(batch_size, sequence_length, hidden_size)
        
        # Create multiple layers of hidden states
        self.teacher_hidden_states = [
            torch.randn(batch_size, sequence_length, hidden_size)
            for _ in range(3)
        ]
        self.student_hidden_states = [
            torch.randn(batch_size, sequence_length, hidden_size)
            for _ in range(3)
        ]
    
    def test_mse_loss(self):
        """Test that MSE loss is computed correctly."""
        # Create loss computer with only MSE loss
        loss_computer = LossComputer(layer_weights=None, cosine_lambda=0.0)
        
        # Compute loss
        loss = loss_computer._compute_layer_loss(
            self.teacher_hidden,
            self.student_hidden,
        )
        
        # Check that loss is a scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        
        # Check that loss is positive
        self.assertGreater(loss.item(), 0.0)
        
        # Check that loss is equal to MSE loss
        expected_loss = torch.nn.functional.mse_loss(
            self.student_hidden,
            self.teacher_hidden,
        )
        self.assertEqual(loss, expected_loss)
    
    def test_cosine_loss(self):
        """Test that cosine similarity loss is computed correctly."""
        # Create loss computer with both MSE and cosine loss
        cosine_lambda = 0.5
        loss_computer = LossComputer(layer_weights=None, cosine_lambda=cosine_lambda)
        
        # Compute loss
        loss = loss_computer._compute_layer_loss(
            self.teacher_hidden,
            self.student_hidden,
        )
        
        # Check that loss is a scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        
        # Check that loss is positive
        self.assertGreater(loss.item(), 0.0)
        
        # Check that loss is greater than MSE loss alone
        mse_loss = torch.nn.functional.mse_loss(
            self.student_hidden,
            self.teacher_hidden,
        )
        self.assertGreater(loss.item(), mse_loss.item())
    
    def test_layer_weights(self):
        """Test that layer weights are applied correctly."""
        # Create loss computer with layer weights
        layer_weights = [0.2, 0.3, 0.5]
        loss_computer = LossComputer(layer_weights=layer_weights, cosine_lambda=0.0)
        
        # Compute loss
        loss = loss_computer.compute_loss(
            self.teacher_hidden_states,
            self.student_hidden_states,
        )
        
        # Check that loss is a scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        
        # Check that loss is positive
        self.assertGreater(loss.item(), 0.0)
        
        # Check that loss is equal to weighted sum of layer losses
        expected_loss = 0.0
        for i, weight in enumerate(layer_weights):
            layer_loss = torch.nn.functional.mse_loss(
                self.student_hidden_states[i],
                self.teacher_hidden_states[i],
            )
            expected_loss += weight * layer_loss
        
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)
    
    def test_distillation_loss_module(self):
        """Test that DistillationLoss module works correctly."""
        # Create distillation loss module
        distillation_loss = DistillationLoss(layer_weights=None, cosine_lambda=0.0)
        
        # Compute loss
        loss = distillation_loss(
            self.teacher_hidden_states,
            self.student_hidden_states,
        )
        
        # Check that loss is a scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        
        # Check that loss is positive
        self.assertGreater(loss.item(), 0.0)
        
        # Check that loss is equal to MSE loss of last layer
        expected_loss = torch.nn.functional.mse_loss(
            self.student_hidden_states[-1],
            self.teacher_hidden_states[-1],
        )
        self.assertEqual(loss, expected_loss)


if __name__ == "__main__":
    unittest.main()
