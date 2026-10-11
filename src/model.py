"""
DeepTemp: Stacked LSTM Neural Network Architecture (Phase 3)
Module: src/model.py

Defines the PyTorch Stacked LSTM network topology for modeling historical
climate sequences and capturing long-range seasonal dependencies.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class StackedLSTM(nn.Module):
    """
    Stacked Long Short-Term Memory (LSTM) Architecture for Climate Time-Series.
    
    Topology:
      1. Input Layer: (Batch, Seq_Len, Features)
      2. Stacked LSTM Layers: Multi-layer recurrent cells with inter-layer dropout.
      3. Regularization: Dropout layer on the latent representation to prevent
         overfitting on extreme seasonal patterns.
      4. Fully Connected Output Head: Linear projection to the predicted forecast step(s).
    """
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super(StackedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Multi-layer LSTM core
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Regularization before the final projection head
        self.fc_dropout = nn.Dropout(p=dropout)
        
        # Dense projection layer mapping latent states to forecast values
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward computation pass.
        
        Args:
            x: Input tensor with shape [batch_size, sequence_length, input_size]
            hidden: Optional initial hidden & cell state tuple (h_0, c_0)
            
        Returns:
            out: Forecast output tensor.
                 If output_size == 1, returns [batch_size]
                 If output_size > 1, returns [batch_size, output_size]
        """
        # lstm_out shape: [batch_size, seq_len, hidden_size * num_directions]
        lstm_out, _ = self.lstm(x, hidden)

        # Slice the representation at the final sequence step (t = W)
        last_step = lstm_out[:, -1, :]  # [batch_size, hidden_size * num_directions]

        # Apply dropout regularization and project to output space
        out = self.fc_dropout(last_step)
        out = self.fc(out)  # [batch_size, output_size]

        if self.output_size == 1:
            out = out.squeeze(-1)  # Flatten to [batch_size]

        return out

    def get_parameter_count(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test for StackedLSTM
    print("=== Testing StackedLSTM Network Topology ===")
    dummy_input = torch.randn(32, 30, 1)  # Batch 32, Window 30 days, 1 feature (tavg)
    model = StackedLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2)
    output = model(dummy_input)

    print(f"Model Architecture:\n{model}")
    print(f"Total Trainable Parameters: {model.get_parameter_count():,}")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    assert output.shape == (32,), f"Expected shape (32,), got {output.shape}"
    print("Smoke test passed successfully!")
