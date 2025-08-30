import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import NeuratekConfig


class NeuroSSMLayer(nn.Module):
    """NeuroSSM layer for cognitive processing"""
    
    def __init__(self, hidden_size: int, state_size: int, expansion_factor: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        
        # ssm parameters
        self.A = nn.Parameter(torch.randn(self.state_size, self.state_size) * 0.1)
        self.B = nn.Parameter(torch.randn(self.state_size, self.hidden_size) * 0.1)
        self.C = nn.Parameter(torch.randn(self.hidden_size, self.state_size) * 0.1)
        self.D = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.1)
        
        # neural enhancements
        self.norm = nn.LayerNorm(self.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * expansion_factor),
            nn.GELU(),
            nn.Linear(self.hidden_size * expansion_factor, self.hidden_size),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = x.shape
        
        if state is None:
            state = torch.zeros(batch_size, self.state_size, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        for t in range(seq_len):
            state = torch.tanh(state @ self.A.T + x[:, t] @ self.B.T)
            output = state @ self.C.T + x[:, t] @ self.D.T
            outputs.append(output)
        
        output_seq = torch.stack(outputs, dim=1)
        output_seq = self.norm(output_seq + x)
        output_seq = output_seq + self.ffn(output_seq)
        
        return output_seq, state


class NeuroSSMCognitiveStream(nn.Module):
    """NeuroSSM cognitive stream for high-level processing"""
    
    def __init__(self, layers: int, hidden_size: int, state_size: int, expansion_factor: int):
        super().__init__()
        self.layers = nn.ModuleList([
            NeuroSSMLayer(hidden_size, state_size, expansion_factor) 
            for _ in range(layers)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        states = []
        
        for layer in self.layers:
            x, state = layer(x)
            states.append(state)
            
        return x, states
