import torch
import torch.nn as nn
from typing import Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import NeuratekConfig


class BioFusionLayer(nn.Module):
    """Bio-fusion layer for intelligent stream integration"""
    
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # multi-head attention for fusion
        self.snn_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.neurossm_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # fusion networks with snn enhancement
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())
        
        # snn stream amplifier
        self.snn_amplifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        
        # neurossm stream dampener
        self.neurossm_dampener = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Dropout(dropout * 2)
        )
        
        self.fusion_transform = nn.Sequential(
            nn.Linear(hidden_size * 2, intermediate_size), nn.GELU(),
            nn.Linear(intermediate_size, hidden_size), nn.Dropout(dropout))
        
        # variability injection
        self.variability_injector = nn.Parameter(torch.ones(1) * 0.2)
        
        # normalization layers
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
    def forward(self, snn_output: torch.Tensor, neurossm_output: torch.Tensor) -> torch.Tensor:
        # rebalance streams: amplify snn, dampen neurossm
        snn_amplified = self.snn_amplifier(snn_output) + snn_output
        neurossm_dampened = self.neurossm_dampener(neurossm_output)
        
        # self-attention on each stream
        snn_attended, _ = self.snn_attention(snn_amplified, snn_amplified, snn_amplified)
        neurossm_attended, _ = self.neurossm_attention(neurossm_dampened, neurossm_dampened, neurossm_dampened)
        
        snn_attended = self.norm1(snn_attended + snn_amplified)
        neurossm_attended = self.norm2(neurossm_attended + neurossm_dampened)
        
        # cross-attention with snn bias
        cross_attended, _ = self.cross_attention(snn_attended, neurossm_attended, neurossm_attended)
        cross_attended = self.norm3(cross_attended + snn_attended)
        
        # fusion with enhanced snn weighting
        combined = torch.cat([cross_attended, neurossm_attended], dim=-1)
        
        # inject additional variability
        variability_noise = torch.randn_like(cross_attended) * torch.clamp(self.variability_injector, 0.1, 0.5)
        cross_attended = cross_attended + variability_noise
        
        gate = self.fusion_gate(combined)
        # enhanced snn influence: 70% snn, 30% neurossm
        snn_weight = gate * 0.7 + 0.3
        neurossm_weight = (1 - gate) * 0.3 + 0.1
        
        fused = snn_weight * cross_attended + neurossm_weight * neurossm_attended
        fused_transformed = self.fusion_transform(combined)
        fused = fused + fused_transformed
        
        return fused
