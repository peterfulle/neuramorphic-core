import torch
import torch.nn as nn
from typing import Optional, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import NeuratekConfig
from layers.snn_layer import SNNReflexiveStream
from layers.neurossm_layer import NeuroSSMCognitiveStream
from layers.biofusion_layer import BioFusionLayer


class NeuromorphicModel(nn.Module):
    """Neuromorphic language model with dual-stream architecture"""
    
    def __init__(self, config: NeuratekConfig):
        super().__init__()
        self.config = config
        
        # embedding layers
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_sequence_length, config.hidden_size)
        
        # dual streams
        self.snn_stream = SNNReflexiveStream(config)
        
        self.neurossm_stream = NeuroSSMCognitiveStream(
            layers=config.neurossm_layers,
            hidden_size=config.neurossm_hidden_size,
            state_size=config.neurossm_state_size,
            expansion_factor=config.neurossm_expansion_factor
        )
        
        # bio-fusion
        self.bio_fusion = BioFusionLayer(
            hidden_size=config.fusion_hidden_size,
            num_heads=config.fusion_attention_heads,
            intermediate_size=config.fusion_intermediate_size,
            dropout=config.fusion_dropout
        )
        
        # output head
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        
        # initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def get_model_info(self) -> Dict:
        """Get model statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # component breakdown
        embedding_params = sum(p.numel() for p in self.embeddings.parameters()) + sum(p.numel() for p in self.position_embeddings.parameters())
        snn_params = sum(p.numel() for p in self.snn_stream.parameters())
        neurossm_params = sum(p.numel() for p in self.neurossm_stream.parameters())
        fusion_params = sum(p.numel() for p in self.bio_fusion.parameters())
        output_params = sum(p.numel() for p in self.output_projection.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_scale_billions': total_params / 1e9,
            'components': {
                'embeddings': {'count': embedding_params, 'percentage': embedding_params/total_params*100},
                'snn_stream': {'count': snn_params, 'percentage': snn_params/total_params*100},
                'neurossm_stream': {'count': neurossm_params, 'percentage': neurossm_params/total_params*100},
                'bio_fusion': {'count': fusion_params, 'percentage': fusion_params/total_params*100},
                'output_head': {'count': output_params, 'percentage': output_params/total_params*100}
            }
        }
        
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> Dict:
        """Forward pass through neuromorphic architecture"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # embeddings
        token_embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        
        # dual stream processing
        snn_output, snn_metrics = self.snn_stream(embeddings)
        neurossm_output, neurossm_states = self.neurossm_stream(embeddings)
        
        # bio-fusion
        fused_output = self.bio_fusion(snn_output, neurossm_output)
        
        # output projection
        fused_output = self.output_norm(fused_output)
        logits = self.output_projection(fused_output)
        
        return {
            'logits': logits,
            'snn_metrics': snn_metrics,
            'neurossm_states': neurossm_states,
            'embeddings': embeddings,
            'snn_output': snn_output,
            'neurossm_output': neurossm_output,
            'fused_output': fused_output
        }
