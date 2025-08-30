"""
Model configuration for Neuratek Neuromorphic Architecture
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class NeuratekConfig:
    """Configuration for neuromorphic language model"""
    
    # model architecture
    vocab_size: int = 75000
    hidden_size: int = 4096
    max_sequence_length: int = 1024
    
    # snn parameters
    snn_layers: int = 18
    snn_hidden_size: int = 2048
    snn_neurons_per_layer: int = 2048
    
    # biological neuron parameters
    tau_membrane: float = 30e-3
    tau_synapse: float = 5e-3
    tau_adaptation: float = 120e-3
    tau_refractoriness: float = 6e-3
    
    # voltage dynamics
    v_threshold_base: float = -44.0
    v_threshold_range: float = 6.0
    v_reset: float = -75.0
    v_rest: float = -70.0
    
    # current injection
    current_injection_scale: float = 0.0003
    current_baseline: float = 0.0
    current_noise_factor: float = 0.8
    
    # noise parameters
    membrane_noise_std: float = 1.5
    synaptic_noise_std: float = 0.8
    threshold_noise_std: float = 3.0
    
    # biological rhythms
    theta_frequency: float = 7.0
    gamma_frequency: float = 40.0
    background_noise_factor: float = 0.15
    
    # variability control
    master_variability_scale: float = 1.0
    
    # plasticity parameters
    stdp_learning_rate: float = 0.002
    stdp_tau_plus: float = 20e-3
    stdp_tau_minus: float = 20e-3
    
    # homeostasis
    homeostasis_tau: float = 10.0
    target_firing_rate: float = 8.0
    homeostasis_strength: float = 0.2
    
    # performance optimization
    enable_fast_mode: bool = False
    temporal_downsampling: int = 2
    
    # advanced variability
    spatiotemporal_dropout_p: float = 0.05
    threshold_diversification_scale: float = 0.1
    adaptive_noise_sensitivity: float = 0.5
    
    # dale's principle
    excitatory_ratio: float = 0.8
    inhibitory_ratio: float = 0.2
    
    # neurossm parameters
    neurossm_layers: int = 28
    neurossm_hidden_size: int = 4096
    neurossm_state_size: int = 320
    neurossm_expansion_factor: int = 4
    
    # bio-fusion parameters
    fusion_hidden_size: int = 4096
    fusion_attention_heads: int = 64
    fusion_intermediate_size: int = 16384
    fusion_dropout: float = 0.1
    
    # training configuration
    batch_size: int = 1
    learning_rate: float = 8e-5
    gradient_clip_val: float = 1.0
    accumulation_steps: int = 16
    
    # validation parameters
    target_spike_rate_min: float = 1.0
    target_spike_rate_max: float = 15.0
    target_variability_min: float = 0.3
    
    # device configuration
    device: str = "auto"  # Auto-detect best available device
    dtype: str = "float32"
    mixed_precision: bool = True
    
    def __post_init__(self):
        """Post initialization to auto-detect device if set to 'auto'"""
        if self.device == "auto":
            self.device = self._get_optimal_device()
    
    def _get_optimal_device(self) -> str:
        """Automatically detect and select the optimal device"""
        if not torch.cuda.is_available():
            return "cpu"
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            return "cpu"
        
        # Find GPU with most available memory
        max_memory = 0
        best_gpu = 0
        
        for i in range(num_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i)
            available = gpu_props.total_memory - allocated
            
            if available > max_memory:
                max_memory = available
                best_gpu = i
        
        return f"cuda:{best_gpu}"
    
    def validate_config(self) -> bool:
        """Validate configuration parameters"""
        if self.hidden_size <= 0:
            return False
        if self.snn_layers <= 0 or self.neurossm_layers <= 0:
            return False
        if not (0 < self.excitatory_ratio < 1):
            return False
        if self.target_firing_rate <= 0:
            return False
        return True
