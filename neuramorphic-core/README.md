# Neuratek Neuromorphic Core

Advanced neuromorphic language model with biological realism and dual-stream architecture.

## Features

- **Biological Spiking Neural Networks**: LIF neurons with STDP plasticity and homeostasis
- **NeuroSSM Cognitive Stream**: State space models for high-level cognitive processing  
- **Bio-Fusion Architecture**: Intelligent integration of reflexive and cognitive streams
- **Professional Modular Design**: Clean, maintainable, and extensible codebase
- **5.5B+ Parameter Scale**: Production-ready large language model architecture

## Architecture

```
Input → Embeddings → Dual Streams → Bio-Fusion → Output
                    ↙          ↘
              SNN Stream    NeuroSSM Stream
           (Reflexive)     (Cognitive)
```

## Installation

```bash
pip install torch torchvision torchaudio
```

## Quick Start

```python
from neuramorphic_core import NeuratekConfig, NeuromorphicModel

# create configuration
config = NeuratekConfig()

# create model
model = NeuromorphicModel(config)

# forward pass
outputs = model(input_ids)
```

## Demo

Run the included demonstration:

```bash
python demo.py
```

## Configuration

Key configuration parameters:

- `vocab_size`: Vocabulary size (default: 75000)
- `hidden_size`: Hidden dimension (default: 4096)
- `snn_layers`: Number of SNN layers (default: 18)
- `neurossm_layers`: Number of NeuroSSM layers (default: 28)
- `target_firing_rate`: Target biological firing rate (default: 8.0 Hz)

## Module Structure

```
neuramorphic-core/
├── config/          # Model configuration
├── models/          # Main model architecture
├── layers/          # Neural network layers
│   ├── snn_layer.py        # Spiking neural networks
│   ├── neurossm_layer.py   # State space models
│   └── biofusion_layer.py  # Stream fusion
├── utils/           # Validation and utilities
├── training/        # Training utilities
└── demo.py          # Demonstration script
```

## Biological Features

- **LIF Neurons**: Leaky Integrate-and-Fire dynamics
- **STDP Plasticity**: Spike-timing dependent plasticity
- **Homeostasis**: Self-regulating firing rates
- **Dale's Principle**: Excitatory/inhibitory neuron types
- **Biological Noise**: Pink noise, rhythms, variability
- **Adaptive Thresholds**: Dynamic firing thresholds

## Performance

- **5.5B Parameters**: Large-scale architecture
- **Biological Validation**: 1-15Hz firing rates
- **High Variability**: Maximum neuronal diversity
- **GPU Optimized**: CUDA acceleration support

## Author

Peter Fulle (@peterfulle)  
Neuratek Company

## License

Proprietary - Neuratek Company
