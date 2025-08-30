"""
Neuratek Neuromorphic Core
Advanced neuromorphic language model with biological realism
"""

from .config.model_config import NeuratekConfig
from .models.neuromorphic_model import NeuromorphicModel
from .layers.snn_layer import BiologicalLIFNeuron, SNNReflexiveStream
from .layers.neurossm_layer import NeuroSSMLayer, NeuroSSMCognitiveStream
from .layers.biofusion_layer import BioFusionLayer
from .utils.validation import validate_architecture
from .training.dataset import SyntheticNeuromorphicDataset

__version__ = "2.0.0"
__author__ = "Peter Fulle (@peterfulle)"
__company__ = "Neuratek Company"

__all__ = [
    "NeuratekConfig",
    "NeuromorphicModel", 
    "BiologicalLIFNeuron",
    "SNNReflexiveStream",
    "NeuroSSMLayer",
    "NeuroSSMCognitiveStream",
    "BioFusionLayer",
    "validate_architecture",
    "SyntheticNeuromorphicDataset"
]
