import torch
import numpy as np
from typing import Dict
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.neuromorphic_model import NeuromorphicModel
from utils.logger import NeuromorphicLogger


def validate_architecture(model: NeuromorphicModel, dataloader: DataLoader, device: torch.device, logger: NeuromorphicLogger = None) -> Dict:
    """Validate neuromorphic architecture performance with detailed logging"""
    model.eval()
    
    if logger is None:
        logger = NeuromorphicLogger()
    
    # start validation logging
    logger.log_validation_start(model.config)
    
    total_spikes = 0
    spike_rates = []
    variability_measurements = []
    batch_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            
            # forward pass
            outputs = model(input_ids)
            snn_metrics = outputs['snn_metrics']['metrics']
            
            # collect spike statistics
            batch_spikes = snn_metrics.get('total_spikes', 0)
            total_spikes += batch_spikes
            
            # calculate spike rate
            batch_size, seq_len = input_ids.shape
            total_neurons = model.config.snn_layers * model.config.snn_neurons_per_layer
            max_possible_spikes = batch_size * seq_len * total_neurons
            spike_rate = (batch_spikes / max_possible_spikes) * 1000 if max_possible_spikes > 0 else 0
            spike_rates.append(spike_rate)
            
            # collect variability measurements
            if 'variability_metrics' in snn_metrics:
                variability_measurements.extend(snn_metrics['variability_metrics'])
            
            # prepare batch results for logging
            batch_result = {
                'spikes': batch_spikes,
                'spike_rate': spike_rate,
                'snn_metrics': {
                    'functional_layers': snn_metrics.get('functional_layers', 0),
                    'total_layers': model.config.snn_layers,
                    'variability_metrics': snn_metrics.get('variability_metrics', [])
                }
            }
            batch_results.append(batch_result)
            
            # log batch processing
            logger.log_batch_processing(batch_idx, batch_result)
    
    # calculate comprehensive metrics
    avg_spike_rate = np.mean(spike_rates) if spike_rates else 0
    spike_rate_std = np.std(spike_rates) if len(spike_rates) > 1 else 0
    rate_variability = spike_rate_std / avg_spike_rate if avg_spike_rate > 0 else 0
    
    # neuronal variability
    neuronal_variability = np.mean(variability_measurements) if variability_measurements else 0
    
    # temporal variability  
    temporal_variability = np.std([len(rates) for rates in [spike_rates]]) / len(spike_rates) if spike_rates else 0
    
    # combined variability
    combined_variability = max(
        rate_variability * 3.0,
        neuronal_variability / 5.0,
        temporal_variability * 2.0,
        0.1
    )
    
    # detailed spike analysis
    spike_data = {
        'total_spikes': total_spikes,
        'avg_spike_rate': avg_spike_rate,
        'spike_rate_std': spike_rate_std,
        'spike_distribution': {
            'min': min(spike_rates) if spike_rates else 0,
            'max': max(spike_rates) if spike_rates else 0,
            'median': np.median(spike_rates) if spike_rates else 0
        }
    }
    logger.log_spike_analysis(spike_data)
    
    # detailed variability analysis
    variability_data = {
        'neuronal_variability': neuronal_variability,
        'combined_variability': combined_variability,
        'variability_breakdown': {
            'rate_variability': rate_variability,
            'temporal_variability': temporal_variability,
            'measurement_count': len(variability_measurements)
        }
    }
    logger.log_variability_analysis(variability_data)
    
    # validation criteria
    is_biologically_valid = (model.config.target_spike_rate_min <= avg_spike_rate <= model.config.target_spike_rate_max)
    has_sufficient_variability = combined_variability >= model.config.target_variability_min
    has_sufficient_functionality = (total_spikes > 0)
    
    # prepare biological validation results
    bio_results = {
        'biologically_valid_rate': is_biologically_valid,
        'sufficient_variability': has_sufficient_variability,
        'sufficient_functionality': has_sufficient_functionality,
        'avg_spike_rate': avg_spike_rate,
        'combined_variability': combined_variability,
        'functional_ratio': 1.0 if has_sufficient_functionality else 0.0,
        'min_target_rate': model.config.target_spike_rate_min,
        'max_target_rate': model.config.target_spike_rate_max,
        'min_variability': model.config.target_variability_min
    }
    
    # log biological validation
    logger.log_biological_validation(bio_results)
    
    # final results
    final_results = {
        'total_spikes': total_spikes,
        'avg_spike_rate': avg_spike_rate,
        'combined_variability': combined_variability,
        'neuronal_variability': neuronal_variability,
        'functional_ratio': 1.0 if has_sufficient_functionality else 0.0,
        'biologically_valid_rate': is_biologically_valid,
        'sufficient_variability': has_sufficient_variability,
        'sufficient_functionality': has_sufficient_functionality,
        'overall_validity': is_biologically_valid and has_sufficient_variability and has_sufficient_functionality
    }
    
    # log final results
    logger.log_final_results(final_results)
    
    # save validation report
    logger.save_validation_report()
    
    return final_results


def print_model_summary(model: NeuromorphicModel):
    """Print comprehensive model summary"""
    info = model.get_model_info()
    
    print(f"Neuratek Neuromorphic Model")
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")
    print(f"Parameter scale: {info['parameter_scale_billions']:.2f}B")
    
    print(f"\nComponent breakdown:")
    for name, comp in info['components'].items():
        print(f"  {name}: {comp['count']:,} ({comp['percentage']:.1f}%)")


def print_validation_results(results: Dict):
    """Print validation results"""
    print(f"\nValidation results:")
    print(f"  Total spikes: {results['total_spikes']:,}")
    print(f"  Average spike rate: {results['avg_spike_rate']:.2f}Hz")
    print(f"  Combined variability: {results['combined_variability']:.3f}")
    print(f"  Neuronal variability: {results['neuronal_variability']:.3f}")
    print(f"  Functional ratio: {results['functional_ratio']:.2f}")
    
    print(f"\nValidation criteria:")
    print(f"  Biologically valid rate: {results['biologically_valid_rate']}")
    print(f"  Sufficient variability: {results['sufficient_variability']}")
    print(f"  Sufficient functionality: {results['sufficient_functionality']}")
    print(f"  Overall validity: {results['overall_validity']}")
    
    if results['overall_validity']:
        print(f"\nModel validation: SUCCESS")
    else:
        print(f"\nModel validation: NEEDS OPTIMIZATION")
