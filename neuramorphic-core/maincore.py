#!/usr/bin/env python3
"""
Neuratek Neuromorphic Core Demo
Professional neuromorphic language model with biological realism
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gc
import time

from config.model_config import NeuratekConfig
from models.neuromorphic_model import NeuromorphicModel
from utils.validation import validate_architecture, print_model_summary, print_validation_results
from utils.logger import NeuromorphicLogger
from training.dataset import SyntheticNeuromorphicDataset


def main():
    """Main execution function"""
    # create logger
    logger = NeuromorphicLogger()
    
    logger.console_logger.info("Neuratek Neuromorphic Core")
    logger.console_logger.info("Advanced neuromorphic language model with biological realism")
    
    try:
        # device setup
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.console_logger.info(f"Device: {device}")
        
        # log system information
        logger.log_system_info()
        
        # configuration
        config = NeuratekConfig()
        config.device = str(device)
        
        if not config.validate_config():
            raise ValueError("Invalid configuration parameters")
        
        logger.console_logger.info(f"\nCreating neuromorphic model...")
        
        # model creation
        model = NeuromorphicModel(config).to(device)
        
        # log model creation
        model_info = model.get_model_info()
        logger.log_model_creation(model_info)
        
        # test forward pass
        logger.console_logger.info(f"\nTesting forward pass...")
        test_input = torch.randint(0, config.vocab_size, (1, config.max_sequence_length)).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(test_input)
        forward_time = time.time() - start_time
        
        logger.console_logger.info(f"Forward time: {forward_time:.2f}s")
        logger.console_logger.info(f"Output shape: {outputs['logits'].shape}")
        
        # snn metrics
        snn_metrics = outputs['snn_metrics']['metrics']
        total_spikes = snn_metrics.get('total_spikes', 0)
        functional_layers = snn_metrics.get('functional_layers', 0)
        
        # calculate spike rate
        batch_size, seq_len = test_input.shape
        total_neurons = config.snn_layers * config.snn_neurons_per_layer
        max_possible_spikes = batch_size * seq_len * total_neurons
        spike_rate = (total_spikes / max_possible_spikes) * 1000 if max_possible_spikes > 0 else 0
        
        # neuronal variability
        variability_metrics = snn_metrics.get('variability_metrics', [])
        neuronal_variability = sum(variability_metrics) / len(variability_metrics) if variability_metrics else 0
        
        logger.console_logger.info(f"\nInitial SNN metrics:")
        logger.console_logger.info(f"  Total spikes: {total_spikes:,}")
        logger.console_logger.info(f"  Functional layers: {functional_layers}/{config.snn_layers}")
        logger.console_logger.info(f"  Neuronal variability: {neuronal_variability:.3f}")
        logger.console_logger.info(f"  Average spike rate: {spike_rate:.2f}Hz")
        
        # validation with detailed logging
        logger.console_logger.info(f"\nStarting comprehensive validation...")
        
        # create synthetic dataset
        dataset = SyntheticNeuromorphicDataset(
            vocab_size=config.vocab_size,
            seq_length=config.max_sequence_length,
            num_samples=5
        )
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        
        # validate architecture with detailed logging
        validation_results = validate_architecture(model, dataloader, device, logger)
        
        # log final system state
        logger.log_system_info()
        
        # get log file locations
        log_files = logger.get_log_files()
        logger.console_logger.info(f"\nGenerated log files:")
        logger.console_logger.info(f"  Detailed log: {log_files['log_file']}")
        logger.console_logger.info(f"  JSON report: {log_files['report_file']}")
        logger.console_logger.info(f"  Session ID: {log_files['session_id']}")
        
        return log_files
            
    except Exception as e:
        if 'logger' in locals():
            logger.console_logger.error(f"Error: {e}")
            logger.file_logger.error(f"Exception occurred: {e}", exc_info=True)
        else:
            print(f"Error: {e}")
        raise
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if 'logger' in locals():
            logger.console_logger.info(f"Cleanup completed")


if __name__ == "__main__":
    main()
