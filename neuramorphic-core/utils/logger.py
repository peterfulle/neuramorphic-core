"""
Advanced logging system for neuromorphic model validation
"""

import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, Any
import torch


class NeuromorphicLogger:
    """Professional logging system for neuromorphic validation"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # create timestamp for this session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # setup console logger
        self.console_logger = self._setup_console_logger()
        
        # setup file logger
        self.file_logger = self._setup_file_logger()
        
        # validation log storage
        self.validation_log = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'model_info': {},
            'validation_steps': [],
            'metrics': {},
            'final_results': {}
        }
        
    def _setup_console_logger(self):
        """Setup console logger with detailed formatting"""
        logger = logging.getLogger(f'neuromorphic_console_{self.session_id}')
        logger.setLevel(logging.INFO)
        
        # clear existing handlers
        logger.handlers.clear()
        
        # console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # detailed formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # prevent propagation to avoid duplicate logs
        logger.propagate = False
        
        return logger
        
    def _setup_file_logger(self):
        """Setup file logger for persistent logging"""
        logger = logging.getLogger(f'neuromorphic_file_{self.session_id}')
        logger.setLevel(logging.DEBUG)
        
        # clear existing handlers
        logger.handlers.clear()
        
        # file handler
        log_file = os.path.join(self.log_dir, f'validation_{self.session_id}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # detailed formatter for file
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # prevent propagation
        logger.propagate = False
        
        return logger
        
    def log_model_creation(self, model_info: Dict):
        """Log model creation details"""
        self.console_logger.info("=" * 60)
        self.console_logger.info("NEUROMORPHIC MODEL CREATION")
        self.console_logger.info("=" * 60)
        
        self.console_logger.info(f"Total parameters: {model_info['total_parameters']:,}")
        self.console_logger.info(f"Parameter scale: {model_info['parameter_scale_billions']:.2f}B")
        
        self.console_logger.info("\nComponent breakdown:")
        for name, comp in model_info['components'].items():
            self.console_logger.info(f"  {name}: {comp['count']:,} ({comp['percentage']:.1f}%)")
            
        # save to validation log
        self.validation_log['model_info'] = model_info
        
        # detailed file logging
        self.file_logger.info(f"Model created with configuration: {model_info}")
        
    def log_validation_start(self, config):
        """Log validation process start"""
        self.console_logger.info("\n" + "=" * 60)
        self.console_logger.info("NEUROMORPHIC VALIDATION PROCESS")
        self.console_logger.info("=" * 60)
        
        self.console_logger.info("Validation criteria:")
        self.console_logger.info(f"  Target spike rate: {config.target_spike_rate_min}-{config.target_spike_rate_max} Hz")
        self.console_logger.info(f"  Minimum variability: {config.target_variability_min}")
        self.console_logger.info(f"  SNN layers: {config.snn_layers}")
        self.console_logger.info(f"  Neurons per layer: {config.snn_neurons_per_layer}")
        
        self.file_logger.info(f"Starting validation with config: {vars(config)}")
        
    def log_batch_processing(self, batch_idx: int, batch_results: Dict):
        """Log individual batch processing"""
        spike_count = batch_results.get('spikes', 0)
        spike_rate = batch_results.get('spike_rate', 0)
        
        self.console_logger.info(f"Batch {batch_idx + 1}: {spike_count:,} spikes, {spike_rate:.2f}Hz")
        
        # detailed batch analysis
        if 'snn_metrics' in batch_results:
            metrics = batch_results['snn_metrics']
            functional_layers = metrics.get('functional_layers', 0)
            total_layers = metrics.get('total_layers', 18)
            
            self.console_logger.info(f"  Functional layers: {functional_layers}/{total_layers}")
            
            if 'variability_metrics' in metrics:
                var_metrics = metrics['variability_metrics']
                avg_variability = sum(var_metrics) / len(var_metrics) if var_metrics else 0
                self.console_logger.info(f"  Neuronal variability: {avg_variability:.3f}")
        
        # save batch data
        self.validation_log['validation_steps'].append({
            'batch_idx': batch_idx,
            'timestamp': datetime.now().isoformat(),
            'results': batch_results
        })
        
        self.file_logger.debug(f"Batch {batch_idx} detailed results: {batch_results}")
        
    def log_spike_analysis(self, spike_data: Dict):
        """Log detailed spike analysis"""
        self.console_logger.info("\nSpike Analysis:")
        self.console_logger.info(f"  Total spikes across all batches: {spike_data['total_spikes']:,}")
        self.console_logger.info(f"  Average spike rate: {spike_data['avg_spike_rate']:.2f}Hz")
        self.console_logger.info(f"  Spike rate std: {spike_data.get('spike_rate_std', 0):.3f}")
        
        if 'spike_distribution' in spike_data:
            self.console_logger.info(f"  Spike distribution analysis:")
            dist = spike_data['spike_distribution']
            self.console_logger.info(f"    Min: {dist.get('min', 0):.2f}Hz")
            self.console_logger.info(f"    Max: {dist.get('max', 0):.2f}Hz")
            self.console_logger.info(f"    Median: {dist.get('median', 0):.2f}Hz")
            
        self.file_logger.info(f"Detailed spike analysis: {spike_data}")
        
    def log_variability_analysis(self, variability_data: Dict):
        """Log detailed variability analysis"""
        self.console_logger.info("\nVariability Analysis:")
        self.console_logger.info(f"  Neuronal variability: {variability_data['neuronal_variability']:.3f}")
        self.console_logger.info(f"  Combined variability: {variability_data['combined_variability']:.3f}")
        
        if 'variability_breakdown' in variability_data:
            breakdown = variability_data['variability_breakdown']
            self.console_logger.info(f"  Variability components:")
            for component, value in breakdown.items():
                self.console_logger.info(f"    {component}: {value:.3f}")
                
        self.file_logger.info(f"Detailed variability analysis: {variability_data}")
        
    def log_biological_validation(self, bio_results: Dict):
        """Log biological validation criteria"""
        self.console_logger.info("\nBiological Validation:")
        
        # firing rate validation
        rate_valid = bio_results['biologically_valid_rate']
        avg_rate = bio_results['avg_spike_rate']
        min_rate = bio_results.get('min_target_rate', 1.0)
        max_rate = bio_results.get('max_target_rate', 15.0)
        
        status = "✓ PASS" if rate_valid else "✗ FAIL"
        self.console_logger.info(f"  Firing rate ({avg_rate:.2f}Hz): {status}")
        self.console_logger.info(f"    Target range: {min_rate}-{max_rate}Hz")
        self.console_logger.info(f"    Within biological range: {rate_valid}")
        
        # variability validation
        var_valid = bio_results['sufficient_variability']
        variability = bio_results['combined_variability']
        min_var = bio_results.get('min_variability', 0.3)
        
        status = "✓ PASS" if var_valid else "✗ FAIL"
        self.console_logger.info(f"  Variability ({variability:.3f}): {status}")
        self.console_logger.info(f"    Minimum required: {min_var}")
        self.console_logger.info(f"    Sufficient diversity: {var_valid}")
        
        # functionality validation
        func_valid = bio_results['sufficient_functionality']
        func_ratio = bio_results['functional_ratio']
        
        status = "✓ PASS" if func_valid else "✗ FAIL"
        self.console_logger.info(f"  Functionality ({func_ratio:.2f}): {status}")
        self.console_logger.info(f"    All layers active: {func_valid}")
        
        self.file_logger.info(f"Biological validation results: {bio_results}")
        
    def log_final_results(self, final_results: Dict):
        """Log final validation results"""
        overall_valid = final_results['overall_validity']
        
        self.console_logger.info("\n" + "=" * 60)
        self.console_logger.info("FINAL VALIDATION RESULTS")
        self.console_logger.info("=" * 60)
        
        if overall_valid:
            self.console_logger.info("🎉 VALIDATION: SUCCESS")
            self.console_logger.info("✓ Model meets all biological criteria")
            self.console_logger.info("✓ Production ready neuromorphic architecture")
        else:
            self.console_logger.info("⚠️  VALIDATION: NEEDS OPTIMIZATION")
            self.console_logger.info("✗ Model requires parameter adjustment")
            
            # show what failed
            if not final_results['biologically_valid_rate']:
                self.console_logger.info("  - Firing rate outside biological range")
            if not final_results['sufficient_variability']:
                self.console_logger.info("  - Insufficient neuronal variability")
            if not final_results['sufficient_functionality']:
                self.console_logger.info("  - Some layers not functioning")
        
        # save final results
        self.validation_log['final_results'] = final_results
        self.validation_log['metrics'] = {
            'total_spikes': final_results['total_spikes'],
            'avg_spike_rate': final_results['avg_spike_rate'],
            'combined_variability': final_results['combined_variability'],
            'neuronal_variability': final_results['neuronal_variability'],
            'functional_ratio': final_results['functional_ratio']
        }
        
        self.file_logger.info(f"Final validation results: {final_results}")
        
    def save_validation_report(self):
        """Save complete validation report to JSON"""
        report_file = os.path.join(self.log_dir, f'validation_report_{self.session_id}.json')
        
        with open(report_file, 'w') as f:
            json.dump(self.validation_log, f, indent=2, default=str)
            
        self.console_logger.info(f"\nValidation report saved: {report_file}")
        self.file_logger.info(f"Validation report saved to: {report_file}")
        
        return report_file
        
    def log_system_info(self):
        """Log system and GPU information"""
        self.console_logger.info("\nSystem Information:")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            current_memory = torch.cuda.memory_allocated(0) / 1e9
            
            self.console_logger.info(f"  GPU: {gpu_name}")
            self.console_logger.info(f"  Total GPU Memory: {gpu_memory:.1f}GB")
            self.console_logger.info(f"  Current GPU Usage: {current_memory:.2f}GB")
        else:
            self.console_logger.info("  GPU: Not available (using CPU)")
            
        self.file_logger.info("System info logged")
        
    def get_log_files(self):
        """Get list of generated log files"""
        log_file = os.path.join(self.log_dir, f'validation_{self.session_id}.log')
        report_file = os.path.join(self.log_dir, f'validation_report_{self.session_id}.json')
        
        return {
            'log_file': log_file,
            'report_file': report_file,
            'session_id': self.session_id
        }
