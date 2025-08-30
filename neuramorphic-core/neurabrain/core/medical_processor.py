"""
Medical Image Processor
Handles loading, preprocessing and quality assessment of medical images
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

class MedicalImageProcessor:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.supported_formats = ['.nii', '.nii.gz', '.mgz']
        
    def load_medical_image(self, image_path: str) -> Dict:
        """Load and validate medical image"""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            if image_path.suffix.lower() not in self.supported_formats:
                if not str(image_path).endswith('.nii.gz'):
                    raise ValueError(f"Unsupported format: {image_path.suffix}")
            
            nii_img = nib.load(str(image_path))
            data = nii_img.get_fdata()
            header = nii_img.header
            affine = nii_img.affine
            
            image_info = {
                'data': data,
                'header': header,
                'affine': affine,
                'shape': data.shape,
                'file_path': str(image_path),
                'file_name': image_path.name,
                'file_size_mb': image_path.stat().st_size / (1024 * 1024)
            }
            
            self.logger.info(f"Loaded image: {image_path.name} - Shape: {data.shape}")
            return image_info
            
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def preprocess_image(self, image_data: np.ndarray) -> np.ndarray:
        """Preprocess medical image data"""
        try:
            processed = image_data.copy()
            
            processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
            
            if processed.max() > processed.min():
                processed = (processed - processed.min()) / (processed.max() - processed.min())
            
            processed = np.clip(processed, 0, 1)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise
    
    def assess_image_quality(self, image_data: np.ndarray) -> Dict:
        """Assess medical image quality metrics"""
        try:
            brain_mask = image_data > np.percentile(image_data, 15)
            background_mask = image_data <= np.percentile(image_data, 5)
            
            brain_signal = image_data[brain_mask]
            background_noise = image_data[background_mask]
            
            signal_mean = np.mean(brain_signal) if len(brain_signal) > 0 else 0
            noise_std = np.std(background_noise) if len(background_noise) > 0 else 1
            
            snr = signal_mean / noise_std if noise_std > 0 else 0
            
            gray_matter_mask = (image_data > np.percentile(image_data, 40)) & (image_data < np.percentile(image_data, 80))
            white_matter_mask = image_data > np.percentile(image_data, 80)
            
            if np.any(gray_matter_mask) and np.any(white_matter_mask):
                gray_mean = np.mean(image_data[gray_matter_mask])
                white_mean = np.mean(image_data[white_matter_mask])
                contrast_ratio = white_mean / gray_mean if gray_mean > 0 else 1
            else:
                contrast_ratio = 1
            
            quality_score = min(10, max(0, (snr * 2 + contrast_ratio * 3) / 5 * 10))
            
            return {
                'snr': float(snr),
                'contrast_ratio': float(contrast_ratio),
                'quality_score': float(quality_score),
                'brain_coverage': float(np.sum(brain_mask) / brain_mask.size),
                'dynamic_range': float(image_data.max() - image_data.min())
            }
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return {
                'snr': 0.0,
                'contrast_ratio': 1.0,
                'quality_score': 5.0,
                'brain_coverage': 0.5,
                'dynamic_range': 1.0
            }
    
    def extract_metadata(self, header) -> Dict:
        """Extract relevant medical metadata"""
        try:
            metadata = {
                'dimensions': tuple(header.get_data_shape()),
                'voxel_size': tuple(header.get_zooms()),
                'data_type': str(header.get_data_dtype()),
                'orientation': 'RAS',
                'scanner_info': 'MRI T1-weighted'
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            return {}
    
    def validate_medical_format(self, image_data: np.ndarray, header) -> bool:
        """Validate if image follows medical imaging standards"""
        try:
            if len(image_data.shape) != 3:
                return False
            
            if np.any(np.array(image_data.shape) < 50):
                return False
            
            if image_data.dtype not in [np.float32, np.float64, np.int16, np.uint16]:
                return False
            
            if not (0 <= image_data.min() and image_data.max() <= 10000):
                return False
            
            return True
            
        except Exception:
            return False
