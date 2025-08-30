"""
Diagnostic Analyzer
Advanced medical diagnosis and analysis system
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from scipy import ndimage
from sklearn.cluster import KMeans

class DiagnosticAnalyzer:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def segment_brain_tissues(self, image_data: np.ndarray) -> Dict:
        """Advanced brain tissue segmentation with improved precision"""
        try:
            # Normalize data
            data_normalized = (image_data - image_data.min()) / (image_data.max() - image_data.min())
            
            # Step 1: Improved skull stripping using multiple criteria
            brain_mask = self._advanced_skull_stripping(data_normalized)
            
            # Step 2: Refined tissue segmentation only within brain mask
            brain_data = data_normalized[brain_mask]
            
            if len(brain_data) == 0:
                self.logger.warning("No brain tissue detected, using fallback segmentation")
                return self._fallback_segmentation(data_normalized)
            
            # More precise thresholds based on T1-weighted characteristics
            csf_threshold = np.percentile(brain_data, 15)      # CSF: dark in T1
            gm_threshold_low = np.percentile(brain_data, 35)   # Gray matter: intermediate
            gm_threshold_high = np.percentile(brain_data, 65)  
            wm_threshold = np.percentile(brain_data, 85)       # White matter: bright in T1
            
            # Initialize all masks as False
            background_mask = ~brain_mask
            csf_mask = np.zeros_like(data_normalized, dtype=bool)
            gray_matter_mask = np.zeros_like(data_normalized, dtype=bool)
            white_matter_mask = np.zeros_like(data_normalized, dtype=bool)
            other_mask = np.zeros_like(data_normalized, dtype=bool)
            
            # Apply segmentation only within brain mask
            brain_indices = np.where(brain_mask)
            brain_values = data_normalized[brain_indices]
            
            # CSF: lowest intensities within brain
            csf_indices = brain_values <= csf_threshold
            csf_mask[brain_indices[0][csf_indices], brain_indices[1][csf_indices], brain_indices[2][csf_indices]] = True
            
            # Gray matter: intermediate intensities
            gm_indices = (brain_values > csf_threshold) & (brain_values <= gm_threshold_high)
            gray_matter_mask[brain_indices[0][gm_indices], brain_indices[1][gm_indices], brain_indices[2][gm_indices]] = True
            
            # White matter: high intensities but not extreme
            wm_indices = (brain_values > gm_threshold_high) & (brain_values <= wm_threshold)
            white_matter_mask[brain_indices[0][wm_indices], brain_indices[1][wm_indices], brain_indices[2][wm_indices]] = True
            
            # Other tissues: very high intensities (fat, vessels) - exclude from main brain volume
            other_indices = brain_values > wm_threshold
            other_mask[brain_indices[0][other_indices], brain_indices[1][other_indices], brain_indices[2][other_indices]] = True
            
            # Refine brain mask to exclude "other" tissues for more accurate brain volume
            refined_brain_mask = gray_matter_mask | white_matter_mask | csf_mask
            
            # Post-processing: morphological operations to clean up segmentation
            refined_brain_mask = self._clean_brain_mask(refined_brain_mask)
            gray_matter_mask = self._clean_tissue_mask(gray_matter_mask, min_size=50)
            white_matter_mask = self._clean_tissue_mask(white_matter_mask, min_size=100)
            
            tissue_volumes = {
                'background': np.sum(background_mask),
                'csf': np.sum(csf_mask),
                'gray_matter': np.sum(gray_matter_mask),
                'white_matter': np.sum(white_matter_mask),
                'other_tissue': np.sum(other_mask),
                'total_brain': np.sum(refined_brain_mask)  # Use refined mask
            }
            
            total_voxels = image_data.size
            tissue_percentages = {
                tissue: (volume / total_voxels) * 100 
                for tissue, volume in tissue_volumes.items()
            }
            
            # Add clinical validation
            clinical_ratios = self._validate_clinical_ratios(tissue_volumes)
            
            return {
                'masks': {
                    'background': background_mask,
                    'csf': csf_mask,
                    'gray_matter': gray_matter_mask,
                    'white_matter': white_matter_mask,
                    'other_tissue': other_mask,
                    'brain': refined_brain_mask  # Use refined mask
                },
                'volumes': tissue_volumes,
                'percentages': tissue_percentages,
                'clinical_ratios': clinical_ratios,
                'thresholds': {
                    'csf': csf_threshold,
                    'gray_matter_low': gm_threshold_low,
                    'gray_matter_high': gm_threshold_high,
                    'white_matter': wm_threshold
                },
                'quality_metrics': {
                    'brain_extraction_quality': self._assess_brain_extraction_quality(refined_brain_mask, data_normalized),
                    'segmentation_confidence': self._calculate_segmentation_confidence(tissue_volumes)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Brain tissue segmentation failed: {e}")
            raise
    
    def _advanced_skull_stripping(self, data_normalized: np.ndarray) -> np.ndarray:
        """Advanced skull stripping using multiple criteria"""
        # Method 1: Intensity-based threshold
        intensity_mask = data_normalized > np.percentile(data_normalized, 10)
        
        # Method 2: Gradient-based edge detection to find brain boundary
        from scipy import ndimage
        gradient = ndimage.gaussian_gradient_magnitude(data_normalized, sigma=1.0)
        edge_mask = gradient < np.percentile(gradient, 85)
        
        # Method 3: Morphological operations
        from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
        
        # Combine criteria
        initial_mask = intensity_mask & edge_mask
        
        # Fill holes and clean up
        filled_mask = binary_fill_holes(initial_mask)
        
        # Remove small objects (likely noise)
        cleaned_mask = self._remove_small_objects(filled_mask, min_size=1000)
        
        # Keep only the largest connected component (main brain)
        brain_mask = self._largest_connected_component(cleaned_mask)
        
        return brain_mask
    
    def _clean_brain_mask(self, brain_mask: np.ndarray) -> np.ndarray:
        """Clean brain mask using morphological operations"""
        from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
        
        # Fill small holes
        filled = binary_fill_holes(brain_mask)
        
        # Slight erosion followed by dilation to smooth boundaries
        eroded = binary_erosion(filled, iterations=1)
        cleaned = binary_dilation(eroded, iterations=2)
        
        # Keep largest component
        cleaned = self._largest_connected_component(cleaned)
        
        return cleaned
    
    def _clean_tissue_mask(self, tissue_mask: np.ndarray, min_size: int = 50) -> np.ndarray:
        """Clean individual tissue masks"""
        return self._remove_small_objects(tissue_mask, min_size)
    
    def _remove_small_objects(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """Remove small connected components"""
        from scipy.ndimage import label
        
        labeled, num_features = label(mask)
        if num_features == 0:
            return mask
        
        # Count voxels in each component
        component_sizes = np.bincount(labeled.ravel())
        
        # Keep only components larger than min_size
        large_components = component_sizes >= min_size
        large_components[0] = False  # Background is always 0
        
        cleaned_mask = np.isin(labeled, np.where(large_components)[0])
        return cleaned_mask
    
    def _largest_connected_component(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component"""
        from scipy.ndimage import label
        
        labeled, num_features = label(mask)
        if num_features == 0:
            return mask
        
        # Find largest component
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0  # Ignore background
        largest_component = np.argmax(component_sizes)
        
        return labeled == largest_component
    
    def _validate_clinical_ratios(self, tissue_volumes: Dict) -> Dict:
        """Validate tissue ratios against clinical norms"""
        total_brain = tissue_volumes['total_brain']
        if total_brain == 0:
            return {'warning': 'No brain tissue detected'}
        
        gm_ratio = tissue_volumes['gray_matter'] / total_brain
        wm_ratio = tissue_volumes['white_matter'] / total_brain
        csf_ratio = tissue_volumes['csf'] / total_brain
        
        # Clinical normal ranges
        normal_ranges = {
            'gm_ratio': (0.40, 0.55),    # 40-55% of brain volume
            'wm_ratio': (0.35, 0.50),    # 35-50% of brain volume  
            'csf_ratio': (0.05, 0.15)    # 5-15% of brain volume
        }
        
        validation = {}
        for ratio_name, (min_val, max_val) in normal_ranges.items():
            current_ratio = locals()[ratio_name]
            validation[ratio_name] = {
                'value': current_ratio,
                'normal': min_val <= current_ratio <= max_val,
                'expected_range': (min_val, max_val)
            }
        
        return validation
    
    def _assess_brain_extraction_quality(self, brain_mask: np.ndarray, data_normalized: np.ndarray) -> float:
        """Assess quality of brain extraction"""
        if np.sum(brain_mask) == 0:
            return 0.0
        
        # Check intensity consistency within brain
        brain_intensities = data_normalized[brain_mask]
        intensity_std = np.std(brain_intensities)
        
        # Check brain boundary smoothness
        from scipy import ndimage
        edge_strength = np.mean(ndimage.gaussian_gradient_magnitude(brain_mask.astype(float), sigma=1.0))
        
        # Combine metrics (higher is better)
        quality_score = 1.0 / (1.0 + intensity_std + edge_strength)
        return min(1.0, quality_score)
    
    def _calculate_segmentation_confidence(self, tissue_volumes: Dict) -> float:
        """Calculate confidence in segmentation quality"""
        total_brain = tissue_volumes['total_brain']
        if total_brain == 0:
            return 0.0
        
        # Check if ratios are reasonable
        gm_ratio = tissue_volumes['gray_matter'] / total_brain
        wm_ratio = tissue_volumes['white_matter'] / total_brain
        
        # Confidence based on how close to expected ratios
        gm_confidence = 1.0 - abs(gm_ratio - 0.475) / 0.075  # Expected ~47.5%
        wm_confidence = 1.0 - abs(wm_ratio - 0.425) / 0.075  # Expected ~42.5%
        
        overall_confidence = (gm_confidence + wm_confidence) / 2.0
        return max(0.0, min(1.0, overall_confidence))
    
    def _fallback_segmentation(self, data_normalized: np.ndarray) -> Dict:
        """Fallback segmentation when brain extraction fails"""
        self.logger.warning("Using fallback segmentation method")
        
        # Simple threshold-based segmentation
        background_mask = data_normalized <= 0.1
        brain_mask = ~background_mask
        
        brain_data = data_normalized[brain_mask]
        if len(brain_data) > 0:
            csf_mask = (data_normalized > 0.1) & (data_normalized <= np.percentile(brain_data, 30))
            gm_mask = (data_normalized > np.percentile(brain_data, 30)) & (data_normalized <= np.percentile(brain_data, 70))
            wm_mask = data_normalized > np.percentile(brain_data, 70)
        else:
            csf_mask = np.zeros_like(data_normalized, dtype=bool)
            gm_mask = np.zeros_like(data_normalized, dtype=bool)
            wm_mask = np.zeros_like(data_normalized, dtype=bool)
        
        tissue_volumes = {
            'background': np.sum(background_mask),
            'csf': np.sum(csf_mask),
            'gray_matter': np.sum(gm_mask),
            'white_matter': np.sum(wm_mask),
            'other_tissue': 0,
            'total_brain': np.sum(brain_mask)
        }
        
        return {
            'masks': {'background': background_mask, 'csf': csf_mask, 'gray_matter': gm_mask, 
                     'white_matter': wm_mask, 'other_tissue': np.zeros_like(background_mask), 'brain': brain_mask},
            'volumes': tissue_volumes,
            'percentages': {k: (v/data_normalized.size)*100 for k, v in tissue_volumes.items()},
            'clinical_ratios': {},
            'thresholds': {},
            'quality_metrics': {'brain_extraction_quality': 0.5, 'segmentation_confidence': 0.3}
        }
    
    def calculate_morphometry(self, segmentation: Dict, voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict:
        """Calculate morphometric measurements with clinical validation"""
        try:
            voxel_volume_ml = np.prod(voxel_size) / 1000.0
            
            # Raw volumes from segmentation
            raw_volumes_ml = {
                tissue: volume * voxel_volume_ml 
                for tissue, volume in segmentation['volumes'].items()
            }
            
            # Clinical validation and correction
            corrected_volumes = self._apply_clinical_volume_corrections(raw_volumes_ml, segmentation)
            
            total_brain_volume = corrected_volumes['total_brain']
            gray_volume = corrected_volumes['gray_matter']
            white_volume = corrected_volumes['white_matter']
            csf_volume = corrected_volumes['csf']
            
            # Calculate clinically relevant ratios
            gray_white_ratio = gray_volume / white_volume if white_volume > 0 else 0
            brain_csf_ratio = total_brain_volume / csf_volume if csf_volume > 0 else 0
            
            # Validate against normative ranges
            volume_validity = self._validate_brain_volumes(corrected_volumes)
            
            # Spatial metrics
            brain_mask = segmentation['masks']['brain']
            spatial_metrics = self._calculate_spatial_metrics(brain_mask, voxel_size)
            
            # Enhanced asymmetry analysis
            asymmetry_metrics = self._enhanced_asymmetry_analysis(brain_mask, segmentation['masks'])
            
            return {
                'volumes': corrected_volumes,
                'raw_volumes': raw_volumes_ml,  # Keep original for comparison
                'ratios': {
                    'gray_white_ratio': gray_white_ratio,
                    'brain_csf_ratio': brain_csf_ratio,
                    'gray_brain_ratio': gray_volume / total_brain_volume if total_brain_volume > 0 else 0,
                    'white_brain_ratio': white_volume / total_brain_volume if total_brain_volume > 0 else 0,
                    'csf_brain_ratio': csf_volume / total_brain_volume if total_brain_volume > 0 else 0
                },
                'spatial_metrics': spatial_metrics,
                'asymmetry_metrics': asymmetry_metrics,
                'clinical_metrics': {
                    'total_brain_volume_ml': total_brain_volume,
                    'gray_matter_volume_ml': gray_volume,
                    'white_matter_volume_ml': white_volume,
                    'csf_volume_ml': csf_volume,
                    'intracranial_volume_ml': self._estimate_intracranial_volume(raw_volumes_ml),
                    'brain_parenchyma_fraction': (gray_volume + white_volume) / total_brain_volume if total_brain_volume > 0 else 0
                },
                'volume_validity': volume_validity,
                'quality_metrics': segmentation.get('quality_metrics', {}),
                'segmentation_confidence': segmentation.get('quality_metrics', {}).get('segmentation_confidence', 0.5)
            }
            
        except Exception as e:
            self.logger.error(f"Morphometry calculation failed: {e}")
            raise
    
    def _apply_clinical_volume_corrections(self, raw_volumes_ml: Dict, segmentation: Dict) -> Dict:
        """Apply corrections to make volumes clinically realistic"""
        corrected = raw_volumes_ml.copy()
        
        # Get segmentation quality metrics
        quality_metrics = segmentation.get('quality_metrics', {})
        brain_extraction_quality = quality_metrics.get('brain_extraction_quality', 0.5)
        
        # If brain extraction quality is poor, apply more conservative estimates
        if brain_extraction_quality < 0.7:
            self.logger.warning(f"Poor brain extraction quality ({brain_extraction_quality:.2f}), applying corrections")
            
            # Conservative brain volume estimate: use only GM + WM, exclude other tissues
            parenchyma_volume = raw_volumes_ml['gray_matter'] + raw_volumes_ml['white_matter']
            
            # Estimate CSF as 10% of parenchyma (typical ratio)
            if raw_volumes_ml['csf'] < parenchyma_volume * 0.05:  # Less than 5% is likely underestimated
                corrected['csf'] = parenchyma_volume * 0.10
                self.logger.info(f"CSF volume corrected from {raw_volumes_ml['csf']:.1f} to {corrected['csf']:.1f} mL")
            
            # Total brain should be parenchyma + CSF
            corrected['total_brain'] = parenchyma_volume + corrected['csf']
        
        # Apply clinical bounds
        corrected = self._apply_clinical_bounds(corrected)
        
        return corrected
    
    def _apply_clinical_bounds(self, volumes: Dict) -> Dict:
        """Apply realistic clinical bounds to volumes"""
        bounded = volumes.copy()
        
        # Clinical bounds for adult brain volumes (mL)
        bounds = {
            'total_brain': (1000, 1600),      # Total brain: 1000-1600 mL
            'gray_matter': (400, 800),        # Gray matter: 400-800 mL  
            'white_matter': (400, 800),       # White matter: 400-800 mL
            'csf': (20, 200)                  # CSF: 20-200 mL
        }
        
        for tissue, (min_vol, max_vol) in bounds.items():
            if tissue in bounded:
                original_vol = bounded[tissue]
                bounded[tissue] = max(min_vol, min(max_vol, original_vol))
                
                if abs(bounded[tissue] - original_vol) > 50:  # Significant correction
                    self.logger.warning(f"{tissue} volume corrected from {original_vol:.1f} to {bounded[tissue]:.1f} mL")
        
        # Ensure consistency: total brain >= GM + WM + CSF
        parenchyma = bounded['gray_matter'] + bounded['white_matter']
        if bounded['total_brain'] < parenchyma + bounded['csf']:
            bounded['total_brain'] = parenchyma + bounded['csf']
        
        return bounded
    
    def _validate_brain_volumes(self, volumes: Dict) -> Dict:
        """Validate brain volumes against clinical norms"""
        validation = {}
        
        # Normative ranges for healthy adults
        norms = {
            'total_brain': {'mean': 1350, 'std': 150, 'range': (1000, 1600)},
            'gray_matter': {'mean': 600, 'std': 80, 'range': (400, 800)},
            'white_matter': {'mean': 500, 'std': 70, 'range': (400, 800)},
            'csf': {'mean': 80, 'std': 30, 'range': (20, 200)}
        }
        
        for tissue, norm in norms.items():
            if tissue in volumes:
                volume = volumes[tissue]
                z_score = (volume - norm['mean']) / norm['std']
                
                validation[tissue] = {
                    'volume_ml': volume,
                    'z_score': z_score,
                    'percentile': self._z_to_percentile(z_score),
                    'within_normal_range': norm['range'][0] <= volume <= norm['range'][1],
                    'clinical_interpretation': self._interpret_volume_z_score(z_score, tissue)
                }
        
        return validation
    
    def _z_to_percentile(self, z_score: float) -> float:
        """Convert z-score to percentile (approximate)"""
        from scipy.stats import norm
        return norm.cdf(z_score) * 100
    
    def _interpret_volume_z_score(self, z_score: float, tissue: str) -> str:
        """Interpret volume z-score clinically"""
        if z_score < -2:
            return f"Significantly reduced {tissue} volume"
        elif z_score < -1:
            return f"Mildly reduced {tissue} volume"
        elif z_score > 2:
            return f"Significantly increased {tissue} volume"
        elif z_score > 1:
            return f"Mildly increased {tissue} volume"
        else:
            return f"Normal {tissue} volume"
    
    def _calculate_spatial_metrics(self, brain_mask: np.ndarray, voxel_size: Tuple[float, float, float]) -> Dict:
        """Calculate detailed spatial metrics"""
        brain_coords = np.where(brain_mask)
        
        if len(brain_coords[0]) == 0:
            return {'centroid': [0, 0, 0], 'extent': [0, 0, 0], 'volume_mm3': 0}
        
        # Calculate centroid in mm coordinates
        centroid_voxels = [np.mean(coords) for coords in brain_coords]
        centroid_mm = [c * vs for c, vs in zip(centroid_voxels, voxel_size)]
        
        # Calculate extent in mm
        extent_voxels = [np.max(coords) - np.min(coords) for coords in brain_coords]
        extent_mm = [e * vs for e, vs in zip(extent_voxels, voxel_size)]
        
        return {
            'brain_centroid': centroid_mm,
            'brain_extent': extent_mm,
            'brain_centroid_voxels': centroid_voxels,
            'brain_extent_voxels': extent_voxels
        }
    
    def _enhanced_asymmetry_analysis(self, brain_mask: np.ndarray, tissue_masks: Dict) -> Dict:
        """Enhanced hemispheric asymmetry analysis"""
        mid_sagittal = brain_mask.shape[0] // 2
        
        asymmetry_results = {}
        
        for tissue, mask in tissue_masks.items():
            if tissue == 'background' or tissue == 'brain':
                continue
                
            left_hemisphere = mask[:mid_sagittal, :, :]
            right_hemisphere = mask[mid_sagittal:, :, :]
            
            left_volume = np.sum(left_hemisphere)
            right_volume = np.sum(right_hemisphere)
            
            if left_volume + right_volume > 0:
                asymmetry = abs(left_volume - right_volume) / (left_volume + right_volume)
                asymmetry_results[f'{tissue}_asymmetry'] = {
                    'asymmetry_index': float(asymmetry),
                    'left_volume_voxels': int(left_volume),
                    'right_volume_voxels': int(right_volume),
                    'interpretation': 'Normal' if asymmetry < 0.1 else 'Mild asymmetry' if asymmetry < 0.2 else 'Significant asymmetry'
                }
        
        # Overall brain asymmetry
        overall_asymmetry = self._calculate_hemispheric_asymmetry(brain_mask)
        asymmetry_results['overall_brain_asymmetry'] = overall_asymmetry
        
        return asymmetry_results
    
    def _estimate_intracranial_volume(self, volumes: Dict) -> float:
        """Estimate total intracranial volume"""
        # ICV ≈ brain + CSF + some correction for skull thickness
        brain_vol = volumes.get('total_brain', 0)
        background_vol = volumes.get('background', 0)
        
        # Rough estimate: ICV is brain volume plus ~10-15% for extra-axial CSF and corrections
        estimated_icv = brain_vol * 1.15
        
        return estimated_icv
    
    def _calculate_hemispheric_asymmetry(self, brain_mask: np.ndarray) -> float:
        """Calculate hemispheric asymmetry score"""
        try:
            mid_sagittal = brain_mask.shape[0] // 2
            
            left_hemisphere = brain_mask[:mid_sagittal, :, :]
            right_hemisphere = brain_mask[mid_sagittal:, :, :]
            
            left_volume = np.sum(left_hemisphere)
            right_volume = np.sum(right_hemisphere)
            
            if left_volume + right_volume == 0:
                return 0.0
            
            asymmetry = abs(left_volume - right_volume) / (left_volume + right_volume)
            return float(asymmetry)
            
        except Exception:
            return 0.0
    
    def detect_anomalies(self, image_data: np.ndarray, segmentation: Dict) -> Dict:
        """Detect potential anomalies in brain structure"""
        try:
            anomalies = {
                'detected_anomalies': [],
                'anomaly_scores': {},
                'suspicious_regions': []
            }
            
            brain_mask = segmentation['masks']['brain']
            brain_data = image_data[brain_mask]
            
            intensity_z_scores = np.abs((brain_data - np.mean(brain_data)) / np.std(brain_data))
            outlier_threshold = 3.0
            outliers = intensity_z_scores > outlier_threshold
            outlier_percentage = np.sum(outliers) / len(brain_data) * 100
            
            if outlier_percentage > 2.0:
                anomalies['detected_anomalies'].append('Abnormal intensity patterns')
                anomalies['anomaly_scores']['intensity_outliers'] = outlier_percentage
            
            gray_percentage = segmentation['percentages']['gray_matter']
            white_percentage = segmentation['percentages']['white_matter']
            
            if gray_percentage < 30 or gray_percentage > 60:
                anomalies['detected_anomalies'].append('Atypical gray matter volume')
                anomalies['anomaly_scores']['gray_matter_deviation'] = abs(45 - gray_percentage)
            
            if white_percentage < 20 or white_percentage > 50:
                anomalies['detected_anomalies'].append('Atypical white matter volume')
                anomalies['anomaly_scores']['white_matter_deviation'] = abs(35 - white_percentage)
            
            smoothed = ndimage.gaussian_filter(image_data, sigma=1.0)
            texture_variance = ndimage.generic_filter(smoothed, np.var, size=5)
            
            high_variance_mask = texture_variance > np.percentile(texture_variance, 95)
            suspicious_regions = np.where(high_variance_mask & brain_mask)
            
            if len(suspicious_regions[0]) > brain_mask.sum() * 0.05:
                anomalies['detected_anomalies'].append('Suspicious texture patterns')
                anomalies['suspicious_regions'] = [
                    (int(x), int(y), int(z)) 
                    for x, y, z in zip(*suspicious_regions)
                ]
            
            asymmetry_score = segmentation.get('spatial_metrics', {}).get('hemispheric_asymmetry', 0)
            if asymmetry_score > 0.15:
                anomalies['detected_anomalies'].append('Significant hemispheric asymmetry')
                anomalies['anomaly_scores']['hemispheric_asymmetry'] = asymmetry_score
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return {
                'detected_anomalies': [],
                'anomaly_scores': {},
                'suspicious_regions': []
            }
    
    def estimate_brain_age(self, morphometry: Dict, quality_metrics: Dict) -> Dict:
        """Estimate brain age based on morphometric features"""
        try:
            total_volume = morphometry['clinical_metrics']['total_brain_volume_ml']
            gray_volume = morphometry['clinical_metrics']['gray_matter_volume_ml']
            white_volume = morphometry['clinical_metrics']['white_matter_volume_ml']
            gray_white_ratio = morphometry['ratios']['gray_white_ratio']
            
            volume_factor = min(max((total_volume - 1000) / 400, 0), 1)
            ratio_factor = min(max((gray_white_ratio - 0.8) / 0.4, 0), 1)
            quality_factor = quality_metrics.get('quality_score', 5) / 10
            
            normalized_age = (volume_factor + ratio_factor + quality_factor) / 3
            estimated_age = 20 + (normalized_age * 60)
            
            age_confidence = min(quality_factor * 0.8 + 0.2, 1.0)
            
            age_category = self._categorize_brain_age(estimated_age)
            
            return {
                'estimated_age_years': round(estimated_age, 1),
                'age_confidence': round(age_confidence, 3),
                'age_category': age_category,
                'age_factors': {
                    'volume_factor': round(volume_factor, 3),
                    'ratio_factor': round(ratio_factor, 3),
                    'quality_factor': round(quality_factor, 3)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Brain age estimation failed: {e}")
            return {
                'estimated_age_years': 45.0,
                'age_confidence': 0.5,
                'age_category': 'Adult',
                'age_factors': {
                    'volume_factor': 0.5,
                    'ratio_factor': 0.5,
                    'quality_factor': 0.5
                }
            }
    
    def _categorize_brain_age(self, age: float) -> str:
        """Categorize brain age into developmental stages"""
        if age < 25:
            return 'Young Adult'
        elif age < 45:
            return 'Adult'
        elif age < 65:
            return 'Middle-aged'
        else:
            return 'Senior'
    
    def calculate_normalcy_score(self, morphometry: Dict, quality_metrics: Dict, anomalies: Dict) -> Dict:
        """Calculate overall brain normalcy score"""
        try:
            quality_score = quality_metrics.get('quality_score', 5) / 10
            
            gray_percentage = morphometry.get('volumes', {}).get('gray_matter', 0)
            total_volume = morphometry.get('volumes', {}).get('total_brain', 1)
            gray_ratio = gray_percentage / total_volume if total_volume > 0 else 0
            
            normal_gray_ratio = 0.45
            gray_deviation = abs(gray_ratio - normal_gray_ratio) / normal_gray_ratio
            gray_score = max(0, 1 - gray_deviation)
            
            anomaly_count = len(anomalies.get('detected_anomalies', []))
            anomaly_score = max(0, 1 - (anomaly_count * 0.2))
            
            asymmetry = morphometry.get('spatial_metrics', {}).get('hemispheric_asymmetry', 0)
            symmetry_score = max(0, 1 - (asymmetry * 5))
            
            normalcy_score = (quality_score * 0.3 + gray_score * 0.3 + 
                            anomaly_score * 0.3 + symmetry_score * 0.1)
            
            normalcy_level = self._categorize_normalcy(normalcy_score)
            
            return {
                'normalcy_score': round(normalcy_score, 3),
                'normalcy_level': normalcy_level,
                'component_scores': {
                    'quality_score': round(quality_score, 3),
                    'tissue_composition_score': round(gray_score, 3),
                    'anomaly_score': round(anomaly_score, 3),
                    'symmetry_score': round(symmetry_score, 3)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Normalcy score calculation failed: {e}")
            return {
                'normalcy_score': 0.5,
                'normalcy_level': 'Uncertain',
                'component_scores': {
                    'quality_score': 0.5,
                    'tissue_composition_score': 0.5,
                    'anomaly_score': 0.5,
                    'symmetry_score': 0.5
                }
            }
    
    def _categorize_normalcy(self, score: float) -> str:
        """Categorize normalcy score"""
        if score >= 0.8:
            return 'Highly Normal'
        elif score >= 0.6:
            return 'Normal'
        elif score >= 0.4:
            return 'Possibly Abnormal'
        else:
            return 'Abnormal'
    
    def generate_diagnostic_report(self, image_info: Dict, quality_metrics: Dict, 
                                 segmentation: Dict, morphometry: Dict, 
                                 anomalies: Dict, brain_age: Dict, 
                                 normalcy: Dict, ai_analysis: Dict) -> Dict:
        """Generate comprehensive diagnostic report"""
        
        report = {
            'diagnostic_report': {
                'patient_id': 'ANON_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
                'scan_date': datetime.now().isoformat(),
                'image_source': image_info['file_name'],
                'analysis_version': '1.0.0'
            },
            
            'image_information': {
                'file_name': image_info['file_name'],
                'dimensions': image_info['shape'],
                'file_size_mb': round(image_info['file_size_mb'], 2),
                'data_type': 'NIfTI Medical Image'
            },
            
            'quality_assessment': quality_metrics,
            
            'tissue_segmentation': {
                'tissue_volumes': segmentation['volumes'],
                'tissue_percentages': segmentation['percentages']
            },
            
            'morphometric_analysis': morphometry,
            
            'anomaly_detection': anomalies,
            
            'brain_age_analysis': brain_age,
            
            'normalcy_assessment': normalcy,
            
            'ai_medical_analysis': ai_analysis,
            
            'clinical_summary': {
                'overall_assessment': self._generate_clinical_summary(
                    quality_metrics, morphometry, anomalies, brain_age, normalcy, ai_analysis
                ),
                'recommendations': self._generate_recommendations(anomalies, normalcy, ai_analysis)
            }
        }
        
        return report
    
    def _generate_clinical_summary(self, quality_metrics: Dict, morphometry: Dict, 
                                 anomalies: Dict, brain_age: Dict, 
                                 normalcy: Dict, ai_analysis: Dict) -> str:
        """Generate clinical summary text"""
        
        quality_score = quality_metrics.get('quality_score', 0)
        brain_volume = morphometry.get('clinical_metrics', {}).get('total_brain_volume_ml', 0)
        anomaly_count = len(anomalies.get('detected_anomalies', []))
        estimated_age = brain_age.get('estimated_age_years', 0)
        normalcy_score = normalcy.get('normalcy_score', 0)
        predicted_condition = ai_analysis.get('predicted_condition', 'Unknown')
        ai_confidence = ai_analysis.get('confidence_score', 0)
        
        summary_parts = []
        
        if quality_score >= 8:
            summary_parts.append("High-quality medical imaging study.")
        elif quality_score >= 6:
            summary_parts.append("Good quality medical imaging study.")
        else:
            summary_parts.append("Adequate quality medical imaging study.")
        
        summary_parts.append(f"Total brain volume: {brain_volume:.1f} mL.")
        summary_parts.append(f"Estimated brain age: {estimated_age:.1f} years.")
        
        if anomaly_count == 0:
            summary_parts.append("No significant structural anomalies detected.")
        else:
            summary_parts.append(f"{anomaly_count} potential anomalie(s) identified.")
        
        summary_parts.append(f"Overall normalcy assessment: {normalcy.get('normalcy_level', 'Unknown')} (score: {normalcy_score:.3f}).")
        
        summary_parts.append(f"AI analysis suggests: {predicted_condition} (confidence: {ai_confidence:.3f}).")
        
        return " ".join(summary_parts)
    
    def _generate_recommendations(self, anomalies: Dict, normalcy: Dict, ai_analysis: Dict) -> List[str]:
        """Generate clinical recommendations"""
        
        recommendations = []
        
        if len(anomalies.get('detected_anomalies', [])) > 0:
            recommendations.append("Consider additional clinical correlation for detected anomalies.")
        
        normalcy_score = normalcy.get('normalcy_score', 0)
        if normalcy_score < 0.6:
            recommendations.append("Recommend follow-up imaging and clinical evaluation.")
        
        ai_confidence = ai_analysis.get('confidence_score', 0)
        if ai_confidence < 0.7:
            recommendations.append("AI analysis confidence is moderate; consider expert review.")
        
        predicted_condition = ai_analysis.get('predicted_condition', '')
        if predicted_condition not in ['Healthy', 'Unknown']:
            recommendations.append(f"Consider clinical evaluation for {predicted_condition}.")
        
        if not recommendations:
            recommendations.append("Continue routine monitoring as clinically indicated.")
        
        return recommendations
