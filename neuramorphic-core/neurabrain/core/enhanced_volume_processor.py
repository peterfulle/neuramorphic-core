"""
Enhanced 3D Volume Slice Processor for Neuromorphic Medical Analysis
Processes each 2D slice individually and creates volumetric reconstructions
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import nibabel as nib

class SliceBySliceProcessor:
    """
    Processes 3D medical volumes slice by slice for detailed neuromorphic analysis
    """
    
    def __init__(self, neuromorphic_core, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.neuromorphic_core = neuromorphic_core
        self.slice_results = []
        
    def extract_all_slices(self, volume_3d: np.ndarray, axis: int = 2) -> List[np.ndarray]:
        """
        Extract all 2D slices from 3D volume
        
        Args:
            volume_3d: 3D numpy array (e.g., 256x256x150)
            axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
        
        Returns:
            List of 2D slices
        """
        slices = []
        num_slices = volume_3d.shape[axis]
        
        self.logger.info(f"📊 Extrayendo {num_slices} cortes del volumen 3D")
        self.logger.info(f"📐 Dimensiones del volumen: {volume_3d.shape}")
        
        for i in range(num_slices):
            if axis == 0:  # Sagittal
                slice_2d = volume_3d[i, :, :]
            elif axis == 1:  # Coronal
                slice_2d = volume_3d[:, i, :]
            else:  # Axial (default)
                slice_2d = volume_3d[:, :, i]
            
            # Normalize slice
            if slice_2d.max() > slice_2d.min():
                slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
            
            slices.append(slice_2d)
        
        self.logger.info(f"✅ Extraídos {len(slices)} cortes 2D de {slices[0].shape}")
        return slices
    
    def process_slice_with_neuromorphic_core(self, slice_2d: np.ndarray, slice_idx: int) -> Dict:
        """
        Process individual 2D slice with neuromorphic core
        
        Args:
            slice_2d: 2D numpy array representing brain slice
            slice_idx: Index of the slice in the volume
            
        Returns:
            Analysis results for this slice
        """
        try:
            # Convert slice to features that neuromorphic core can process
            features = self._extract_slice_features(slice_2d)
            
            # Process with neuromorphic core
            neuromorphic_output = self.neuromorphic_core.medical_inference(features)
            
            # Add slice-specific metadata
            slice_analysis = {
                'slice_index': slice_idx,
                'slice_shape': slice_2d.shape,
                'slice_mean_intensity': float(np.mean(slice_2d)),
                'slice_std_intensity': float(np.std(slice_2d)),
                'slice_max_intensity': float(np.max(slice_2d)),
                'slice_min_intensity': float(np.min(slice_2d)),
                'neuromorphic_analysis': neuromorphic_output,
                'tissue_distribution': self._analyze_slice_tissues(slice_2d)
            }
            
            return slice_analysis
            
        except Exception as e:
            self.logger.error(f"Error procesando slice {slice_idx}: {e}")
            return {
                'slice_index': slice_idx,
                'error': str(e),
                'status': 'failed'
            }
    
    def _extract_slice_features(self, slice_2d: np.ndarray) -> np.ndarray:
        """
        Extract meaningful features from 2D slice for neuromorphic processing
        """
        features = []
        
        # Basic intensity statistics
        features.extend([
            np.mean(slice_2d),
            np.std(slice_2d),
            np.median(slice_2d),
            np.percentile(slice_2d, 25),
            np.percentile(slice_2d, 75)
        ])
        
        # Texture features (simple)
        # Compute gradients
        grad_x = np.gradient(slice_2d, axis=1)
        grad_y = np.gradient(slice_2d, axis=0)
        
        features.extend([
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y)),
            np.std(grad_x),
            np.std(grad_y)
        ])
        
        # Edge density
        edges = cv2.Canny((slice_2d * 255).astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        return np.array(features, dtype=np.float32)
    
    def _analyze_slice_tissues(self, slice_2d: np.ndarray) -> Dict:
        """
        Analyze tissue distribution in slice
        """
        # Simple intensity-based tissue classification
        background_mask = slice_2d < 0.1
        csf_mask = (slice_2d >= 0.1) & (slice_2d < 0.3)
        gray_matter_mask = (slice_2d >= 0.3) & (slice_2d < 0.7)
        white_matter_mask = slice_2d >= 0.7
        
        total_pixels = slice_2d.size
        
        return {
            'background_ratio': float(np.sum(background_mask) / total_pixels),
            'csf_ratio': float(np.sum(csf_mask) / total_pixels),
            'gray_matter_ratio': float(np.sum(gray_matter_mask) / total_pixels),
            'white_matter_ratio': float(np.sum(white_matter_mask) / total_pixels)
        }
    
    def process_full_volume_slicewise(self, volume_3d: np.ndarray, 
                                    axis: int = 2) -> Dict:
        """
        Process entire 3D volume slice by slice
        
        Args:
            volume_3d: 3D medical volume
            axis: Slicing axis (0=sagittal, 1=coronal, 2=axial)
            
        Returns:
            Complete volumetric analysis
        """
        self.logger.info("🧠 Iniciando procesamiento slice-por-slice del volumen completo")
        
        # Extract all slices
        slices = self.extract_all_slices(volume_3d, axis)
        
        # Process each slice
        slice_analyses = []
        for i, slice_2d in enumerate(slices):
            self.logger.info(f"🔬 Procesando slice {i+1}/{len(slices)}")
            
            slice_result = self.process_slice_with_neuromorphic_core(slice_2d, i)
            slice_analyses.append(slice_result)
        
        # Aggregate results across all slices
        volumetric_analysis = self._aggregate_slice_results(slice_analyses, volume_3d.shape)
        
        # Store for video generation
        self.slice_results = slice_analyses
        
        self.logger.info(f"✅ Procesamiento volumétrico completado: {len(slice_analyses)} slices analizados")
        
        return {
            'slice_analyses': slice_analyses,
            'volumetric_summary': volumetric_analysis,
            'volume_shape': volume_3d.shape,
            'slice_axis': axis,
            'total_slices': len(slices)
        }
    
    def process_full_volume_slicewise_limited(self, volume_3d: np.ndarray, 
                                            output_dir: str,
                                            axis: int = 2, 
                                            max_slices: int = 10) -> Dict:
        """
        Process 3D volume slice by slice with LIMITATION (optimized for testing)
        
        Args:
            volume_3d: 3D medical volume
            axis: Slicing axis (0=sagittal, 1=coronal, 2=axial)
            max_slices: Maximum number of slices to process
            
        Returns:
            Complete volumetric analysis (limited)
        """
        self.logger.info(f"🧠 Iniciando procesamiento slice-por-slice LIMITADO del volumen completo")
        self.logger.info(f"🎯 Procesando solo {max_slices} slices de {volume_3d.shape[axis]} disponibles")
        
        # Extract all slices
        slices = self.extract_all_slices(volume_3d, axis)
        
        # Limit the number of slices to process
        limited_slices = slices[:max_slices]
        
        self.logger.info(f"📊 Slices limitados: {len(limited_slices)} de {len(slices)} totales")
        
        # Process each limited slice
        slice_analyses = []
        
        # Create img directory for saving individual slice images
        img_dir = Path(output_dir) / "img"
        img_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f"📁 Created images directory: {img_dir}")
        
        for i, slice_2d in enumerate(limited_slices):
            self.logger.info(f"🔬 Procesando slice {i+1}/{len(limited_slices)} (optimizado)")
            
            slice_result = self.process_slice_with_neuromorphic_core(slice_2d, i)
            
            # Save individual slice image
            slice_image_path = img_dir / f"slice_{i+1:03d}.png"
            self._save_slice_image(slice_2d, slice_image_path, i+1, len(limited_slices))
            slice_result['slice_image_path'] = str(slice_image_path)
            
            slice_analyses.append(slice_result)
        
        # Aggregate results across processed slices only
        volumetric_analysis = self._aggregate_slice_results(slice_analyses, volume_3d.shape)
        
        # Add optimization metadata
        volumetric_analysis['optimization_applied'] = True
        volumetric_analysis['slices_processed'] = len(limited_slices)
        volumetric_analysis['slices_available'] = len(slices)
        volumetric_analysis['processing_ratio'] = len(limited_slices) / len(slices)
        
        # Store for video generation
        self.slice_results = slice_analyses
        
        self.logger.info(f"✅ Procesamiento volumétrico LIMITADO completado: {len(slice_analyses)} slices analizados de {len(slices)} disponibles")
        
        return {
            'slice_analyses': slice_analyses,
            'volumetric_summary': volumetric_analysis,
            'volume_shape': volume_3d.shape,
            'slice_axis': axis,
            'total_slices': len(limited_slices),
            'optimization_metadata': {
                'max_slices_setting': max_slices,
                'total_available_slices': len(slices),
                'processing_efficiency': f"{len(limited_slices)}/{len(slices)} slices"
            }
        }
    
    def _aggregate_slice_results(self, slice_analyses: List[Dict], 
                               volume_shape: Tuple) -> Dict:
        """
        Aggregate results from all slices into volumetric summary
        """
        successful_analyses = [s for s in slice_analyses if 'error' not in s]
        
        if not successful_analyses:
            return {'error': 'No successful slice analyses'}
        
        # Aggregate neuromorphic predictions
        all_predictions = []
        all_confidences = []
        
        for analysis in successful_analyses:
            if 'neuromorphic_analysis' in analysis:
                neuro_result = analysis['neuromorphic_analysis']
                if 'prediction' in neuro_result:
                    all_predictions.append(neuro_result['prediction'])
                if 'confidence' in neuro_result:
                    all_confidences.append(neuro_result['confidence'])
        
        # Compute volumetric statistics
        slice_intensities = [s['slice_mean_intensity'] for s in successful_analyses]
        tissue_ratios = [s['tissue_distribution'] for s in successful_analyses]
        
        # Average tissue distribution across volume
        avg_tissue_dist = {}
        if tissue_ratios:
            for tissue_type in tissue_ratios[0].keys():
                avg_tissue_dist[tissue_type] = np.mean([tr[tissue_type] for tr in tissue_ratios])
        
        return {
            'total_successful_slices': len(successful_analyses),
            'total_failed_slices': len(slice_analyses) - len(successful_analyses),
            'volume_predictions': {
                'predictions': all_predictions,
                'mean_prediction': np.mean(all_predictions) if all_predictions else 0,
                'prediction_std': np.std(all_predictions) if all_predictions else 0,
                'mean_confidence': np.mean(all_confidences) if all_confidences else 0
            },
            'volume_intensity_stats': {
                'mean_across_slices': np.mean(slice_intensities),
                'std_across_slices': np.std(slice_intensities),
                'min_slice_intensity': np.min(slice_intensities),
                'max_slice_intensity': np.max(slice_intensities)
            },
            'average_tissue_distribution': avg_tissue_dist,
            'volume_shape': volume_shape
        }
    
    def generate_slice_video(self, slices: List[np.ndarray], 
                           output_path: str, fps: int = 10) -> str:
        """
        Generate video from processed slices
        
        Args:
            slices: List of 2D slices
            output_path: Path to save video
            fps: Frames per second
            
        Returns:
            Path to generated video
        """
        self.logger.info(f"🎬 Generando video de {len(slices)} slices a {fps} FPS")
        
        # Setup video writer
        height, width = slices[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        video_path = str(Path(output_path) / 'brain_slices_video.mp4')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height), False)
        
        for i, slice_2d in enumerate(slices):
            # Convert to uint8
            slice_uint8 = (slice_2d * 255).astype(np.uint8)
            
            # Ensure the array is contiguous for OpenCV
            slice_uint8 = np.ascontiguousarray(slice_uint8)
            
            # Add slice number overlay
            cv2.putText(slice_uint8, f'Slice {i+1}/{len(slices)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            
            out.write(slice_uint8)
        
        out.release()
        
        self.logger.info(f"✅ Video generado: {video_path}")
        return video_path
    
    def create_enhanced_visualizations(self, volume_3d: np.ndarray, 
                                     analysis_results: Dict,
                                     output_dir: str) -> Dict:
        """
        Create enhanced visualizations showing slice-by-slice analysis
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        visualizations = {}
        
        # 1. Multi-planar reconstruction
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Show sample slices from different orientations
        mid_sagittal = volume_3d.shape[0] // 2
        mid_coronal = volume_3d.shape[1] // 2
        mid_axial = volume_3d.shape[2] // 2
        
        axes[0, 0].imshow(volume_3d[mid_sagittal, :, :], cmap='gray')
        axes[0, 0].set_title(f'Sagittal (slice {mid_sagittal})')
        
        axes[0, 1].imshow(volume_3d[:, mid_coronal, :], cmap='gray')
        axes[0, 1].set_title(f'Coronal (slice {mid_coronal})')
        
        axes[0, 2].imshow(volume_3d[:, :, mid_axial], cmap='gray')
        axes[0, 2].set_title(f'Axial (slice {mid_axial})')
        
        # Show analysis results
        if 'slice_analyses' in analysis_results:
            slice_predictions = []
            slice_confidences = []
            
            for analysis in analysis_results['slice_analyses']:
                if 'neuromorphic_analysis' in analysis:
                    neuro = analysis['neuromorphic_analysis']
                    if 'prediction' in neuro:
                        slice_predictions.append(neuro['prediction'])
                    if 'confidence' in neuro:
                        slice_confidences.append(neuro['confidence'])
            
            if slice_predictions:
                axes[1, 0].plot(slice_predictions)
                axes[1, 0].set_title('Predictions por Slice')
                axes[1, 0].set_xlabel('Slice Index')
                axes[1, 0].set_ylabel('Prediction')
            
            if slice_confidences:
                axes[1, 1].plot(slice_confidences)
                axes[1, 1].set_title('Confidence por Slice')
                axes[1, 1].set_xlabel('Slice Index')
                axes[1, 1].set_ylabel('Confidence')
        
        # Tissue distribution heatmap
        if 'volumetric_summary' in analysis_results:
            tissue_dist = analysis_results['volumetric_summary'].get('average_tissue_distribution', {})
            if tissue_dist:
                tissues = list(tissue_dist.keys())
                values = list(tissue_dist.values())
                
                axes[1, 2].bar(tissues, values)
                axes[1, 2].set_title('Distribución Promedio de Tejidos')
                axes[1, 2].set_ylabel('Ratio')
                plt.setp(axes[1, 2].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        multi_planar_path = output_path / 'multi_planar_analysis.png'
        plt.savefig(multi_planar_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualizations['multi_planar'] = str(multi_planar_path)
        
        self.logger.info(f"✅ Visualizaciones mejoradas guardadas en {output_path}")
        
        return visualizations

    def _save_slice_image(self, slice_2d: np.ndarray, image_path: Path, 
                         slice_num: int, total_slices: int) -> None:
        """
        Save individual slice image with medical styling
        
        Args:
            slice_2d: 2D slice data
            image_path: Path to save the image
            slice_num: Current slice number
            total_slices: Total number of slices being processed
        """
        import matplotlib.pyplot as plt
        
        # Create figure with medical styling
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Display slice
        im = ax.imshow(slice_2d, cmap='gray', origin='lower')
        
        # Add medical information
        ax.set_title(f'🧠 Neuromorphic Analysis - Slice {slice_num}/{total_slices}', 
                    fontsize=14, fontweight='bold', color='white')
        ax.set_xlabel('X (pixels)', color='white')
        ax.set_ylabel('Y (pixels)', color='white')
        
        # Style the plot
        ax.tick_params(colors='white')
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
        
        # Add slice information
        ax.text(0.02, 0.98, f'Shape: {slice_2d.shape[0]}×{slice_2d.shape[1]}', 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', color='cyan',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Save with high quality
        plt.tight_layout()
        plt.savefig(image_path, dpi=150, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.close()
        
        self.logger.info(f"💾 Slice image saved: {image_path}")


class Enhanced3DVolumeAnalyzer:
    """
    Main class that integrates slice-by-slice processing with existing medical engine
    """
    
    def __init__(self, neuromorphic_engine, logger: Optional[logging.Logger] = None):
        self.neuromorphic_engine = neuromorphic_engine
        self.logger = logger or logging.getLogger(__name__)
        self.slice_processor = SliceBySliceProcessor(neuromorphic_engine, logger)
    
    def analyze_nifti_volume_enhanced_limited(self, nifti_path: str, 
                                            output_dir: str,
                                            slice_axis: int = 2,
                                            max_slices: int = 10) -> Dict:
        """
        Enhanced analysis of NIfTI volume with LIMITED slice-by-slice processing (optimized)
        
        Args:
            nifti_path: Path to .nii.gz file
            output_dir: Directory for outputs
            slice_axis: Axis for slicing (0=sagittal, 1=coronal, 2=axial)
            max_slices: Maximum number of slices to process
            
        Returns:
            Comprehensive analysis results (limited)
        """
        self.logger.info(f"🧠 Iniciando análisis volumétrico LIMITADO de {nifti_path}")
        self.logger.info(f"🎯 Procesando solo {max_slices} slices para optimización")
        
        # Load NIfTI volume
        nii_img = nib.load(nifti_path)
        volume_3d = nii_img.get_fdata()
        
        self.logger.info(f"📊 Volumen cargado: {volume_3d.shape}")
        self.logger.info(f"📐 Dimensiones: {volume_3d.shape[0]} x {volume_3d.shape[1]} x {volume_3d.shape[2]}")
        
        # Preprocess volume
        volume_3d = self._preprocess_volume(volume_3d)
        
        # Process LIMITED slice by slice
        slice_analysis = self.slice_processor.process_full_volume_slicewise_limited(
            volume_3d, output_dir, axis=slice_axis, max_slices=max_slices)
        
        # Generate video from limited slices
        slices = self.slice_processor.extract_all_slices(volume_3d, slice_axis)
        limited_slices = slices[:max_slices]  # Take only first max_slices
        video_path = self.slice_processor.generate_slice_video(
            limited_slices, output_dir, fps=5)  # Slower FPS for fewer frames
        
        # Create enhanced visualizations
        visualizations = self.slice_processor.create_enhanced_visualizations(
            volume_3d, slice_analysis, output_dir)
        
        # Compile comprehensive results
        enhanced_results = {
            'file_info': {
                'path': nifti_path,
                'shape': volume_3d.shape,
                'total_voxels': volume_3d.size,
                'slice_axis': slice_axis,
                'optimization_applied': True,
                'max_slices_processed': max_slices,
                'total_slices_available': volume_3d.shape[slice_axis]
            },
            'slice_by_slice_analysis': slice_analysis,
            'video_output': video_path,
            'visualizations': visualizations,
            'processing_summary': {
                'total_slices_processed': min(max_slices, slice_analysis['total_slices']),
                'successful_analyses': slice_analysis['volumetric_summary']['total_successful_slices'],
                'failed_analyses': slice_analysis['volumetric_summary']['total_failed_slices'],
                'optimization_note': f'Limited to {max_slices} slices out of {volume_3d.shape[slice_axis]} available'
            }
        }
        
        # Save comprehensive report
        import json
        report_path = Path(output_dir) / 'enhanced_volume_analysis_limited.json'
        with open(report_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_compatible = self._make_json_serializable(enhanced_results)
            json.dump(json_compatible, f, indent=2)
        
        self.logger.info(f"✅ Análisis volumétrico LIMITADO completo guardado en {report_path}")
        
        return enhanced_results
        """
        Enhanced analysis of NIfTI volume with slice-by-slice processing
        
        Args:
            nifti_path: Path to .nii.gz file
            output_dir: Directory for outputs
            slice_axis: Axis for slicing (0=sagittal, 1=coronal, 2=axial)
            
        Returns:
            Comprehensive analysis results
        """
        self.logger.info(f"🧠 Iniciando análisis volumétrico mejorado de {nifti_path}")
        
        # Load NIfTI volume
        nii_img = nib.load(nifti_path)
        volume_3d = nii_img.get_fdata()
        
        self.logger.info(f"📊 Volumen cargado: {volume_3d.shape}")
        self.logger.info(f"📐 Dimensiones: {volume_3d.shape[0]} x {volume_3d.shape[1]} x {volume_3d.shape[2]}")
        
        # Preprocess volume
        volume_3d = self._preprocess_volume(volume_3d)
        
        # Process slice by slice
        slice_analysis = self.slice_processor.process_full_volume_slicewise(
            volume_3d, axis=slice_axis)
        
        # Generate video
        slices = self.slice_processor.extract_all_slices(volume_3d, slice_axis)
        video_path = self.slice_processor.generate_slice_video(
            slices, output_dir)
        
        # Create enhanced visualizations
        visualizations = self.slice_processor.create_enhanced_visualizations(
            volume_3d, slice_analysis, output_dir)
        
        # Compile comprehensive results
        enhanced_results = {
            'file_info': {
                'path': nifti_path,
                'shape': volume_3d.shape,
                'total_voxels': volume_3d.size,
                'slice_axis': slice_axis
            },
            'slice_by_slice_analysis': slice_analysis,
            'video_output': video_path,
            'visualizations': visualizations,
            'processing_summary': {
                'total_slices_processed': slice_analysis['total_slices'],
                'successful_analyses': slice_analysis['volumetric_summary']['total_successful_slices'],
                'failed_analyses': slice_analysis['volumetric_summary']['total_failed_slices']
            }
        }
        
        # Save comprehensive report
        import json
        report_path = Path(output_dir) / 'enhanced_volume_analysis.json'
        with open(report_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_compatible = self._make_json_serializable(enhanced_results)
            json.dump(json_compatible, f, indent=2)
        
        self.logger.info(f"✅ Análisis volumétrico completo guardado en {report_path}")
        
        return enhanced_results
    
    def _preprocess_volume(self, volume_3d: np.ndarray) -> np.ndarray:
        """Preprocess 3D volume"""
        # Remove NaN values
        volume_3d = np.nan_to_num(volume_3d, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize to [0, 1]
        if volume_3d.max() > volume_3d.min():
            volume_3d = (volume_3d - volume_3d.min()) / (volume_3d.max() - volume_3d.min())
        
        return volume_3d
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
