"""
Medical Analysis Engine
Main orchestrator for complete medical image analysis
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from .medical_processor import MedicalImageProcessor
from .neuromorphic_engine import NeuromorphicMedicalEngine
from .diagnostic_analyzer import DiagnosticAnalyzer
from .visualizer import MedicalVisualizer
from .anatomical_analyzer import AdvancedAnatomicalAnalyzer
from .pdf_generator import ProfessionalMedicalPDFGenerator
from .enhanced_volume_processor import Enhanced3DVolumeAnalyzer

class MedicalAnalysisEngine:
    def __init__(self, output_dir: str = "analysis_results", 
                 log_level: int = logging.INFO):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.setup_logging(log_level)
        
        self.image_processor = MedicalImageProcessor(self.logger)
        self.neuromorphic_engine = NeuromorphicMedicalEngine(logger=self.logger)
        self.diagnostic_analyzer = DiagnosticAnalyzer(self.logger)
        self.visualizer = MedicalVisualizer(self.logger)
        self.anatomical_analyzer = AdvancedAnatomicalAnalyzer(self.logger)
        self.pdf_generator = ProfessionalMedicalPDFGenerator(self.logger)
        
        # 🚀 NEW: Enhanced volume processor for slice-by-slice analysis
        self.enhanced_volume_analyzer = Enhanced3DVolumeAnalyzer(
            self.neuromorphic_engine, self.logger)
        
        self.analysis_results = []
        
        self.logger.info("Medical Analysis Engine initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Neuromorphic core type: {self.neuromorphic_engine.core_type}")
        self.logger.info("🧠 Advanced Anatomical Analyzer loaded")
        self.logger.info("📄 Professional PDF Generator loaded")
        self.logger.info("🎬 Enhanced Volume Processor (slice-by-slice) loaded")
    
    def setup_logging(self, log_level: int):
        """Setup logging configuration"""
        
        log_file = self.output_dir / f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
    
    def analyze_single_image(self, image_path: str) -> Dict:
        """Complete analysis of a single medical image"""
        
        try:
            self.logger.info(f"Starting analysis of: {image_path}")
            
            image_info = self.image_processor.load_medical_image(image_path)
            
            if not self.image_processor.validate_medical_format(
                image_info['data'], image_info['header']):
                raise ValueError("Invalid medical image format")
            
            processed_data = self.image_processor.preprocess_image(image_info['data'])
            
            quality_metrics = self.image_processor.assess_image_quality(processed_data)
            self.logger.info(f"Image quality score: {quality_metrics['quality_score']:.2f}")
            
            segmentation = self.diagnostic_analyzer.segment_brain_tissues(processed_data)
            
            voxel_size = image_info['header'].get_zooms()[:3]
            morphometry = self.diagnostic_analyzer.calculate_morphometry(segmentation, voxel_size)
            
            # 🧠 NUEVO: Análisis anatómico estructural avanzado
            self.logger.info("🔬 Iniciando análisis anatómico estructural avanzado...")
            anatomical_analysis = self.anatomical_analyzer.analyze_anatomical_structures(
                processed_data, voxel_size)
            self.logger.info("✅ Análisis anatómico estructural completado")
            
            anomalies = self.diagnostic_analyzer.detect_anomalies(processed_data, segmentation)
            
            brain_age = self.diagnostic_analyzer.estimate_brain_age(morphometry, quality_metrics)
            
            normalcy = self.diagnostic_analyzer.calculate_normalcy_score(
                morphometry, quality_metrics, anomalies)
            
            statistical_features = self._extract_statistical_features(processed_data, segmentation)
            ai_analysis = self.neuromorphic_engine.medical_inference(statistical_features)
            
            metadata = self.image_processor.extract_metadata(image_info['header'])
            image_info.update(metadata)
            
            complete_results = self.diagnostic_analyzer.generate_diagnostic_report(
                image_info, quality_metrics, segmentation, morphometry,
                anomalies, brain_age, normalcy, ai_analysis
            )
            
            # 🧠 Integrar análisis anatómico estructural en el reporte
            complete_results['anatomical_analysis'] = anatomical_analysis
            complete_results['structural_findings'] = anatomical_analysis['clinical_interpretation']
            
            image_name = Path(image_path).stem.replace('.nii', '')
            visualization_path = self.output_dir / f"analysis_{image_name}.png"
            pdf_report_path = self.output_dir / f"medical_report_{image_name}.pdf"
            
            # Generar visualización PNG tradicional
            self.visualizer.create_comprehensive_analysis_figure(
                processed_data, complete_results, str(visualization_path)
            )
            
            # 📄 NUEVO: Generar reporte PDF profesional
            self.logger.info("📄 Generando reporte médico PDF profesional...")
            self.pdf_generator.generate_comprehensive_medical_report(
                complete_results, processed_data, str(pdf_report_path)
            )
            
            complete_results['visualization_path'] = str(visualization_path)
            complete_results['pdf_report_path'] = str(pdf_report_path)
            
            report_path = self.output_dir / f"report_{image_name}.json"
            with open(report_path, 'w') as f:
                json.dump(complete_results, f, indent=2, default=str)
            
            complete_results['report_path'] = str(report_path)
            
            self.analysis_results.append(complete_results)
            
            self.logger.info(f"Analysis completed for: {image_path}")
            self.logger.info(f"Report saved: {report_path}")
            self.logger.info(f"Visualization saved: {visualization_path}")
            self.logger.info(f"📄 PDF Report saved: {pdf_report_path}")
            
            return complete_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {image_path}: {e}")
            raise
    
    def analyze_multiple_images(self, image_paths: List[str]) -> List[Dict]:
        """Analyze multiple medical images"""
        
        self.logger.info(f"Starting batch analysis of {len(image_paths)} images")
        
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                self.logger.info(f"Processing image {i}/{len(image_paths)}: {Path(image_path).name}")
                result = self.analyze_single_image(image_path)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze {image_path}: {e}")
                continue
        
        if len(results) > 1:
            self._create_batch_summary(results)
        
        self.logger.info(f"Batch analysis completed. {len(results)} images processed successfully.")
        
        return results
    
    def _extract_statistical_features(self, image_data: np.ndarray, segmentation: Dict) -> np.ndarray:
        """Extract statistical features for neuromorphic processing"""
        
        brain_mask = segmentation['masks']['brain']
        brain_data = image_data[brain_mask]
        
        if len(brain_data) == 0:
            brain_data = image_data.flatten()
        
        features = np.array([
            np.mean(brain_data),
            np.std(brain_data),
            np.min(brain_data),
            np.max(brain_data),
            np.median(brain_data),
            np.percentile(brain_data, 25),
            np.percentile(brain_data, 75),
            np.percentile(brain_data, 90),
            np.percentile(brain_data, 95),
            len(brain_data) / image_data.size
        ])
        
        return features
    
    def _create_batch_summary(self, results: List[Dict]):
        """Create summary for batch analysis"""
        
        try:
            summary_visualization_path = self.output_dir / "batch_summary.png"
            self.visualizer.create_summary_comparison_figure(
                results, str(summary_visualization_path)
            )
            
            batch_summary = {
                'batch_analysis_summary': {
                    'total_images': len(results),
                    'analysis_date': datetime.now().isoformat(),
                    'summary_visualization': str(summary_visualization_path)
                },
                'individual_results': [
                    {
                        'image_name': result['image_information']['file_name'],
                        'quality_score': result['quality_assessment']['quality_score'],
                        'brain_volume': result['morphometric_analysis']['clinical_metrics']['total_brain_volume_ml'],
                        'normalcy_score': result['normalcy_assessment']['normalcy_score'],
                        'predicted_condition': result['ai_medical_analysis']['predicted_condition'],
                        'confidence': result['ai_medical_analysis']['confidence_score']
                    }
                    for result in results
                ],
                'aggregate_statistics': {
                    'average_quality_score': np.mean([r['quality_assessment']['quality_score'] for r in results]),
                    'average_brain_volume': np.mean([r['morphometric_analysis']['clinical_metrics']['total_brain_volume_ml'] for r in results]),
                    'average_normalcy_score': np.mean([r['normalcy_assessment']['normalcy_score'] for r in results]),
                    'condition_distribution': self._calculate_condition_distribution(results)
                }
            }
            
            summary_report_path = self.output_dir / "batch_summary_report.json"
            with open(summary_report_path, 'w') as f:
                json.dump(batch_summary, f, indent=2, default=str)
            
            self.logger.info(f"Batch summary created: {summary_report_path}")
            self.logger.info(f"Summary visualization: {summary_visualization_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create batch summary: {e}")
    
    def _calculate_condition_distribution(self, results: List[Dict]) -> Dict:
        """Calculate distribution of predicted conditions"""
        
        conditions = [result['ai_medical_analysis']['predicted_condition'] for result in results]
        condition_counts = {}
        
        for condition in conditions:
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        total = len(conditions)
        condition_percentages = {
            condition: (count / total) * 100 
            for condition, count in condition_counts.items()
        }
        
        return {
            'counts': condition_counts,
            'percentages': condition_percentages
        }
    
    def get_analysis_summary(self) -> Dict:
        """Get summary of all analyses performed"""
        
        if not self.analysis_results:
            return {'message': 'No analyses performed yet'}
        
        return {
            'total_analyses': len(self.analysis_results),
            'output_directory': str(self.output_dir),
            'log_file': str(self.log_file),
            'latest_analysis': self.analysis_results[-1]['image_information']['file_name'] if self.analysis_results else None,
            'average_quality_score': np.mean([r['quality_assessment']['quality_score'] for r in self.analysis_results]),
            'neuromorphic_core_type': self.neuromorphic_engine.core_type
        }
    
    def analyze_single_image_enhanced(self, image_path: str, 
                                    enable_slice_analysis: bool = True,
                                    slice_axis: int = 2) -> Dict:
        """
        Enhanced analysis with slice-by-slice processing
        
        Args:
            image_path: Path to medical image (.nii.gz)
            enable_slice_analysis: Whether to perform slice-by-slice analysis
            slice_axis: Axis for slicing (0=sagittal, 1=coronal, 2=axial)
            
        Returns:
            Enhanced analysis results including slice-by-slice data
        """
        try:
            self.logger.info(f"🧠 Starting ENHANCED analysis of: {image_path}")
            
            # Perform standard analysis first
            standard_results = self.analyze_single_image(image_path)
            
            enhanced_results = {
                'standard_analysis': standard_results,
                'enhanced_features': {},
                'analysis_type': 'enhanced_volumetric'
            }
            
            if enable_slice_analysis:
                self.logger.info("🎬 Iniciando análisis slice-por-slice")
                
                # Perform slice-by-slice analysis
                slice_results = self.enhanced_volume_analyzer.analyze_nifti_volume_enhanced(
                    image_path, str(self.output_dir), slice_axis)
                
                enhanced_results['enhanced_features'] = {
                    'slice_by_slice_analysis': slice_results,
                    'processing_mode': 'enhanced_volumetric',
                    'slice_axis_used': slice_axis,
                    'total_slices_analyzed': slice_results['processing_summary']['total_slices_processed'],
                    'video_generated': slice_results.get('video_output', None)
                }
                
                # Merge key findings into main results
                if 'slice_by_slice_analysis' in slice_results and 'volumetric_summary' in slice_results['slice_by_slice_analysis']:
                    vol_summary = slice_results['slice_by_slice_analysis']['volumetric_summary']
                    enhanced_results['volumetric_predictions'] = vol_summary.get('volume_predictions', {})
                    enhanced_results['slice_consistency'] = {
                        'successful_slices': vol_summary.get('total_successful_slices', 0),
                        'failed_slices': vol_summary.get('total_failed_slices', 0),
                        'success_rate': (vol_summary.get('total_successful_slices', 0) / 
                                       slice_results['processing_summary']['total_slices_processed']) 
                                       if slice_results['processing_summary']['total_slices_processed'] > 0 else 0
                    }
                
                self.logger.info("✅ Análisis slice-por-slice completado")
            else:
                self.logger.info("🔄 Análisis estándar (sin slice-by-slice)")
            
            # Save enhanced results
            image_name = Path(image_path).stem.replace('.nii', '')
            enhanced_report_path = self.output_dir / f"enhanced_report_{image_name}.json"
            
            with open(enhanced_report_path, 'w') as f:
                # Convert numpy types for JSON serialization
                json_compatible = self.enhanced_volume_analyzer._make_json_serializable(enhanced_results)
                json.dump(json_compatible, f, indent=2)
            
            enhanced_results['enhanced_report_path'] = str(enhanced_report_path)
            
            self.logger.info(f"✅ Enhanced analysis completed for: {image_path}")
            self.logger.info(f"📄 Enhanced report saved: {enhanced_report_path}")
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Enhanced analysis failed for {image_path}: {e}")
            raise

    def analyze_single_image_basic(self, image_path: str) -> Dict:
        """
        Basic analysis WITHOUT final diagnosis - for slice-by-slice workflow
        
        Args:
            image_path: Path to medical image (.nii.gz)
            
        Returns:
            Basic analysis results without PDF/visualization generation
        """
        try:
            self.logger.info(f"🔬 Starting BASIC analysis of: {image_path}")
            
            image_info = self.image_processor.load_medical_image(image_path)
            
            if not self.image_processor.validate_medical_format(
                image_info['data'], image_info['header']):
                raise ValueError("Invalid medical image format")
            
            processed_data = self.image_processor.preprocess_image(image_info['data'])
            
            quality_metrics = self.image_processor.assess_image_quality(processed_data)
            self.logger.info(f"Image quality score: {quality_metrics['quality_score']:.2f}")
            
            segmentation = self.diagnostic_analyzer.segment_brain_tissues(processed_data)
            
            voxel_size = image_info['header'].get_zooms()[:3]
            morphometry = self.diagnostic_analyzer.calculate_morphometry(segmentation, voxel_size)
            
            # 🧠 Análisis anatómico estructural avanzado
            self.logger.info("🔬 Iniciando análisis anatómico estructural avanzado")
            anatomical_analysis = self.anatomical_analyzer.analyze_anatomical_structures(
                processed_data, voxel_size)
            self.logger.info("✅ Análisis anatómico estructural completado")
            
            anomalies = self.diagnostic_analyzer.detect_anomalies(processed_data, segmentation)
            
            brain_age = self.diagnostic_analyzer.estimate_brain_age(morphometry, quality_metrics)
            
            normalcy = self.diagnostic_analyzer.calculate_normalcy_score(
                morphometry, quality_metrics, anomalies)
            
            statistical_features = self._extract_statistical_features(processed_data, segmentation)
            ai_analysis = self.neuromorphic_engine.medical_inference(statistical_features)
            
            metadata = self.image_processor.extract_metadata(image_info['header'])
            image_info.update(metadata)
            
            # Return basic results WITHOUT final diagnostic report generation
            basic_results = {
                'image_info': image_info,
                'quality_metrics': quality_metrics,
                'segmentation': segmentation,
                'morphometry': morphometry,
                'anatomical_analysis': anatomical_analysis,
                'anomalies': anomalies,
                'brain_age': brain_age,
                'normalcy': normalcy,
                'ai_analysis': ai_analysis,
                'statistical_features': statistical_features,
                'processed_data': processed_data
            }
            
            self.logger.info(f"✅ Basic analysis completed for: {image_path}")
            return basic_results
            
        except Exception as e:
            self.logger.error(f"Basic analysis failed for {image_path}: {e}")
            raise

    def analyze_single_image_enhanced_limited(self, image_path: str, 
                                            enable_slice_analysis: bool = True,
                                            slice_axis: int = 2,
                                            max_slices: int = 10) -> Dict:
        """
        Enhanced analysis with LIMITED slice-by-slice processing (optimized for testing)
        NUEVO FLUJO: Básico → Slice-by-slice → Diagnóstico integral
        
        Args:
            image_path: Path to medical image (.nii.gz)
            enable_slice_analysis: Whether to perform slice-by-slice analysis
            slice_axis: Axis for slicing (0=sagittal, 1=coronal, 2=axial)
            max_slices: Maximum number of slices to process (for testing/optimization)
            
        Returns:
            Enhanced analysis results including limited slice-by-slice data
        """
        try:
            self.logger.info(f"🧠 Starting OPTIMIZED ENHANCED analysis of: {image_path}")
            self.logger.info(f"🎯 Limiting analysis to {max_slices} slices for optimization")
            
            # PASO 1: Análisis básico SIN diagnóstico final
            basic_results = self.analyze_single_image_basic(image_path)
            
            enhanced_results = {
                'basic_analysis': basic_results,
                'enhanced_features': {},
                'analysis_type': 'enhanced_volumetric_limited',
                'optimization_settings': {
                    'max_slices': max_slices,
                    'slice_axis': slice_axis
                }
            }
            
            if enable_slice_analysis:
                self.logger.info(f"🎬 Iniciando análisis slice-por-slice LIMITADO ({max_slices} slices)")
                
                # PASO 2: Análisis slice-by-slice LIMITADO (con imágenes en img/)
                slice_results = self.enhanced_volume_analyzer.analyze_nifti_volume_enhanced_limited(
                    image_path, str(self.output_dir), slice_axis, max_slices)
                
                enhanced_results['enhanced_features'] = {
                    'slice_by_slice_analysis': slice_results,
                    'processing_mode': 'enhanced_volumetric_limited',
                    'slice_axis_used': slice_axis,
                    'total_slices_analyzed': slice_results['processing_summary']['total_slices_processed'],
                    'video_generated': slice_results.get('video_output', None),
                    'optimization_applied': True,
                    'max_slices_setting': max_slices
                }
                
                # PASO 3: DIAGNÓSTICO INTEGRAL combinando básico + slice-by-slice
                self.logger.info("🎯 Generando DIAGNÓSTICO INTEGRAL (básico + slice-by-slice)")
                integrated_diagnosis = self._generate_integrated_diagnosis(
                    basic_results, slice_results, image_path)
                
                enhanced_results['integrated_diagnosis'] = integrated_diagnosis
                
                # Merge key findings into main results
                if 'slice_by_slice_analysis' in slice_results and 'volumetric_summary' in slice_results['slice_by_slice_analysis']:
                    vol_summary = slice_results['slice_by_slice_analysis']['volumetric_summary']
                    enhanced_results['volumetric_predictions'] = vol_summary.get('volume_predictions', {})
                    enhanced_results['slice_consistency'] = {
                        'successful_slices': vol_summary.get('total_successful_slices', 0),
                        'failed_slices': vol_summary.get('total_failed_slices', 0),
                        'success_rate': (vol_summary.get('total_successful_slices', 0) / 
                                       slice_results['processing_summary']['total_slices_processed']) 
                                       if slice_results['processing_summary']['total_slices_processed'] > 0 else 0
                    }
                
                self.logger.info("✅ Análisis slice-por-slice LIMITADO completado")
            else:
                self.logger.info("🔄 Análisis estándar (sin slice-by-slice)")
                # Solo generar diagnóstico básico
                integrated_diagnosis = self._generate_basic_diagnosis(basic_results, image_path)
                enhanced_results['integrated_diagnosis'] = integrated_diagnosis
            
            # Save enhanced results
            image_name = Path(image_path).stem.replace('.nii', '')
            enhanced_report_path = self.output_dir / f"enhanced_limited_report_{image_name}.json"
            
            with open(enhanced_report_path, 'w') as f:
                # Convert numpy types for JSON serialization
                json_compatible = self.enhanced_volume_analyzer._make_json_serializable(enhanced_results)
                json.dump(json_compatible, f, indent=2, default=str)
            
            enhanced_results['enhanced_report_path'] = str(enhanced_report_path)
            
            self.logger.info(f"✅ Optimized enhanced analysis completed for: {image_path}")
            self.logger.info(f"📄 Optimized report saved: {enhanced_report_path}")
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Optimized enhanced analysis failed for {image_path}: {e}")
            raise

    def _generate_integrated_diagnosis(self, basic_results: Dict, 
                                     slice_results: Dict, 
                                     image_path: str) -> Dict:
        """
        Generate integrated diagnosis combining basic and slice-by-slice analysis
        
        Args:
            basic_results: Results from basic analysis
            slice_results: Results from slice-by-slice analysis
            image_path: Original image path
            
        Returns:
            Complete integrated diagnosis with PDF and visualizations
        """
        try:
            self.logger.info("🎯 Generating INTEGRATED diagnosis from basic + slice analysis")
            
            # Generate diagnostic report combining both analyses
            complete_results = self.diagnostic_analyzer.generate_diagnostic_report(
                basic_results['image_info'], 
                basic_results['quality_metrics'], 
                basic_results['segmentation'], 
                basic_results['morphometry'],
                basic_results['anomalies'], 
                basic_results['brain_age'], 
                basic_results['normalcy'], 
                basic_results['ai_analysis']
            )
            
            # Integrate anatomical analysis
            complete_results['anatomical_analysis'] = basic_results['anatomical_analysis']
            complete_results['structural_findings'] = basic_results['anatomical_analysis']['clinical_interpretation']
            
            # Integrate slice-by-slice findings
            complete_results['slice_by_slice_analysis'] = slice_results
            complete_results['enhanced_volumetric_findings'] = {
                'total_slices_analyzed': slice_results.get('processing_summary', {}).get('total_slices_processed', 0),
                'slice_consistency': slice_results.get('slice_by_slice_analysis', {}).get('volumetric_summary', {}),
                'video_generated': slice_results.get('video_output', None),
                'individual_slice_images': slice_results.get('slice_by_slice_analysis', {}).get('slice_analyses', [])
            }
            
            # Generate final visualizations and reports
            image_name = Path(image_path).stem.replace('.nii', '')
            visualization_path = self.output_dir / f"analysis_{image_name}.png"
            pdf_report_path = self.output_dir / f"medical_report_{image_name}.pdf"
            
            self.visualizer.create_comprehensive_analysis_figure(
                basic_results['processed_data'], 
                complete_results, 
                str(visualization_path)
            )
            
            self.logger.info(f"Enhanced comprehensive analysis figure saved: {visualization_path}")
            
            # Generate PDF report
            self.logger.info("📄 Generando reporte médico PDF profesional...")
            self.pdf_generator.generate_comprehensive_medical_report(
                complete_results, basic_results['processed_data'], str(pdf_report_path))
            
            complete_results['report_paths'] = {
                'json_report': str(self.output_dir / f"report_{image_name}.json"),
                'visualization': str(visualization_path),
                'pdf_report': str(pdf_report_path)
            }
            
            # Save JSON report
            report_path = self.output_dir / f"report_{image_name}.json"
            with open(report_path, 'w') as f:
                json.dump(complete_results, f, indent=2, default=str)
            
            self.logger.info(f"Analysis completed for: {image_path}")
            self.logger.info(f"Report saved: {report_path}")
            self.logger.info(f"Visualization saved: {visualization_path}")
            self.logger.info(f"📄 PDF Report saved: {pdf_report_path}")
            
            return complete_results
            
        except Exception as e:
            self.logger.error(f"Integrated diagnosis generation failed: {e}")
            raise

    def _generate_basic_diagnosis(self, basic_results: Dict, image_path: str) -> Dict:
        """Generate basic diagnosis when slice-by-slice analysis is disabled"""
        return self._generate_integrated_diagnosis(basic_results, {}, image_path)
