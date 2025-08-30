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
        
        self.analysis_results = []
        
        self.logger.info("Medical Analysis Engine initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Neuromorphic core type: {self.neuromorphic_engine.core_type}")
        self.logger.info("🧠 Advanced Anatomical Analyzer loaded")
        self.logger.info("📄 Professional PDF Generator loaded")
    
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
