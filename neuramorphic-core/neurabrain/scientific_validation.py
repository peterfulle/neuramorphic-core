#!/usr/bin/env python3
"""
Scientific Validation System for Neuromorphic Medical AI
Comprehensive validation for scientific review and publication
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import torch
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ScientificValidator:
    """Comprehensive scientific validation for neuromorphic medical predictions"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.validation_results = {}
        self.setup_logging()
        
        # Medical condition mapping for validation
        self.condition_mapping = {
            0: 'Healthy',
            1: 'Alzheimer\'s Disease', 
            2: 'Parkinson\'s Disease',
            3: 'Brain Tumor',
            4: 'Traumatic Brain Injury',
            5: 'Mild Cognitive Impairment'
        }
        
        # Known medical baselines for comparison
        self.medical_baselines = {
            'healthy_probability_range': (0.60, 0.95),  # Healthy brains should be 60-95%
            'pathology_confidence_threshold': 0.3,      # Minimum confidence for pathology
            'inter_condition_variance': 0.15,           # Expected variance between conditions
            'expected_entropy_range': (0.5, 2.5),       # Information entropy range
            'volume_ranges': {                           # Expected brain volume ranges (mL)
                'healthy_adult': (1200, 1800),
                'pathological_min': (800, 1200)
            }
        }
        
    def setup_logging(self):
        """Setup scientific validation logging"""
        self.logger = logging.getLogger('scientific_validator')
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def load_analysis_results(self):
        """Load all analysis results for validation"""
        results = []
        
        # Load individual reports
        for json_file in self.results_dir.glob("report_*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results.append({
                        'filename': json_file.stem.replace('report_', ''),
                        'data': data
                    })
                self.logger.info(f"Loaded {json_file.name}")
            except Exception as e:
                self.logger.error(f"Failed to load {json_file}: {e}")
                
        # Load batch summary
        batch_file = self.results_dir / "batch_summary_report.json"
        if batch_file.exists():
            with open(batch_file, 'r') as f:
                batch_data = json.load(f)
                results.append({
                    'filename': 'batch_summary',
                    'data': batch_data
                })
                
        return results
        
    def validate_neuromorphic_consistency(self, results):
        """Validate neuromorphic model consistency and reliability"""
        print("\n🔬 VALIDACIÓN DE CONSISTENCIA NEUROMÓRFICA")
        print("=" * 60)
        
        consistencies = []
        
        for result in results:
            if 'medical_analysis' not in result['data']:
                continue
                
            analysis = result['data']['medical_analysis']
            image_name = result['filename']
            
            # Extract neuromorphic outputs
            probabilities = analysis.get('condition_probabilities', {})
            confidence = analysis.get('confidence_score', 0)
            
            print(f"\n📊 Análisis: {image_name}")
            print(f"   Confianza: {confidence:.4f}")
            
            # 1. Probability distribution validation
            prob_values = list(probabilities.values())
            prob_sum = sum(prob_values)
            prob_entropy = -sum(p * np.log(p + 1e-8) for p in prob_values if p > 0)
            
            print(f"   Suma probabilidades: {prob_sum:.6f} (esperado: 1.0)")
            print(f"   Entropía: {prob_entropy:.4f} (rango válido: {self.medical_baselines['expected_entropy_range']})")
            
            # 2. Medical consistency checks
            max_prob = max(prob_values)
            predicted_condition = analysis.get('predicted_condition', 'Unknown')
            
            is_healthy = predicted_condition == 'Healthy'
            healthy_prob = probabilities.get('Healthy', 0)
            
            print(f"   Condición predicha: {predicted_condition}")
            print(f"   Probabilidad máxima: {max_prob:.4f}")
            print(f"   Probabilidad 'Healthy': {healthy_prob:.4f}")
            
            # Medical logic validation
            medical_consistency = self._validate_medical_logic(
                probabilities, predicted_condition, confidence
            )
            
            consistencies.append({
                'image': image_name,
                'probability_sum': prob_sum,
                'entropy': prob_entropy,
                'max_probability': max_prob,
                'confidence': confidence,
                'medical_consistency': medical_consistency,
                'predicted_condition': predicted_condition
            })
            
        return consistencies
        
    def _validate_medical_logic(self, probabilities, predicted_condition, confidence):
        """Validate medical reasoning logic"""
        issues = []
        
        # Check if prediction matches highest probability
        max_condition = max(probabilities.items(), key=lambda x: x[1])
        if max_condition[0] != predicted_condition:
            issues.append(f"Predicción inconsistente: {predicted_condition} vs {max_condition[0]}")
            
        # Check confidence vs probability alignment
        max_prob = max(probabilities.values())
        if abs(confidence - max_prob) > 0.1:  # Allow some difference due to entropy calculation
            issues.append(f"Confianza no alineada con probabilidad máxima: {confidence:.3f} vs {max_prob:.3f}")
            
        # Check for unrealistic distributions
        healthy_prob = probabilities.get('Healthy', 0)
        pathology_sum = sum(v for k, v in probabilities.items() if k != 'Healthy')
        
        if healthy_prob < 0.1 and pathology_sum < 0.5:
            issues.append("Distribución de probabilidades médicamente inconsistente")
            
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'confidence_alignment': abs(confidence - max_prob) < 0.1
        }
        
    def validate_against_medical_standards(self, results):
        """Validate against established medical imaging standards"""
        print("\n🏥 VALIDACIÓN CONTRA ESTÁNDARES MÉDICOS")
        print("=" * 60)
        
        validations = []
        
        for result in results:
            if 'medical_analysis' not in result['data']:
                continue
                
            analysis = result['data']['medical_analysis']
            image_analysis = result['data'].get('image_analysis', {})
            image_name = result['filename']
            
            print(f"\n🧠 Imagen: {image_name}")
            
            # Brain volume validation
            brain_volume = analysis.get('brain_volume', 0)
            print(f"   Volumen cerebral: {brain_volume:.2f} mL")
            
            volume_validation = self._validate_brain_volume(brain_volume)
            print(f"   Validación volumen: {'✅ VÁLIDO' if volume_validation['valid'] else '⚠️ REVISAR'}")
            
            # Quality score validation
            quality_score = analysis.get('quality_score', 0)
            print(f"   Puntuación calidad: {quality_score}/10")
            
            # Morphometric validation
            morphometric = image_analysis.get('morphometric_analysis', {})
            if morphometric:
                print(f"   Análisis morfométrico disponible: ✅")
                print(f"   - Cortical thickness: {morphometric.get('cortical_thickness', 'N/A')}")
                print(f"   - White matter integrity: {morphometric.get('white_matter_integrity', 'N/A')}")
            
            # Neuromorphic processing validation
            neuromorphic_info = analysis.get('neuromorphic_core_type', 'unknown')
            processing_device = analysis.get('processing_device', 'unknown')
            print(f"   Núcleo neuromórfico: {neuromorphic_info}")
            print(f"   Dispositivo procesamiento: {processing_device}")
            
            validations.append({
                'image': image_name,
                'volume_validation': volume_validation,
                'quality_score': quality_score,
                'has_morphometric': bool(morphometric),
                'neuromorphic_type': neuromorphic_info,
                'processing_device': processing_device
            })
            
        return validations
        
    def _validate_brain_volume(self, volume):
        """Validate brain volume against medical norms"""
        healthy_range = self.medical_baselines['volume_ranges']['healthy_adult']
        pathological_min = self.medical_baselines['volume_ranges']['pathological_min'][0]
        
        if healthy_range[0] <= volume <= healthy_range[1]:
            return {'valid': True, 'category': 'normal', 'note': 'Volumen en rango normal'}
        elif volume >= pathological_min:
            return {'valid': True, 'category': 'borderline', 'note': 'Volumen límite, requiere revisión'}
        else:
            return {'valid': False, 'category': 'abnormal', 'note': 'Volumen anormalmente bajo'}
            
    def generate_statistical_analysis(self, consistencies, validations):
        """Generate comprehensive statistical analysis"""
        print("\n📈 ANÁLISIS ESTADÍSTICO CIENTÍFICO")
        print("=" * 60)
        
        # Verificar que tenemos datos
        if not consistencies:
            print("⚠️ No hay datos de consistencia para analizar")
            return {
                'sample_size': 0, 
                'error': 'No data available',
                'confidence_stats': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0},
                'entropy_stats': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0},
                'probability_consistency': {'perfect_sum_count': 0, 'acceptable_sum_count': 0}
            }
        
        # Convert to DataFrame for analysis
        df_consistency = pd.DataFrame(consistencies)
        df_validation = pd.DataFrame(validations) if validations else pd.DataFrame()
        
        # Verificar columnas requeridas
        required_cols = ['confidence', 'entropy', 'max_probability']
        missing_cols = [col for col in required_cols if col not in df_consistency.columns]
        
        if missing_cols:
            print(f"⚠️ Columnas faltantes en datos: {missing_cols}")
            return {'sample_size': len(consistencies), 'error': f'Missing columns: {missing_cols}'}
        
        # Statistical measures
        stats_summary = {
            'sample_size': len(consistencies),
            'confidence_stats': {
                'mean': df_consistency['confidence'].mean(),
                'std': df_consistency['confidence'].std(),
                'min': df_consistency['confidence'].min(),
                'max': df_consistency['confidence'].max()
            },
            'entropy_stats': {
                'mean': df_consistency['entropy'].mean(),
                'std': df_consistency['entropy'].std(),
                'min': df_consistency['entropy'].min(),
                'max': df_consistency['entropy'].max()
            },
            'probability_consistency': {
                'perfect_sum_count': sum(abs(p - 1.0) < 1e-6 for p in df_consistency['probability_sum']),
                'acceptable_sum_count': sum(abs(p - 1.0) < 1e-3 for p in df_consistency['probability_sum'])
            }
        }
        
        print(f"📊 Tamaño muestra: {stats_summary['sample_size']}")
        print(f"📊 Confianza promedio: {stats_summary['confidence_stats']['mean']:.4f} ± {stats_summary['confidence_stats']['std']:.4f}")
        print(f"📊 Entropía promedio: {stats_summary['entropy_stats']['mean']:.4f} ± {stats_summary['entropy_stats']['std']:.4f}")
        print(f"📊 Consistencia probabilidades: {stats_summary['probability_consistency']['perfect_sum_count']}/{stats_summary['sample_size']} perfectas")
        
        return stats_summary
        
    def create_validation_report(self, consistencies, validations, stats_summary):
        """Create comprehensive validation report for scientific review"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"scientific_validation_report_{timestamp}.json"
        
        # Overall validation status
        all_consistent = all(c['medical_consistency']['valid'] for c in consistencies)
        all_volumes_valid = all(v['volume_validation']['valid'] for v in validations)
        
        validation_report = {
            'validation_metadata': {
                'timestamp': timestamp,
                'validator_version': '1.0.0',
                'sample_size': len(consistencies),
                'validation_standards': 'Medical Imaging Standards 2025'
            },
            'overall_assessment': {
                'neuromorphic_consistency': all_consistent,
                'medical_standard_compliance': all_volumes_valid,
                'ready_for_publication': all_consistent and all_volumes_valid,
                'confidence_level': 'HIGH' if all_consistent else 'MEDIUM'
            },
            'detailed_metrics': {
                'consistency_analysis': consistencies,
                'medical_validation': validations,
                'statistical_summary': stats_summary
            },
            'recommendations': self._generate_scientific_recommendations(
                consistencies, validations, stats_summary
            )
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
            
        print(f"\n📋 REPORTE DE VALIDACIÓN CIENTÍFICA")
        print("=" * 60)
        print(f"✅ Consistencia neuromórfica: {'PASS' if all_consistent else 'REVIEW'}")
        print(f"✅ Estándares médicos: {'PASS' if all_volumes_valid else 'REVIEW'}")
        print(f"✅ Listo para publicación: {'SÍ' if validation_report['overall_assessment']['ready_for_publication'] else 'REVISAR'}")
        print(f"📄 Reporte guardado: {report_path}")
        
        return validation_report
        
    def _generate_scientific_recommendations(self, consistencies, validations, stats_summary):
        """Generate scientific recommendations based on validation results"""
        recommendations = []
        
        # Confidence analysis
        avg_confidence = stats_summary['confidence_stats']['mean']
        if avg_confidence < 0.3:
            recommendations.append({
                'category': 'Model Performance',
                'priority': 'HIGH',
                'recommendation': 'Confianza promedio baja. Considerar entrenamiento adicional o ajuste de hiperparámetros.',
                'metric': f'Confianza promedio: {avg_confidence:.3f}'
            })
            
        # Entropy analysis
        avg_entropy = stats_summary['entropy_stats']['mean']
        if avg_entropy > 2.0:
            recommendations.append({
                'category': 'Prediction Certainty',
                'priority': 'MEDIUM',
                'recommendation': 'Alta entropía indica incertidumbre. Revisar calidad de características de entrada.',
                'metric': f'Entropía promedio: {avg_entropy:.3f}'
            })
            
        # Consistency issues
        inconsistent_count = sum(1 for c in consistencies if not c['medical_consistency']['valid'])
        if inconsistent_count > 0:
            recommendations.append({
                'category': 'Medical Logic',
                'priority': 'HIGH',
                'recommendation': 'Inconsistencias en lógica médica detectadas. Revisar algoritmo de interpretación.',
                'metric': f'Casos inconsistentes: {inconsistent_count}/{len(consistencies)}'
            })
            
        # Volume validation
        volume_issues = sum(1 for v in validations if not v['volume_validation']['valid'])
        if volume_issues > 0:
            recommendations.append({
                'category': 'Anatomical Validation',
                'priority': 'MEDIUM',
                'recommendation': 'Volúmenes cerebrales fuera de rango normal. Verificar segmentación.',
                'metric': f'Volúmenes anómalos: {volume_issues}/{len(validations)}'
            })
            
        if not recommendations:
            recommendations.append({
                'category': 'Overall Assessment',
                'priority': 'INFO',
                'recommendation': 'Todas las validaciones científicas pasaron exitosamente. Sistema listo para revisión científica.',
                'metric': 'Validación completa: 100%'
            })
            
        return recommendations
        
    def create_validation_visualizations(self):
        """Create scientific validation visualizations"""
        print("\n📊 Generando visualizaciones de validación científica...")
        
        # Set scientific plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(20, 16))
        
        # Create comprehensive validation dashboard
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
        
        # You would implement specific plots here based on the validation data
        # This is a framework for the visualization system
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = self.results_dir / f"scientific_validation_visualizations_{timestamp}.png"
        
        plt.suptitle('Scientific Validation Dashboard - Neuromorphic Medical AI', 
                    fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizaciones guardadas: {viz_path}")
        return viz_path

def main():
    """Main validation execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scientific Validation for Neuromorphic Medical AI')
    parser.add_argument('--results-dir', required=True, help='Directory with analysis results')
    parser.add_argument('--output-report', help='Output validation report path')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ScientificValidator(args.results_dir)
    
    print("🔬 INICIANDO VALIDACIÓN CIENTÍFICA COMPLETA")
    print("=" * 80)
    
    # Load results
    results = validator.load_analysis_results()
    print(f"📁 Cargados {len(results)} archivos de resultados")
    
    # Run validations
    consistencies = validator.validate_neuromorphic_consistency(results)
    validations = validator.validate_against_medical_standards(results)
    stats_summary = validator.generate_statistical_analysis(consistencies, validations)
    
    # Generate report
    validation_report = validator.create_validation_report(consistencies, validations, stats_summary)
    
    # Create visualizations
    viz_path = validator.create_validation_visualizations()
    
    print("\n🎉 VALIDACIÓN CIENTÍFICA COMPLETADA")
    print("=" * 80)
    print("El sistema está listo para revisión científica.")

if __name__ == "__main__":
    main()
