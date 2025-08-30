#!/usr/bin/env python3
"""
Validador Científico Simplificado para Revisión por Científicos
Análisis específico para resultados neuromórficos médicos
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class MedicalValidatorSimple:
    """Validador médico simplificado para revisión científica"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.results = []
        self.validation_criteria = {
            'min_confidence': 0.10,      # Mínima confianza aceptable (10%)
            'max_confidence': 0.95,      # Máxima confianza realista (95%)
            'probability_tolerance': 1e-6, # Tolerancia para suma de probabilidades
            'expected_conditions': ['Healthy', 'Alzheimer', 'Parkinson', 'Brain Tumor', 'TBI', 'MCI']
        }
        
    def load_and_validate_results(self):
        """Cargar y validar resultados básicos"""
        print("🔍 CARGANDO RESULTADOS PARA VALIDACIÓN CIENTÍFICA")
        print("=" * 60)
        
        # Cargar archivos de reporte individual
        for json_file in self.results_dir.glob("report_*.json"):
            print(f"📄 Procesando: {json_file.name}")
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                # Extraer datos relevantes
                if 'ai_medical_analysis' in data:
                    analysis = data['ai_medical_analysis']
                    image_name = json_file.stem.replace('report_', '')
                    
                    result = {
                        'image_name': image_name,
                        'predicted_condition': analysis.get('predicted_condition', 'Unknown'),
                        'confidence_score': analysis.get('confidence_score', 0.0),
                        'severity_score': analysis.get('severity_score', 0.0),
                        'max_probability': analysis.get('max_probability', 0.0),
                        'condition_probabilities': analysis.get('condition_probabilities', {}),
                        'neuromorphic_core_type': analysis.get('neuromorphic_core_type', 'unknown'),
                        'processing_device': analysis.get('processing_device', 'unknown'),
                        'brain_volume': analysis.get('brain_volume', 0.0),
                        'quality_score': data.get('image_analysis', {}).get('quality_assessment', {}).get('overall_score', 0.0)
                    }
                    
                    self.results.append(result)
                    print(f"   ✅ Datos extraídos exitosamente")
                    print(f"   📊 Condición: {result['predicted_condition']}")
                    print(f"   📊 Confianza: {result['confidence_score']:.4f}")
                    print(f"   📊 Núcleo: {result['neuromorphic_core_type']}")
                    
            except Exception as e:
                print(f"   ❌ Error procesando {json_file.name}: {e}")
                
        print(f"\n📊 Total imágenes analizadas: {len(self.results)}")
        return self.results
        
    def validate_neuromorphic_outputs(self):
        """Validar salidas del núcleo neuromórfico"""
        print("\n🧠 VALIDACIÓN NÚCLEO NEUROMÓRFICO")
        print("=" * 60)
        
        validation_results = []
        
        for result in self.results:
            image_name = result['image_name']
            print(f"\n🔬 Validando: {image_name}")
            
            # 1. Validar que no hay resultados fallback
            is_fallback = (
                result['predicted_condition'] == 'Unknown' or 
                result['confidence_score'] == 0.0 or
                result['neuromorphic_core_type'] == 'fallback'
            )
            
            print(f"   ✅ Sin modo fallback: {'SÍ' if not is_fallback else 'NO - CRÍTICO'}")
            
            # 2. Validar confianza
            confidence = result['confidence_score']
            confidence_valid = self.validation_criteria['min_confidence'] <= confidence <= self.validation_criteria['max_confidence']
            print(f"   📊 Confianza válida: {'SÍ' if confidence_valid else 'NO'} ({confidence:.4f})")
            
            # 3. Validar probabilidades
            probabilities = result['condition_probabilities']
            if probabilities:
                prob_sum = sum(probabilities.values())
                prob_sum_valid = abs(prob_sum - 1.0) < self.validation_criteria['probability_tolerance']
                print(f"   🎲 Suma probabilidades: {'SÍ' if prob_sum_valid else 'NO'} ({prob_sum:.6f})")
                
                # Verificar que todas las condiciones están presentes
                expected_conditions = self.validation_criteria['expected_conditions']
                has_all_conditions = all(cond in probabilities for cond in expected_conditions)
                print(f"   📋 Todas las condiciones: {'SÍ' if has_all_conditions else 'NO'}")
                
                # Calcular entropía (medida de incertidumbre)
                entropy = -sum(p * np.log(p + 1e-8) for p in probabilities.values() if p > 0)
                print(f"   🔀 Entropía: {entropy:.4f} (menor = más certeza)")
            else:
                prob_sum_valid = False
                has_all_conditions = False
                entropy = float('inf')
                print(f"   ❌ No hay distribución de probabilidades")
            
            # 4. Validar consistencia médica
            predicted = result['predicted_condition']
            max_prob_condition = max(probabilities.items(), key=lambda x: x[1])[0] if probabilities else 'None'
            prediction_consistent = predicted == max_prob_condition
            print(f"   🏥 Predicción consistente: {'SÍ' if prediction_consistent else 'NO'}")
            print(f"       Predicho: {predicted}, Mayor prob: {max_prob_condition}")
            
            # 5. Validar núcleo neuromórfico real
            using_real_core = result['neuromorphic_core_type'] == 'full'
            print(f"   🧠 Núcleo real: {'SÍ' if using_real_core else 'NO - USAR REAL'}")
            
            validation_result = {
                'image_name': image_name,
                'no_fallback': not is_fallback,
                'confidence_valid': confidence_valid,
                'probabilities_valid': prob_sum_valid if probabilities else False,
                'all_conditions_present': has_all_conditions if probabilities else False,
                'prediction_consistent': prediction_consistent,
                'using_real_core': using_real_core,
                'entropy': entropy,
                'confidence_score': confidence,
                'overall_valid': all([
                    not is_fallback,
                    confidence_valid,
                    prob_sum_valid if probabilities else False,
                    has_all_conditions if probabilities else False,
                    prediction_consistent,
                    using_real_core
                ])
            }
            
            validation_results.append(validation_result)
            print(f"   🎯 VALIDACIÓN GENERAL: {'✅ PASA' if validation_result['overall_valid'] else '❌ FALLA'}")
            
        return validation_results
        
    def generate_scientific_summary(self, validation_results):
        """Generar resumen científico"""
        print("\n📊 RESUMEN CIENTÍFICO PARA REVISIÓN")
        print("=" * 60)
        
        total_samples = len(validation_results)
        valid_samples = sum(1 for v in validation_results if v['overall_valid'])
        
        print(f"📈 MÉTRICAS GENERALES:")
        print(f"   Total muestras: {total_samples}")
        print(f"   Muestras válidas: {valid_samples}/{total_samples} ({100*valid_samples/total_samples:.1f}%)")
        
        # Estadísticas de confianza
        confidences = [r['confidence_score'] for r in self.results]
        print(f"\n📊 ESTADÍSTICAS DE CONFIANZA:")
        print(f"   Promedio: {np.mean(confidences):.4f}")
        print(f"   Desviación estándar: {np.std(confidences):.4f}")
        print(f"   Mínimo: {np.min(confidences):.4f}")
        print(f"   Máximo: {np.max(confidences):.4f}")
        
        # Distribución de condiciones predichas
        conditions = [r['predicted_condition'] for r in self.results]
        condition_counts = {cond: conditions.count(cond) for cond in set(conditions)}
        print(f"\n🏥 DISTRIBUCIÓN DE CONDICIONES:")
        for condition, count in condition_counts.items():
            print(f"   {condition}: {count} casos ({100*count/total_samples:.1f}%)")
        
        # Validaciones específicas
        print(f"\n🔍 VALIDACIONES ESPECÍFICAS:")
        no_fallback_count = sum(1 for v in validation_results if v['no_fallback'])
        real_core_count = sum(1 for v in validation_results if v['using_real_core'])
        confidence_valid_count = sum(1 for v in validation_results if v['confidence_valid'])
        
        print(f"   Sin modo fallback: {no_fallback_count}/{total_samples} ({100*no_fallback_count/total_samples:.1f}%)")
        print(f"   Núcleo real usado: {real_core_count}/{total_samples} ({100*real_core_count/total_samples:.1f}%)")
        print(f"   Confianza válida: {confidence_valid_count}/{total_samples} ({100*confidence_valid_count/total_samples:.1f}%)")
        
        # Evaluación para publicación científica
        publication_ready = (
            valid_samples == total_samples and
            no_fallback_count == total_samples and
            real_core_count == total_samples and
            np.mean(confidences) > 0.15  # Confianza mínima razonable
        )
        
        print(f"\n🎓 EVALUACIÓN PARA PUBLICACIÓN CIENTÍFICA:")
        print(f"   Estado: {'✅ LISTO' if publication_ready else '⚠️ NECESITA REVISIÓN'}")
        
        if not publication_ready:
            print(f"\n🔧 RECOMENDACIONES:")
            if valid_samples < total_samples:
                print(f"   - Corregir {total_samples - valid_samples} muestras con validaciones fallidas")
            if no_fallback_count < total_samples:
                print(f"   - Eliminar modo fallback en {total_samples - no_fallback_count} casos")
            if real_core_count < total_samples:
                print(f"   - Usar núcleo neuromórfico real en {total_samples - real_core_count} casos")
            if np.mean(confidences) <= 0.15:
                print(f"   - Mejorar confianza promedio (actual: {np.mean(confidences):.3f}, objetivo: >0.15)")
        
        # Crear reporte para científicos
        scientific_report = {
            'validation_summary': {
                'total_samples': total_samples,
                'valid_samples': valid_samples,
                'validation_rate': valid_samples / total_samples,
                'publication_ready': publication_ready
            },
            'confidence_statistics': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            },
            'condition_distribution': condition_counts,
            'technical_validation': {
                'no_fallback_rate': no_fallback_count / total_samples,
                'real_core_rate': real_core_count / total_samples,
                'confidence_valid_rate': confidence_valid_count / total_samples
            },
            'detailed_results': validation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar reporte
        report_path = self.results_dir / f"scientific_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(scientific_report, f, indent=2, default=str)
        
        print(f"\n📄 Reporte científico guardado: {report_path}")
        
        return scientific_report

def main():
    """Función principal"""
    import sys
    
    if len(sys.argv) != 2:
        print("Uso: python3 scientific_validation_simple.py <directorio_resultados>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    print("🔬 VALIDACIÓN CIENTÍFICA PARA REVISIÓN POR EXPERTOS")
    print("=" * 80)
    print(f"📁 Directorio: {results_dir}")
    
    validator = MedicalValidatorSimple(results_dir)
    
    # Cargar resultados
    results = validator.load_and_validate_results()
    
    if not results:
        print("❌ No se encontraron resultados válidos para validar")
        sys.exit(1)
    
    # Ejecutar validaciones
    validation_results = validator.validate_neuromorphic_outputs()
    
    # Generar resumen científico
    scientific_report = validator.generate_scientific_summary(validation_results)
    
    print("\n🎉 VALIDACIÓN CIENTÍFICA COMPLETADA")
    print("=" * 80)
    print("Los resultados están listos para revisión por científicos.")

if __name__ == "__main__":
    main()
