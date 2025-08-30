#!/usr/bin/env python3
"""
Test Script for Enhanced Slice-by-Slice Processing
Validates the new volumetric analysis capabilities
"""

import sys
import os
from pathlib import Path
import logging
import time

# Add the neurabrain directory to path
sys.path.append(str(Path(__file__).parent))

from core.medical_engine import MedicalAnalysisEngine

def test_enhanced_slice_processing():
    """Test the enhanced slice-by-slice processing"""
    
    print("🧠" + "="*70)
    print("     TEST: Enhanced Slice-by-Slice Processing")
    print("     Validando procesamiento volumétrico mejorado")
    print("="*73)
    
    # Setup test environment
    output_dir = f"test_enhanced_analysis_{int(time.time())}"
    engine = MedicalAnalysisEngine(output_dir=output_dir, log_level=logging.INFO)
    
    # Find available NIfTI files
    test_images = []
    for nii_file in Path('.').glob('*.nii.gz'):
        test_images.append(str(nii_file))
    
    if not test_images:
        print("❌ No se encontraron archivos .nii.gz en el directorio actual")
        print("📁 Archivos disponibles:")
        for file in Path('.').iterdir():
            if file.is_file():
                print(f"   - {file.name}")
        return False
    
    print(f"📊 Encontradas {len(test_images)} imágenes para analizar:")
    for img in test_images:
        print(f"   - {Path(img).name}")
    
    # Test different slice axes
    test_configurations = [
        {'axis': 2, 'name': 'Axial (default)'},
        {'axis': 1, 'name': 'Coronal'}, 
        {'axis': 0, 'name': 'Sagittal'}
    ]
    
    results_summary = []
    
    for config in test_configurations:
        print(f"\n🔬 Testando configuración: {config['name']} (eje {config['axis']})")
        print("-" * 50)
        
        for image_path in test_images[:1]:  # Test with first image only
            try:
                print(f"📷 Procesando: {Path(image_path).name}")
                
                start_time = time.time()
                
                # Run enhanced analysis
                results = engine.analyze_single_image_enhanced(
                    image_path=image_path,
                    enable_slice_analysis=True,
                    slice_axis=config['axis']
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Extract key metrics
                enhanced_features = results.get('enhanced_features', {})
                slice_analysis = enhanced_features.get('slice_by_slice_analysis', {})
                
                test_result = {
                    'configuration': config['name'],
                    'axis': config['axis'],
                    'image': Path(image_path).name,
                    'processing_time': processing_time,
                    'total_slices': slice_analysis.get('processing_summary', {}).get('total_slices_processed', 0),
                    'successful_slices': slice_analysis.get('processing_summary', {}).get('successful_analyses', 0),
                    'video_generated': enhanced_features.get('video_generated', None),
                    'success': True
                }
                
                # Print results
                print(f"   ✅ Análisis completado en {processing_time:.2f} segundos")
                print(f"   📊 Slices procesados: {test_result['total_slices']}")
                print(f"   ✅ Slices exitosos: {test_result['successful_slices']}")
                
                if test_result['video_generated']:
                    print(f"   🎬 Video generado: {Path(test_result['video_generated']).name}")
                
                # Check slice consistency
                slice_consistency = results.get('slice_consistency', {})
                if slice_consistency:
                    success_rate = slice_consistency.get('success_rate', 0)
                    print(f"   📈 Tasa de éxito: {success_rate:.2%}")
                
                results_summary.append(test_result)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results_summary.append({
                    'configuration': config['name'],
                    'axis': config['axis'],
                    'image': Path(image_path).name,
                    'error': str(e),
                    'success': False
                })
    
    # Print final summary
    print("\n" + "="*70)
    print("                    RESUMEN DE PRUEBAS")
    print("="*70)
    
    successful_tests = [r for r in results_summary if r.get('success', False)]
    failed_tests = [r for r in results_summary if not r.get('success', False)]
    
    print(f"✅ Pruebas exitosas: {len(successful_tests)}")
    print(f"❌ Pruebas fallidas: {len(failed_tests)}")
    
    if successful_tests:
        avg_time = sum(r['processing_time'] for r in successful_tests) / len(successful_tests)
        total_slices = sum(r['total_slices'] for r in successful_tests)
        
        print(f"⏱️  Tiempo promedio: {avg_time:.2f} segundos")
        print(f"📊 Total de slices procesados: {total_slices}")
        print(f"🎬 Videos generados: {sum(1 for r in successful_tests if r.get('video_generated'))}")
    
    print(f"\n📁 Resultados guardados en: {output_dir}/")
    
    # List generated files
    output_path = Path(output_dir)
    if output_path.exists():
        print("\n📋 Archivos generados:")
        for file in sorted(output_path.iterdir()):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   - {file.name} ({size_mb:.1f} MB)")
    
    return len(failed_tests) == 0

def test_slice_extraction_performance():
    """Test slice extraction performance with different volumes"""
    
    print("\n🧠" + "="*70)
    print("     PERFORMANCE TEST: Slice Extraction")
    print("     Midiendo rendimiento de extracción de slices")
    print("="*73)
    
    # Find NIfTI files
    test_images = list(Path('.').glob('*.nii.gz'))
    
    if not test_images:
        print("❌ No se encontraron archivos .nii.gz")
        return False
    
    from core.enhanced_volume_processor import SliceBySliceProcessor
    from core.neuromorphic_engine import NeuromorphicMedicalEngine
    import nibabel as nib
    import numpy as np
    
    # Initialize components
    neuromorphic_engine = NeuromorphicMedicalEngine()
    processor = SliceBySliceProcessor(neuromorphic_engine)
    
    for image_path in test_images[:1]:  # Test first image
        print(f"\n📷 Analizando: {image_path.name}")
        
        # Load volume
        nii_img = nib.load(str(image_path))
        volume_3d = nii_img.get_fdata()
        
        print(f"📐 Dimensiones: {volume_3d.shape}")
        print(f"💾 Tamaño: {volume_3d.size:,} voxeles")
        
        # Test slice extraction for each axis
        for axis, axis_name in enumerate(['Sagittal', 'Coronal', 'Axial']):
            print(f"\n🔍 Extrayendo slices {axis_name} (eje {axis})...")
            
            start_time = time.time()
            slices = processor.extract_all_slices(volume_3d, axis=axis)
            extraction_time = time.time() - start_time
            
            print(f"   ⏱️  Tiempo de extracción: {extraction_time:.3f} segundos")
            print(f"   📊 Slices extraídos: {len(slices)}")
            print(f"   📐 Dimensión de slice: {slices[0].shape}")
            print(f"   🚀 Velocidad: {len(slices)/extraction_time:.1f} slices/segundo")
    
    return True

def validate_neuromorphic_integration():
    """Validate that neuromorphic core processes slices correctly"""
    
    print("\n🧠" + "="*70)
    print("     INTEGRATION TEST: Neuromorphic Core")
    print("     Validando integración con core neuromórfico")
    print("="*73)
    
    try:
        from core.enhanced_volume_processor import SliceBySliceProcessor
        from core.neuromorphic_engine import NeuromorphicMedicalEngine
        import numpy as np
        
        # Initialize components
        neuromorphic_engine = NeuromorphicMedicalEngine()
        processor = SliceBySliceProcessor(neuromorphic_engine)
        
        # Create synthetic test slice
        test_slice = np.random.rand(256, 256).astype(np.float32)
        test_slice = (test_slice - test_slice.min()) / (test_slice.max() - test_slice.min())
        
        print("🔬 Procesando slice sintético de prueba...")
        
        # Test feature extraction
        features = processor._extract_slice_features(test_slice)
        print(f"   ✅ Features extraídos: {len(features)} características")
        print(f"   📊 Rango de features: [{features.min():.3f}, {features.max():.3f}]")
        
        # Test tissue analysis
        tissue_dist = processor._analyze_slice_tissues(test_slice)
        print(f"   🧬 Distribución de tejidos: {len(tissue_dist)} tipos")
        
        total_ratio = sum(tissue_dist.values())
        print(f"   ✅ Suma de ratios: {total_ratio:.3f} (esperado: ~1.0)")
        
        # Test neuromorphic processing
        slice_result = processor.process_slice_with_neuromorphic_core(test_slice, 0)
        
        if 'error' not in slice_result:
            print("   ✅ Procesamiento neuromórfico exitoso")
            print(f"   📊 Campos en resultado: {list(slice_result.keys())}")
            
            if 'neuromorphic_analysis' in slice_result:
                neuro_analysis = slice_result['neuromorphic_analysis']
                print(f"   🧠 Análisis neuromórfico: {type(neuro_analysis)}")
        else:
            print(f"   ❌ Error en procesamiento: {slice_result['error']}")
            return False
        
        print("✅ Integración neuromórfica validada correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en validación de integración: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🧠🧬🎬" + "="*67)
    print("    SUITE DE PRUEBAS: Enhanced Slice-by-Slice Processing")
    print("    Sistema de Análisis Médico Neuromórfico Mejorado")
    print("="*70)
    
    test_results = []
    
    # Test 1: Basic enhanced processing
    print("\n1️⃣ Ejecutando pruebas de procesamiento mejorado...")
    test_results.append(test_enhanced_slice_processing())
    
    # Test 2: Performance testing
    print("\n2️⃣ Ejecutando pruebas de rendimiento...")
    test_results.append(test_slice_extraction_performance())
    
    # Test 3: Integration validation
    print("\n3️⃣ Ejecutando validación de integración...")
    test_results.append(validate_neuromorphic_integration())
    
    # Final summary
    print("\n" + "🏁" + "="*69)
    print("                    RESUMEN FINAL")
    print("="*70)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"✅ Pruebas exitosas: {passed_tests}/{total_tests}")
    print(f"❌ Pruebas fallidas: {total_tests - passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("💫 El sistema de análisis slice-por-slice está funcionando correctamente")
        print("🚀 Listo para procesamiento volumétrico avanzado con core neuromórfico")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} prueba(s) fallaron")
        print("🔧 Revisar logs para detalles de errores")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
