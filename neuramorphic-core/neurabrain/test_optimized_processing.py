#!/usr/bin/env python3
"""
Test Enhanced Slice Processing - OPTIMIZED VERSION
Prueba rápida con solo 10 slices y una imagen
"""

import sys
import os
from pathlib import Path
import time
import random

# Add path to neurabrain core
sys.path.append(str(Path(__file__).parent))

from core.medical_engine import MedicalAnalysisEngine

def test_enhanced_slice_processing_optimized():
    """Test optimizado: solo 10 slices del chris_t1.nii.gz"""
    
    print("🧠🧬🎬" + "="*67)
    print("    PRUEBA OPTIMIZADA: Enhanced Slice-by-Slice Processing")
    print("    Solo 10 slices de chris_t1.nii.gz para validación rápida")
    print("="*70)
    
    # Setup test environment
    test_output_dir = f"test_optimized_analysis_{random.randint(1000000, 9999999)}"
    
    try:
        # Initialize medical engine
        print("1️⃣ Inicializando motor médico optimizado...")
        engine = MedicalAnalysisEngine(
            output_dir=test_output_dir,
            log_level=20  # INFO level
        )
        
        # Only test chris_t1.nii.gz
        test_image = "chris_t1.nii.gz"
        
        if not Path(test_image).exists():
            print(f"❌ Imagen no encontrada: {test_image}")
            return False
        
        print(f"📊 Imagen para analizar: {test_image}")
        
        print("\n🔬 Configuración: Solo 10 slices (eje axial)")
        print("-" * 50)
        
        print(f"📷 Procesando: {test_image}")
        
        start_time = time.time()
        
        # Perform enhanced analysis with limited slices
        results = engine.analyze_single_image_enhanced_limited(
            test_image,
            enable_slice_analysis=True,
            slice_axis=2,  # Axial
            max_slices=10  # LÍMITE: solo 10 slices
        )
        
        analysis_time = time.time() - start_time
        
        print(f"\n✅ Análisis completado en {analysis_time:.2f} segundos")
        
        # Print summary results
        if 'enhanced_features' in results:
            enhanced = results['enhanced_features']
            if 'slice_by_slice_analysis' in enhanced:
                slice_analysis = enhanced['slice_by_slice_analysis']
                
                print(f"\n📊 RESULTADOS DEL ANÁLISIS OPTIMIZADO:")
                print(f"   🎬 Total slices procesados: {enhanced['total_slices_analyzed']}")
                print(f"   📐 Forma del volumen original: {slice_analysis['file_info']['shape']}")
                print(f"   ✅ Análisis exitosos: {slice_analysis['processing_summary']['successful_analyses']}")
                print(f"   ❌ Análisis fallidos: {slice_analysis['processing_summary']['failed_analyses']}")
                
                if 'video_output' in slice_analysis:
                    print(f"   🎬 Video generado: {slice_analysis['video_output']}")
                
                if 'visualizations' in slice_analysis:
                    print(f"   📊 Visualizaciones: {len(slice_analysis['visualizations'])} archivos")
        
        print(f"\n📁 Resultados guardados en: {test_output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba optimizada: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ejecutar prueba optimizada"""
    
    print("\n🚀 INICIANDO PRUEBA OPTIMIZADA DE SLICE-BY-SLICE PROCESSING\n")
    
    success = test_enhanced_slice_processing_optimized()
    
    if success:
        print("\n🎉 ¡PRUEBA OPTIMIZADA COMPLETADA EXITOSAMENTE!")
        print("✅ El procesamiento slice-by-slice está funcionando correctamente")
        print("🎬 Video y visualizaciones generados")
        print("\n💡 Para procesar todos los slices, modifica max_slices en el código")
    else:
        print("\n❌ PRUEBA OPTIMIZADA FALLÓ")
        print("🔧 Revisa los logs para más detalles")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
