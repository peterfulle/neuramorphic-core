#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Neuromorphic Medical AI Analysis System"
echo "Professional Brain Image Analysis + PDF Reports"
echo "=========================================="

if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found"
    exit 1
fi

echo "Checking Python environment..."

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    echo "Please install Python 3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python 3.8+ required, found $PYTHON_VERSION"
    exit 1
fi

echo "Python version: $PYTHON_VERSION (OK)"

echo "Installing/checking dependencies..."
pip3 install -r requirements.txt --quiet

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo "Dependencies installed successfully"

echo "Scanning for medical images..."

NII_FILES=(*.nii.gz)

if [ ! -e "${NII_FILES[0]}" ]; then
    echo "Error: No .nii.gz files found in current directory"
    echo "Please place medical image files (.nii.gz) in this directory"
    echo "Expected files: mni152.nii.gz, chris_t1.nii.gz, or other medical images"
    exit 1
fi

echo "Found medical image files:"
for file in "${NII_FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "  - $file ($size)"
    fi
done

OUTPUT_DIR="medical_analysis_$(date +%Y%m%d_%H%M%S)"
PDF_DIR="reportes_pdf_finales"

echo ""
echo "Starting comprehensive analysis..."
echo "Output directory: $OUTPUT_DIR"
echo "PDF Reports directory: $PDF_DIR"
echo "Timestamp: $(date)"
echo ""

# Paso 1: Análisis neuromorphic principal
echo "🧠 FASE 1: Análisis Neuromórfico Principal"
echo "==========================================="
python3 main.py --auto-scan --output-dir "$OUTPUT_DIR" --log-level INFO

MAIN_EXIT_CODE=$?

if [ $MAIN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ ERROR EN ANÁLISIS PRINCIPAL"
    echo "Exit code: $MAIN_EXIT_CODE"
    echo "Check the log files for error details"
    exit $MAIN_EXIT_CODE
fi

echo ""
echo "✅ Análisis principal completado exitosamente"

# Paso 2: Validación científica
echo ""
echo "🔬 FASE 2: Validación Científica"
echo "================================="
python3 scientific_validation.py --results-dir "$OUTPUT_DIR" --output-report "$OUTPUT_DIR/scientific_validation_report.json"

VALIDATION_EXIT_CODE=$?

if [ $VALIDATION_EXIT_CODE -ne 0 ]; then
    echo "⚠️  Validación científica falló, pero continuando con generación de PDFs..."
fi

# Paso 3: Generación de PDFs profesionales
echo ""
echo "📄 FASE 3: Generación de PDFs Profesionales"
echo "==========================================="

mkdir -p "$PDF_DIR"

# Script para generar PDFs ordenados desde los reportes
python3 -c "
import os
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

print('🔄 Iniciando generación de PDFs profesionales...')

output_dir = '$OUTPUT_DIR'
pdf_dir = '$PDF_DIR'

# Buscar todos los archivos JSON de reportes
json_files = list(Path(output_dir).glob('report_*.json'))
print(f'📊 Encontrados {len(json_files)} reportes para procesar')

for json_file in json_files:
    try:
        # Extraer nombre base del archivo
        base_name = json_file.stem.replace('report_', '')
        print(f'\\n📋 Procesando reporte: {base_name}')
        
        # Cargar datos del reporte
        with open(json_file, 'r') as f:
            report_data = json.load(f)
        
        # Buscar imagen de análisis correspondiente
        png_files = list(Path(output_dir).glob(f'analysis_{base_name}.png'))
        
        if png_files:
            analysis_image = png_files[0]
            print(f'  ✓ Imagen encontrada: {analysis_image.name}')
            
            # Crear PDF profesional
            pdf_path = Path(pdf_dir) / f'reporte_medico_profesional_{base_name}.pdf'
            
            # Generar PDF con portada y análisis
            fig = plt.figure(figsize=(8.5, 11))
            
            # Página 1: Portada con información del reporte
            ax1 = fig.add_subplot(111)
            ax1.text(0.5, 0.9, 'REPORTE MÉDICO NEUROMÓRFICO', 
                    horizontalalignment='center', fontsize=20, fontweight='bold')
            ax1.text(0.5, 0.8, f'Análisis de: {base_name}', 
                    horizontalalignment='center', fontsize=16)
            ax1.text(0.5, 0.7, f'Fecha: {datetime.now().strftime(\"%Y-%m-%d %H:%M\")}', 
                    horizontalalignment='center', fontsize=12)
            
            # Agregar información médica del reporte
            medical_analysis = report_data.get('medical_analysis', {})
            predicted_condition = medical_analysis.get('predicted_condition', 'N/A')
            confidence = medical_analysis.get('confidence_score', 0)
            
            ax1.text(0.1, 0.5, f'Condición Predicha: {predicted_condition}', fontsize=14, fontweight='bold')
            ax1.text(0.1, 0.45, f'Confianza: {confidence:.4f}', fontsize=12)
            
            # Probabilidades por condición
            probabilities = medical_analysis.get('condition_probabilities', {})
            y_pos = 0.35
            ax1.text(0.1, y_pos, 'Probabilidades por Condición:', fontsize=12, fontweight='bold')
            for condition, prob in probabilities.items():
                y_pos -= 0.03
                ax1.text(0.15, y_pos, f'• {condition}: {prob:.4f}', fontsize=10)
            
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
            
            # Guardar como imagen temporal
            temp_cover = Path(pdf_dir) / f'temp_cover_{base_name}.png'
            plt.savefig(temp_cover, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Crear PDF con portada y análisis
            images_for_pdf = []
            
            # Agregar portada
            cover_img = Image.open(temp_cover)
            if cover_img.mode != 'RGB':
                cover_img = cover_img.convert('RGB')
            images_for_pdf.append(cover_img)
            
            # Agregar imagen de análisis
            analysis_img = Image.open(analysis_image)
            if analysis_img.mode != 'RGB':
                analysis_img = analysis_img.convert('RGB')
            images_for_pdf.append(analysis_img)
            
            # Guardar PDF
            if images_for_pdf:
                images_for_pdf[0].save(pdf_path, save_all=True, append_images=images_for_pdf[1:], format='PDF')
                
                if pdf_path.exists():
                    size = pdf_path.stat().st_size
                    print(f'  ✅ PDF creado: {pdf_path.name} ({size:,} bytes)')
                else:
                    print(f'  ❌ Error creando PDF para {base_name}')
            
            # Limpiar archivos temporales
            if temp_cover.exists():
                temp_cover.unlink()
                
        else:
            print(f'  ⚠️  No se encontró imagen de análisis para {base_name}')
            
    except Exception as e:
        print(f'  ❌ Error procesando {base_name}: {e}')

print(f'\\n✅ Generación de PDFs completada')
print(f'📁 PDFs guardados en: {pdf_dir}/')
"

PDF_EXIT_CODE=$?

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "🎉 ANÁLISIS COMPLETO EXITOSO"
    echo "=========================================="
    echo "Resultados disponibles en:"
    echo "  📁 Análisis: $OUTPUT_DIR/"
    echo "  📄 PDFs: $PDF_DIR/"
    echo ""
    echo "Archivos generados:"
    echo ""
    echo "📊 REPORTES DE ANÁLISIS:"
    if [ -d "$OUTPUT_DIR" ]; then
        find "$OUTPUT_DIR" -type f -name "*.png" -o -name "*.json" -o -name "*.log" | while read file; do
            size=$(du -h "$file" | cut -f1)
            echo "  - $(basename "$file") ($size)"
        done
    fi
    echo ""
    echo "📄 REPORTES PDF PROFESIONALES:"
    if [ -d "$PDF_DIR" ]; then
        find "$PDF_DIR" -type f -name "*.pdf" | while read file; do
            size=$(du -h "$file" | cut -f1)
            echo "  - $(basename "$file") ($size)"
        done
    fi
    echo ""
    echo "🔬 Validación científica: $OUTPUT_DIR/scientific_validation_report.json"
    echo "📈 Logs de análisis: $OUTPUT_DIR/analysis_log_*.log"
    echo "🖼️  Imágenes de análisis: $OUTPUT_DIR/analysis_*.png"
    echo "📋 Reportes JSON: $OUTPUT_DIR/report_*.json"
    echo ""
    echo "✅ Sistema listo para revisión médica profesional"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ ANÁLISIS FALLÓ"
    echo "=========================================="
    echo "Exit code: $EXIT_CODE"
    echo "Check the log files for error details"
    echo ""
fi

exit $EXIT_CODE
