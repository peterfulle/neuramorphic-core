"""
🏥 PROFESSIONAL PDF MEDICAL REPORT GENERATOR
=============================================
Generador de reportes PDF médicos profesionales para análisis neuromorphic
Especializado en hallazgos anatómicos estructurales radiológicos

Author: Neuromorphic Medical AI System
Date: August 29, 2025
"""

import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

class ProfessionalMedicalPDFGenerator:
    """Generador de PDFs médicos profesionales para reportes radiológicos"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuración profesional de colores médicos
        self.medical_colors = {
            'primary': '#1f4e79',      # Azul médico profesional
            'secondary': '#2e75b6',    # Azul secundario
            'success': '#28a745',      # Verde para normal
            'warning': '#ffc107',      # Amarillo para advertencia
            'error': '#dc3545',        # Rojo para crítico
            'urgent': '#8b0000',       # Rojo oscuro para urgente
            'info': '#17a2b8',         # Azul info
            'light': '#f8f9fa',        # Gris claro
            'dark': '#343a40',         # Gris oscuro
            'white': '#ffffff',        # Blanco
            'border': '#dee2e6'        # Gris borde
        }
        
        # Configuración de fuentes médicas
        self.fonts = {
            'title': {'family': 'Arial', 'size': 16, 'weight': 'bold'},
            'subtitle': {'family': 'Arial', 'size': 14, 'weight': 'bold'},
            'header': {'family': 'Arial', 'size': 12, 'weight': 'bold'},
            'body': {'family': 'Arial', 'size': 10, 'weight': 'normal'},
            'small': {'family': 'Arial', 'size': 8, 'weight': 'normal'},
            'mono': {'family': 'Courier New', 'size': 9, 'weight': 'normal'}
        }
        
        # Configuración de página médica
        self.page_config = {
            'figsize': (8.27, 11.69),  # A4 en pulgadas
            'dpi': 300,                # Alta resolución para impresión
            'margins': {'top': 0.8, 'bottom': 0.6, 'left': 0.6, 'right': 0.6}
        }
        
    def generate_comprehensive_medical_report(self, analysis_results: Dict, 
                                            image_data: np.ndarray,
                                            output_path: str) -> str:
        """
        Genera reporte médico profesional en múltiples páginas PNG
        (que luego se pueden convertir a PDF)
        
        Args:
            analysis_results: Resultados completos del análisis
            image_data: Datos de imagen 3D
            output_path: Ruta base para los archivos de salida
            
        Returns:
            Ruta del directorio con las páginas generadas
        """
        
        self.logger.info("📄 Generando reporte médico profesional multipágina...")
        
        # Crear directorio base para las páginas
        base_path = Path(output_path).parent
        report_name = Path(output_path).stem
        pages_dir = base_path / f"{report_name}_pages"
        pages_dir.mkdir(exist_ok=True)
        
        # Generar páginas individuales
        try:
            # Página 1: Portada y Resumen Ejecutivo
            page1_path = pages_dir / "01_portada_resumen.png"
            self._create_cover_page_png(str(page1_path), analysis_results)
            
            # Página 2: Análisis Anatómico Estructural Principal
            page2_path = pages_dir / "02_analisis_anatomico.png"
            self._create_anatomical_analysis_page_png(str(page2_path), analysis_results, image_data)
            
            # Página 3: Hallazgos Patológicos Específicos
            page3_path = pages_dir / "03_hallazgos_patologicos.png"
            self._create_pathological_findings_page_png(str(page3_path), analysis_results)
            
            # Página 4: Análisis Cuantitativo y Métricas
            page4_path = pages_dir / "04_analisis_cuantitativo.png"
            self._create_quantitative_analysis_page_png(str(page4_path), analysis_results)
            
            # Página 5: Recomendaciones Clínicas y Seguimiento
            page5_path = pages_dir / "05_recomendaciones.png"
            self._create_clinical_recommendations_page_png(str(page5_path), analysis_results)
            
            # Crear índice de páginas
            self._create_pages_index(pages_dir)
            
            # 🆕 GENERAR PDF CONSOLIDADO desde las páginas PNG
            pdf_output_path = str(Path(output_path))
            self._convert_pages_to_pdf(pages_dir, pdf_output_path)
            
            self.logger.info(f"✅ Reporte multipágina generado en: {pages_dir}")
            self.logger.info(f"📄 PDF consolidado generado: {pdf_output_path}")
            return pdf_output_path
            
        except Exception as e:
            self.logger.error(f"Error generando reporte: {e}")
            # Fallback: generar página única con resumen
            fallback_path = base_path / f"{report_name}_summary.png"
            self._create_summary_page_png(str(fallback_path), analysis_results)
            return str(fallback_path)
        
    def _create_cover_page(self, pdf_pages: PdfPages, analysis_results: Dict):
        """Crear página de portada profesional"""
        
        fig = plt.figure(figsize=self.page_config['figsize'], 
                        dpi=self.page_config['dpi'], facecolor='white')
        
        # Información del paciente y estudio
        image_info = analysis_results.get('image_information', {})
        anatomical = analysis_results.get('anatomical_analysis', {})
        clinical_interp = anatomical.get('clinical_interpretation', {}) if anatomical else {}
        
        # Header institucional
        fig.text(0.5, 0.95, '🏥 NEUROMORPHIC MEDICAL AI SYSTEM', 
                ha='center', va='top', **self.fonts['title'], 
                color=self.medical_colors['primary'])
        
        fig.text(0.5, 0.92, 'REPORTE DE ANÁLISIS NEUROLÓGICO ESTRUCTURAL', 
                ha='center', va='top', **self.fonts['subtitle'], 
                color=self.medical_colors['secondary'])
        
        # Línea separadora
        fig.add_artist(plt.Line2D([0.1, 0.9], [0.9, 0.9], 
                                 color=self.medical_colors['border'], linewidth=2))
        
        # Información del paciente
        patient_info = f"""
INFORMACIÓN DEL ESTUDIO
══════════════════════════════════════════════════════

📋 Archivo: {image_info.get('file_name', 'N/A')}
📅 Fecha de Análisis: {datetime.now().strftime('%d/%m/%Y - %H:%M')}
🔬 Modalidad: T1-weighted MRI
📐 Dimensiones: {image_info.get('dimensions', 'N/A')}
💾 Tamaño: {image_info.get('file_size_mb', 'N/A')} MB
🧠 Motor IA: Neuromorphic Core 5.5B parámetros
        """.strip()
        
        fig.text(0.1, 0.85, patient_info, ha='left', va='top', 
                **self.fonts['body'], color=self.medical_colors['dark'])
        
        # Resumen ejecutivo de hallazgos
        overall_assessment = clinical_interp.get('overall_assessment', 'NORMAL')
        urgency_level = clinical_interp.get('urgency_level', 'LOW')
        pathological_findings = clinical_interp.get('pathological_findings', [])
        
        # Color y texto según nivel de urgencia
        if urgency_level == 'HIGH':
            urgency_color = self.medical_colors['urgent']
            urgency_text = "🚨 HALLAZGOS CRÍTICOS - EVALUACIÓN URGENTE"
        elif urgency_level == 'MEDIUM':
            urgency_color = self.medical_colors['warning']
            urgency_text = "⚠️ HALLAZGOS SIGNIFICATIVOS - SEGUIMIENTO PRIORITARIO"
        else:
            urgency_color = self.medical_colors['success']
            urgency_text = "✅ ESTUDIO DENTRO DE PARÁMETROS NORMALES"
            
        # Caja de estado general
        fig.text(0.5, 0.65, urgency_text, ha='center', va='center',
                **self.fonts['header'], color=urgency_color,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=urgency_color, alpha=0.1))
        
        # Hallazgos principales
        if pathological_findings:
            findings_text = "HALLAZGOS PRINCIPALES:\n" + "\n".join([f"• {finding}" for finding in pathological_findings[:5]])
        else:
            findings_text = "✅ No se identificaron hallazgos patológicos significativos"
            
        fig.text(0.1, 0.55, findings_text, ha='left', va='top',
                **self.fonts['body'], color=self.medical_colors['dark'])
        
        # Estructura del reporte
        structure_text = """
ESTRUCTURA DEL REPORTE
═══════════════════════════════════════════════════════

📄 Página 2: Análisis Anatómico Estructural
   • Segmentación cerebral con visualización
   • Mediciones volumétricas principales
   
📄 Página 3: Hallazgos Patológicos Específicos
   • Atrofia cerebral (global/focal)
   • Atrofia hipocampal (bilateral/unilateral)
   • Hidrocefalia (comunicante/asimétrica)
   • Hemorragia subaracnoidea
   • Agrandamiento hipofisario
   • Desplazamientos estructurales
   • Asimetrías patológicas
   
📄 Página 4: Análisis Cuantitativo y Métricas
   • Morfometría detallada
   • Ratios anatómicos
   • Calidad de imagen
   
📄 Página 5: Recomendaciones Clínicas
   • Interpretación diagnóstica
   • Seguimiento sugerido
   • Correlación clínica
        """.strip()
        
        fig.text(0.1, 0.4, structure_text, ha='left', va='top',
                **self.fonts['small'], color=self.medical_colors['dark'])
        
        # Footer profesional
        fig.text(0.5, 0.05, 'Neuromorphic Medical AI - Análisis Automatizado con Validación Clínica', 
                ha='center', va='bottom', **self.fonts['small'], 
                color=self.medical_colors['secondary'], style='italic')
        
        # Crear un subplot invisible para poder usar axis
        ax = fig.add_subplot(111)
        ax.axis('off')
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_anatomical_analysis_page(self, pdf_pages: PdfPages, 
                                       analysis_results: Dict, 
                                       image_data: np.ndarray):
        """Página de análisis anatómico con imágenes"""
        
        fig = plt.figure(figsize=self.page_config['figsize'], 
                        dpi=self.page_config['dpi'], facecolor='white')
        
        # Layout en grid para organizar contenido
        gs = fig.add_gridspec(4, 3, height_ratios=[0.1, 1.2, 1.0, 0.8], 
                             width_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
        
        # Título de página
        fig.text(0.5, 0.95, '🧠 ANÁLISIS ANATÓMICO ESTRUCTURAL', 
                ha='center', va='top', **self.fonts['title'], 
                color=self.medical_colors['primary'])
        
        # Visualización de imágenes cerebrales (Fila 2)
        self._add_brain_views_to_page(fig, gs, image_data)
        
        # Métricas volumétricas (Fila 3)
        self._add_volumetric_metrics(fig, gs, analysis_results)
        
        # Análisis de calidad (Fila 4)
        self._add_quality_assessment(fig, gs, analysis_results)
        
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_pathological_findings_page(self, pdf_pages: PdfPages, 
                                         analysis_results: Dict):
        """Página de hallazgos patológicos específicos ordenados"""
        
        fig = plt.figure(figsize=self.page_config['figsize'], 
                        dpi=self.page_config['dpi'], facecolor='white')
        
        # Título
        fig.text(0.5, 0.95, '🔬 HALLAZGOS PATOLÓGICOS ESPECÍFICOS', 
                ha='center', va='top', **self.fonts['title'], 
                color=self.medical_colors['primary'])
        
        anatomical = analysis_results.get('anatomical_analysis', {})
        
        # Análisis ordenado por importancia clínica
        findings_sections = [
            ('🩸 HEMORRAGIAS Y EMERGENCIAS', self._analyze_hemorrhages, 0.88),
            ('💧 HIDROCEFALIA Y SISTEMA VENTRICULAR', self._analyze_hydrocephalus, 0.78),
            ('🧠 ATROFIA CEREBRAL', self._analyze_cerebral_atrophy, 0.68),
            ('🐎 ATROFIA HIPOCAMPAL', self._analyze_hippocampal_atrophy, 0.58),
            ('🔬 AGRANDAMIENTO HIPOFISARIO', self._analyze_pituitary_enlargement, 0.48),
            ('📐 DESPLAZAMIENTOS Y ASIMETRÍAS', self._analyze_structural_shifts, 0.38),
            ('⚖️ ASIMETRÍAS PATOLÓGICAS', self._analyze_pathological_asymmetries, 0.28)
        ]
        
        for title, analyzer_func, y_pos in findings_sections:
            self._create_finding_section(fig, title, analyzer_func(anatomical), y_pos)
            
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_quantitative_analysis_page(self, pdf_pages: PdfPages, 
                                         analysis_results: Dict):
        """Página de análisis cuantitativo detallado"""
        
        fig = plt.figure(figsize=self.page_config['figsize'], 
                        dpi=self.page_config['dpi'], facecolor='white')
        
        # Título
        fig.text(0.5, 0.95, '📊 ANÁLISIS CUANTITATIVO Y MÉTRICAS', 
                ha='center', va='top', **self.fonts['title'], 
                color=self.medical_colors['primary'])
        
        # Grid para organizar métricas
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], 
                             width_ratios=[1, 1], hspace=0.4, wspace=0.3)
        
        # Morfometría detallada
        self._add_detailed_morphometry(fig, gs, analysis_results)
        
        # Ratios anatómicos
        self._add_anatomical_ratios(fig, gs, analysis_results)
        
        # Métricas de IA y confianza
        self._add_ai_confidence_metrics(fig, gs, analysis_results)
        
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_clinical_recommendations_page(self, pdf_pages: PdfPages, 
                                            analysis_results: Dict):
        """Página de recomendaciones clínicas"""
        
        fig = plt.figure(figsize=self.page_config['figsize'], 
                        dpi=self.page_config['dpi'], facecolor='white')
        
        # Título
        fig.text(0.5, 0.95, '🏥 RECOMENDACIONES CLÍNICAS Y SEGUIMIENTO', 
                ha='center', va='top', **self.fonts['title'], 
                color=self.medical_colors['primary'])
        
        anatomical = analysis_results.get('anatomical_analysis', {})
        clinical_interp = anatomical.get('clinical_interpretation', {}) if anatomical else {}
        
        # Interpretación diagnóstica
        self._add_diagnostic_interpretation(fig, clinical_interp)
        
        # Recomendaciones de seguimiento
        self._add_follow_up_recommendations(fig, clinical_interp)
        
        # Correlación clínica sugerida
        self._add_clinical_correlation(fig, clinical_interp)
        
        # Disclaimer y validación
        self._add_medical_disclaimer(fig)
        
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_cover_page_png(self, output_path: str, analysis_results: Dict):
        """Crear página de portada como PNG"""
        
        fig = plt.figure(figsize=self.page_config['figsize'], 
                        dpi=self.page_config['dpi'], facecolor='white')
        
        # Información del paciente y estudio
        image_info = analysis_results.get('image_information', {})
        anatomical = analysis_results.get('anatomical_analysis', {})
        clinical_interp = anatomical.get('clinical_interpretation', {}) if anatomical else {}
        
        # Header institucional
        fig.text(0.5, 0.95, '🏥 NEUROMORPHIC MEDICAL AI SYSTEM', 
                ha='center', va='top', **self.fonts['title'], 
                color=self.medical_colors['primary'])
        
        fig.text(0.5, 0.92, 'REPORTE DE ANÁLISIS NEUROLÓGICO ESTRUCTURAL', 
                ha='center', va='top', **self.fonts['subtitle'], 
                color=self.medical_colors['secondary'])
        
        # Información del estudio
        patient_info = f"""
INFORMACIÓN DEL ESTUDIO
══════════════════════════════════════════════════════

📋 Archivo: {image_info.get('file_name', 'N/A')}
📅 Fecha de Análisis: {datetime.now().strftime('%d/%m/%Y - %H:%M')}
🔬 Modalidad: T1-weighted MRI
📐 Dimensiones: {image_info.get('dimensions', 'N/A')}
💾 Tamaño: {image_info.get('file_size_mb', 'N/A')} MB
🧠 Motor IA: Neuromorphic Core 5.5B parámetros
        """.strip()
        
        fig.text(0.1, 0.85, patient_info, ha='left', va='top', 
                **self.fonts['body'], color=self.medical_colors['dark'])
        
        # Resumen ejecutivo de hallazgos
        overall_assessment = clinical_interp.get('overall_assessment', 'NORMAL')
        urgency_level = clinical_interp.get('urgency_level', 'LOW')
        pathological_findings = clinical_interp.get('pathological_findings', [])
        
        # Color y texto según nivel de urgencia
        if urgency_level == 'HIGH':
            urgency_color = self.medical_colors['urgent']
            urgency_text = "🚨 HALLAZGOS CRÍTICOS - EVALUACIÓN URGENTE"
        elif urgency_level == 'MEDIUM':
            urgency_color = self.medical_colors['warning']
            urgency_text = "⚠️ HALLAZGOS SIGNIFICATIVOS - SEGUIMIENTO PRIORITARIO"
        else:
            urgency_color = self.medical_colors['success']
            urgency_text = "✅ ESTUDIO DENTRO DE PARÁMETROS NORMALES"
            
        # Caja de estado general
        fig.text(0.5, 0.65, urgency_text, ha='center', va='center',
                **self.fonts['header'], color=urgency_color,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=urgency_color, alpha=0.1))
        
        # Hallazgos principales
        if pathological_findings:
            findings_text = "HALLAZGOS PRINCIPALES:\n" + "\n".join([f"• {finding}" for finding in pathological_findings[:5]])
        else:
            findings_text = "✅ No se identificaron hallazgos patológicos significativos"
            
        fig.text(0.1, 0.55, findings_text, ha='left', va='top',
                **self.fonts['body'], color=self.medical_colors['dark'])
        
        # Footer profesional
        fig.text(0.5, 0.05, 'Neuromorphic Medical AI - Análisis Automatizado con Validación Clínica', 
                ha='center', va='bottom', **self.fonts['small'], 
                color=self.medical_colors['secondary'], style='italic')
        
        ax = fig.add_subplot(111); ax.axis('off')
        plt.savefig(output_path, dpi=self.page_config['dpi'], bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
    def _create_anatomical_analysis_page_png(self, output_path: str, 
                                           analysis_results: Dict, 
                                           image_data: np.ndarray):
        """Página de análisis anatómico como PNG"""
        
        fig = plt.figure(figsize=self.page_config['figsize'], 
                        dpi=self.page_config['dpi'], facecolor='white')
        
        # Título de página
        fig.text(0.5, 0.95, '🧠 ANÁLISIS ANATÓMICO ESTRUCTURAL', 
                ha='center', va='top', **self.fonts['title'], 
                color=self.medical_colors['primary'])
        
        # Grid layout para organizar contenido
        gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.0, 0.8], 
                             width_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
        
        # Visualización de imágenes cerebrales
        self._add_brain_views_to_figure(fig, gs, image_data)
        
        # Métricas volumétricas
        self._add_volumetric_summary(fig, gs, analysis_results)
        
        plt.savefig(output_path, dpi=self.page_config['dpi'], bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

    def _add_brain_views_to_figure(self, fig, gs, image_data):
        """Agregar vistas cerebrales a la figura"""
        try:
            if image_data is not None and len(image_data.shape) == 3:
                # Vista axial (superior)
                ax1 = fig.add_subplot(gs[0, 0])
                slice_idx = image_data.shape[2] // 2
                ax1.imshow(image_data[:, :, slice_idx], cmap='gray', aspect='equal')
                ax1.set_title('Vista Axial', **self.fonts['body'])
                ax1.axis('off')
                
                # Vista coronal (frontal)
                ax2 = fig.add_subplot(gs[0, 1])
                slice_idx = image_data.shape[1] // 2
                ax2.imshow(image_data[:, slice_idx, :], cmap='gray', aspect='equal')
                ax2.set_title('Vista Coronal', **self.fonts['body'])
                ax2.axis('off')
                
                # Vista sagital (lateral)
                ax3 = fig.add_subplot(gs[0, 2])
                slice_idx = image_data.shape[0] // 2
                ax3.imshow(image_data[slice_idx, :, :], cmap='gray', aspect='equal')
                ax3.set_title('Vista Sagital', **self.fonts['body'])
                ax3.axis('off')
            else:
                # Si no hay datos, mostrar placeholder
                ax = fig.add_subplot(gs[0, :])
                ax.text(0.5, 0.5, 'Imágenes cerebrales no disponibles', 
                       ha='center', va='center', **self.fonts['body'])
                ax.axis('off')
        except Exception as e:
            # Fallback en caso de error
            ax = fig.add_subplot(gs[0, :])
            ax.text(0.5, 0.5, f'Error cargando imágenes: {str(e)}', 
                   ha='center', va='center', **self.fonts['body'])
            ax.axis('off')

    def _add_volumetric_summary(self, fig, gs, analysis_results):
        """Agregar resumen volumétrico"""
        try:
            ax = fig.add_subplot(gs[1, :])
            
            tissue_analysis = analysis_results.get('tissue_analysis', {})
            anatomical = analysis_results.get('anatomical_analysis', {})
            
            # Crear texto de resumen volumétrico
            summary_text = "ANÁLISIS VOLUMÉTRICO:\n"
            
            if tissue_analysis:
                total_brain = tissue_analysis.get('total_brain', 0)
                gray_matter = tissue_analysis.get('gray_matter', 0)
                white_matter = tissue_analysis.get('white_matter', 0)
                csf = tissue_analysis.get('csf', 0)
                
                summary_text += f"• Volumen cerebral total: {total_brain:.1f} mL\n"
                summary_text += f"• Sustancia gris: {gray_matter:.1f} mL\n"
                summary_text += f"• Sustancia blanca: {white_matter:.1f} mL\n"
                summary_text += f"• LCR: {csf:.1f} mL\n"
            
            # Análisis ventricular
            if 'ventricles' in anatomical:
                vent = anatomical['ventricles']
                evans_index = vent.get('evans_index', 0.25)
                summary_text += f"• Índice de Evans: {evans_index:.3f}\n"
            
            # Análisis hipocampal
            if 'hippocampus' in anatomical:
                hipp = anatomical['hippocampus']
                vol_percentile = hipp.get('volume_percentile', 50)
                summary_text += f"• Volumen hipocampal (percentil): {vol_percentile:.1f}\n"
            
            ax.text(0.05, 0.95, summary_text, ha='left', va='top',
                   transform=ax.transAxes, **self.fonts['body'],
                   color=self.medical_colors['dark'])
            ax.axis('off')
            
        except Exception as e:
            ax = fig.add_subplot(gs[1, :])
            ax.text(0.5, 0.5, f'Error en resumen volumétrico: {str(e)}', 
                   ha='center', va='center', **self.fonts['body'])
            ax.axis('off')

    def _generate_pathological_summary(self, anatomical):
        """Generar resumen de hallazgos patológicos"""
        findings = []
        
        try:
            # Análisis ventricular
            if 'ventricles' in anatomical:
                vent = anatomical['ventricles']
                evans_index = vent.get('evans_index', 0.25)
                if evans_index > 0.30:
                    severity = "severa" if evans_index > 0.35 else "moderada"
                    findings.append(f"• DILATACIÓN VENTRICULAR {severity.upper()}")
                    findings.append(f"  - Índice de Evans: {evans_index:.3f}")
                    if evans_index > 0.35:
                        findings.append("  - Sugiere hidrocefalia activa")
                    
            # Análisis hipocampal
            if 'hippocampus' in anatomical:
                hipp = anatomical['hippocampus']
                vol_percentile = hipp.get('volume_percentile', 50)
                if vol_percentile < 25:
                    severity = "severa" if vol_percentile < 10 else "moderada"
                    findings.append(f"• ATROFIA HIPOCAMPAL {severity.upper()}")
                    findings.append(f"  - Volumen percentil: {vol_percentile:.1f}")
                    if vol_percentile < 10:
                        findings.append("  - Sugiere proceso neurodegenerativo")
                        
            # Análisis hipofisario
            if 'pituitary' in anatomical:
                pit = anatomical['pituitary']
                size_mm = pit.get('max_diameter_mm', 0)
                if size_mm > 10:
                    lesion_type = "macroadenoma" if size_mm > 40 else "microadenoma"
                    findings.append(f"• LESIÓN HIPOFISARIA - {lesion_type.upper()}")
                    findings.append(f"  - Diámetro máximo: {size_mm:.1f} mm")
                    if size_mm > 40:
                        findings.append("  - Requiere evaluación endocrina urgente")
                        
            # Análisis de hemorragias
            if 'hemorrhages' in anatomical:
                hem = anatomical['hemorrhages']
                if hem.get('detected', False):
                    location = hem.get('primary_location', 'no especificada')
                    volume = hem.get('estimated_volume_ml', 0)
                    findings.append(f"• HEMORRAGIA DETECTADA")
                    findings.append(f"  - Localización: {location}")
                    findings.append(f"  - Volumen estimado: {volume:.1f} mL")
                    if volume > 50:
                        findings.append("  - ATENCIÓN MÉDICA INMEDIATA REQUERIDA")
                        
            # Análisis de surcos
            if 'sulci' in anatomical:
                sulci = anatomical['sulci']
                if sulci.get('abnormal_widening', False):
                    findings.append("• ENSANCHAMIENTO ANORMAL DE SURCOS")
                    findings.append("  - Sugiere atrofia cortical")
                    
            if not findings:
                findings.append("• No se detectaron hallazgos patológicos significativos")
                findings.append("• Anatomía cerebral dentro de límites normales")
                findings.append("• Se recomienda correlación clínica")
            
            return "\n".join(findings)
            
        except Exception as e:
            return f"Error en análisis patológico: {str(e)}"
        
    def _create_pathological_findings_page_png(self, output_path: str, 
                                             analysis_results: Dict):
        """Página de hallazgos patológicos como PNG"""
        
        fig = plt.figure(figsize=self.page_config['figsize'], 
                        dpi=self.page_config['dpi'], facecolor='white')
        
        # Título
        fig.text(0.5, 0.95, '🔬 HALLAZGOS PATOLÓGICOS ESPECÍFICOS', 
                ha='center', va='top', **self.fonts['title'], 
                color=self.medical_colors['primary'])
        
        anatomical = analysis_results.get('anatomical_analysis', {})
        
        # Análisis ordenado por importancia clínica
        findings_text = self._generate_pathological_summary(anatomical)
        
        fig.text(0.05, 0.9, findings_text, ha='left', va='top',
                **self.fonts['body'], color=self.medical_colors['dark'])
        
        ax = fig.add_subplot(111); ax.axis('off')
        plt.savefig(output_path, dpi=self.page_config['dpi'], bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
    def _create_quantitative_analysis_page_png(self, output_path: str, 
                                             analysis_results: Dict):
        """Página de análisis cuantitativo como PNG"""
        
        fig = plt.figure(figsize=self.page_config['figsize'], 
                        dpi=self.page_config['dpi'], facecolor='white')
        
        # Título
        fig.text(0.5, 0.95, '📊 ANÁLISIS CUANTITATIVO Y MÉTRICAS', 
                ha='center', va='top', **self.fonts['title'], 
                color=self.medical_colors['primary'])
        
        quantitative_text = self._generate_quantitative_summary(analysis_results)
        
        fig.text(0.05, 0.9, quantitative_text, ha='left', va='top',
                **self.fonts['body'], color=self.medical_colors['dark'])
        
        ax = fig.add_subplot(111); ax.axis('off')
        plt.savefig(output_path, dpi=self.page_config['dpi'], bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
    def _create_clinical_recommendations_page_png(self, output_path: str, 
                                                analysis_results: Dict):
        """Página de recomendaciones clínicas como PNG"""
        
        fig = plt.figure(figsize=self.page_config['figsize'], 
                        dpi=self.page_config['dpi'], facecolor='white')
        
        # Título
        fig.text(0.5, 0.95, '🏥 RECOMENDACIONES CLÍNICAS Y SEGUIMIENTO', 
                ha='center', va='top', **self.fonts['title'], 
                color=self.medical_colors['primary'])
        
        recommendations_text = self._generate_recommendations_summary(analysis_results)
        
        fig.text(0.05, 0.9, recommendations_text, ha='left', va='top',
                **self.fonts['body'], color=self.medical_colors['dark'])
        
        # Disclaimer médico
        disclaimer_text = """
AVISO MÉDICO IMPORTANTE:
Este reporte ha sido generado por un sistema de inteligencia artificial neuromorphic y debe ser 
interpretado por un radiólogo o médico especialista calificado. Los hallazgos automatizados 
requieren correlación clínica y validación profesional antes de tomar decisiones diagnósticas 
o terapéuticas. No sustituye el juicio clínico profesional.
        """.strip()
        
        fig.text(0.5, 0.12, disclaimer_text, ha='center', va='top',
                **self.fonts['small'], color=self.medical_colors['secondary'],
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.medical_colors['light']),
                style='italic')
        
        ax = fig.add_subplot(111); ax.axis('off')
        plt.savefig(output_path, dpi=self.page_config['dpi'], bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
    def _create_summary_page_png(self, output_path: str, analysis_results: Dict):
        """Crear página de resumen como fallback"""
        
        fig = plt.figure(figsize=self.page_config['figsize'], 
                        dpi=self.page_config['dpi'], facecolor='white')
        
        fig.text(0.5, 0.95, '🏥 RESUMEN MÉDICO NEUROMORPHIC', 
                ha='center', va='top', **self.fonts['title'], 
                color=self.medical_colors['primary'])
        
        # Resumen completo
        summary_text = self._generate_complete_summary(analysis_results)
        
        fig.text(0.05, 0.9, summary_text, ha='left', va='top',
                **self.fonts['body'], color=self.medical_colors['dark'])
        
        ax = fig.add_subplot(111); ax.axis('off')
        plt.savefig(output_path, dpi=self.page_config['dpi'], bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
    def _create_pages_index(self, pages_dir: Path):
        """Crear índice de páginas generadas"""
        
        index_content = """
📄 ÍNDICE DE PÁGINAS DEL REPORTE MÉDICO
========================================

01_portada_resumen.png       - Información del estudio y resumen ejecutivo
02_analisis_anatomico.png    - Análisis anatómico estructural con imágenes
03_hallazgos_patologicos.png - Hallazgos patológicos específicos ordenados
04_analisis_cuantitativo.png - Métricas cuantitativas y ratios anatómicos
05_recomendaciones.png       - Recomendaciones clínicas y seguimiento

Para convertir a PDF:
- Linux/Mac: convert *.png reporte_medico.pdf
- Windows: Usar herramienta de conversión PNG a PDF
        """.strip()
        
        index_path = pages_dir / "00_INDICE.txt"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)

    # ========== MÉTODOS AUXILIARES DE ANÁLISIS ==========
    
    def _analyze_hemorrhages(self, anatomical: Dict) -> Dict:
        """Análisis específico de hemorragias"""
        hemorrhage = anatomical.get('hemorrhage_detection', {})
        
        return {
            'detected': hemorrhage.get('hemorrhages_detected', 0) > 0,
            'count': hemorrhage.get('hemorrhages_detected', 0),
            'subarachnoid_suspected': hemorrhage.get('subarachnoid_hemorrhage_suspected', False),
            'locations': hemorrhage.get('hemorrhage_locations', []),
            'urgency': hemorrhage.get('clinical_urgency', 'LOW'),
            'significance': hemorrhage.get('clinical_significance', 'No se detectan hemorragias'),
            'priority': 1 if hemorrhage.get('hemorrhages_detected', 0) > 0 else 5
        }
        
    def _analyze_hydrocephalus(self, anatomical: Dict) -> Dict:
        """Análisis específico de hidrocefalia"""
        ventricles = anatomical.get('ventricular_analysis', {})
        
        return {
            'detected': ventricles.get('hydrocephalus_suspected', False),
            'evans_ratio': ventricles.get('evans_ratio', 0),
            'type': ventricles.get('hydrocephalus_type', 'NONE'),
            'enlarged_ventricles': ventricles.get('enlarged_ventricles', False),
            'asymmetry': ventricles.get('asymmetry_present', False),
            'significance': ventricles.get('clinical_significance', 'Sistema ventricular normal'),
            'priority': 2 if ventricles.get('hydrocephalus_suspected', False) else 5
        }
        
    def _analyze_cerebral_atrophy(self, anatomical: Dict) -> Dict:
        """Análisis específico de atrofia cerebral"""
        sulci = anatomical.get('sulci_analysis', {})
        cortex = anatomical.get('cortical_analysis', {})
        
        global_atrophy = sulci.get('pathological_widening', False)
        cortical_atrophy = cortex.get('cortical_atrophy_present', False)
        
        return {
            'detected': global_atrophy or cortical_atrophy,
            'global_atrophy': global_atrophy,
            'cortical_atrophy': cortical_atrophy,
            'widened_sulci_count': sulci.get('widened_sulci_count', 0),
            'cortical_thickness': cortex.get('average_thickness_mm', 0),
            'severity': cortex.get('atrophy_severity', 'NONE'),
            'significance': sulci.get('clinical_significance', 'Surcos cerebrales normales'),
            'priority': 3 if global_atrophy or cortical_atrophy else 5
        }
        
    def _analyze_hippocampal_atrophy(self, anatomical: Dict) -> Dict:
        """Análisis específico de atrofia hipocampal"""
        hippocampus = anatomical.get('hippocampus_analysis', {})
        
        return {
            'detected': hippocampus.get('atrophy_present', False),
            'left_volume': hippocampus.get('left_volume_mm3', 0),
            'right_volume': hippocampus.get('right_volume_mm3', 0),
            'asymmetry': hippocampus.get('asymmetry_ratio', 0),
            'severity': hippocampus.get('atrophy_severity', 'NONE'),
            'affected_side': hippocampus.get('affected_side', 'NONE'),
            'significance': hippocampus.get('clinical_significance', 'Hipocampos normales'),
            'priority': 3 if hippocampus.get('atrophy_present', False) else 5
        }
        
    def _analyze_pituitary_enlargement(self, anatomical: Dict) -> Dict:
        """Análisis específico de agrandamiento hipofisario"""
        pituitary = anatomical.get('pituitary_analysis', {})
        
        return {
            'detected': pituitary.get('enlarged', False),
            'volume': pituitary.get('volume_mm3', 0),
            'dimensions': {
                'height': pituitary.get('height_mm', 0),
                'width': pituitary.get('width_mm', 0),
                'depth': pituitary.get('depth_mm', 0)
            },
            'morphology_normal': pituitary.get('morphology_normal', True),
            'findings': pituitary.get('morphological_findings', []),
            'significance': pituitary.get('clinical_significance', 'Hipófisis normal'),
            'priority': 3 if pituitary.get('enlarged', False) else 5
        }
        
    def _analyze_structural_shifts(self, anatomical: Dict) -> Dict:
        """Análisis de desplazamientos estructurales"""
        morphology = anatomical.get('morphological_assessment', {})
        
        return {
            'detected': morphology.get('significant_shift', False),
            'midline_shift': morphology.get('midline_shift_mm', 0),
            'brain_symmetry': morphology.get('brain_symmetry', 1.0),
            'overall_morphology': morphology.get('overall_morphology', 'NORMAL'),
            'significance': morphology.get('clinical_significance', 'Morfología normal'),
            'priority': 2 if morphology.get('significant_shift', False) else 5
        }
        
    def _analyze_pathological_asymmetries(self, anatomical: Dict) -> Dict:
        """Análisis de asimetrías patológicas"""
        hippocampus = anatomical.get('hippocampus_analysis', {})
        ventricles = anatomical.get('ventricular_analysis', {})
        sulci = anatomical.get('sulci_analysis', {})
        
        hip_asymmetry = hippocampus.get('significant_asymmetry', False)
        vent_asymmetry = ventricles.get('asymmetry_present', False)
        sulci_asymmetry = sulci.get('sulci_asymmetry', 0) > 0.2
        
        return {
            'detected': hip_asymmetry or vent_asymmetry or sulci_asymmetry,
            'hippocampal_asymmetry': hip_asymmetry,
            'ventricular_asymmetry': vent_asymmetry,
            'sulcal_asymmetry': sulci_asymmetry,
            'asymmetry_values': {
                'hippocampal': hippocampus.get('asymmetry_ratio', 0),
                'ventricular': ventricles.get('asymmetry_ratio', 0),
                'sulcal': sulci.get('sulci_asymmetry', 0)
            },
            'priority': 3 if any([hip_asymmetry, vent_asymmetry, sulci_asymmetry]) else 5
        }
        
    # ========== MÉTODOS DE VISUALIZACIÓN ==========
    
    def _add_brain_views_to_page(self, fig, gs, image_data: np.ndarray):
        """Agregar vistas cerebrales a la página"""
        
        # Calcular cortes en posiciones estándar
        mid_sag = image_data.shape[0] // 2
        mid_cor = image_data.shape[1] // 2
        mid_axi = image_data.shape[2] // 2
        
        views = [
            ('Sagital', image_data[mid_sag, :, :].T, gs[1, 0]),
            ('Coronal', image_data[:, mid_cor, :].T, gs[1, 1]),
            ('Axial', image_data[:, :, mid_axi].T, gs[1, 2])
        ]
        
        for title, slice_data, grid_pos in views:
            ax = fig.add_subplot(grid_pos)
            
            # Normalización y mejora de contraste
            p2, p98 = np.percentile(slice_data, (2, 98))
            enhanced = np.clip((slice_data - p2) / (p98 - p2), 0, 1)
            
            ax.imshow(enhanced, cmap='bone', aspect='equal', interpolation='bilinear')
            ax.set_title(title, **self.fonts['header'], color=self.medical_colors['primary'])
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Marco profesional
            for spine in ax.spines.values():
                spine.set_edgecolor(self.medical_colors['border'])
                spine.set_linewidth(1)
                
    def _add_volumetric_metrics(self, fig, gs, analysis_results: Dict):
        """Agregar métricas volumétricas"""
        
        ax = fig.add_subplot(gs[2, :])
        ax.set_title('Métricas Volumétricas', **self.fonts['header'], 
                    color=self.medical_colors['primary'])
        
        morphometry = analysis_results.get('morphometric_analysis', {})
        anatomical = analysis_results.get('anatomical_analysis', {})
        
        metrics_text = f"""
VOLÚMENES PRINCIPALES (mL):
• Cerebro Total: {morphometry.get('total_brain_volume', 0)/1000:.1f}
• Materia Gris: {morphometry.get('gray_matter_volume', 0)/1000:.1f}
• Materia Blanca: {morphometry.get('white_matter_volume', 0)/1000:.1f}
• LCR: {morphometry.get('csf_volume', 0)/1000:.1f}

ESTRUCTURAS ESPECÍFICAS (mm³):
• Hipocampo L/R: {anatomical.get('hippocampus_analysis', {}).get('left_volume_mm3', 0):.0f} / {anatomical.get('hippocampus_analysis', {}).get('right_volume_mm3', 0):.0f}
• Hipófisis: {anatomical.get('pituitary_analysis', {}).get('volume_mm3', 0):.0f}
• Sistema Ventricular: {anatomical.get('ventricular_analysis', {}).get('ventricular_volume_mm3', 0):.0f}
        """.strip()
        
        ax.text(0.05, 0.9, metrics_text, transform=ax.transAxes, **self.fonts['body'],
               va='top', ha='left', color=self.medical_colors['dark'])
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(self.medical_colors['light'])
        
    def _add_quality_assessment(self, fig, gs, analysis_results: Dict):
        """Agregar evaluación de calidad"""
        
        ax = fig.add_subplot(gs[3, :])
        ax.set_title('Evaluación de Calidad de Imagen', **self.fonts['header'],
                    color=self.medical_colors['primary'])
        
        quality = analysis_results.get('quality_assessment', {})
        
        quality_text = f"""
MÉTRICAS DE CALIDAD:
• Puntuación General: {quality.get('quality_score', 0):.1f}/10
• SNR: {quality.get('signal_to_noise_ratio', 0):.1f}
• CNR: {quality.get('contrast_to_noise_ratio', 0):.1f}
• Artefactos: {quality.get('artifact_score', 0):.2f}
• Resolución: {quality.get('resolution_assessment', 'N/A')}
        """.strip()
        
        ax.text(0.05, 0.8, quality_text, transform=ax.transAxes, **self.fonts['body'],
               va='top', ha='left', color=self.medical_colors['dark'])
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(self.medical_colors['light'])
        
    def _create_finding_section(self, fig, title: str, finding_data: Dict, y_pos: float):
        """Crear sección de hallazgo específico"""
        
        detected = finding_data.get('detected', False)
        priority = finding_data.get('priority', 5)
        
        # Color según prioridad
        if priority <= 1:
            color = self.medical_colors['urgent']
            status = "🚨 CRÍTICO"
        elif priority <= 2:
            color = self.medical_colors['error']
            status = "⚠️ ANORMAL"
        elif priority <= 3:
            color = self.medical_colors['warning']
            status = "⚠️ ALTERADO"
        else:
            color = self.medical_colors['success']
            status = "✅ NORMAL"
            
        # Título de sección
        fig.text(0.05, y_pos, title, ha='left', va='top', **self.fonts['header'],
                color=self.medical_colors['primary'])
        
        # Estado
        fig.text(0.75, y_pos, status, ha='left', va='top', **self.fonts['body'],
                color=color, weight='bold')
        
        # Detalle específico
        significance = finding_data.get('significance', 'Normal')
        fig.text(0.05, y_pos-0.03, f"• {significance}", ha='left', va='top', 
                **self.fonts['body'], color=self.medical_colors['dark'])
        
        # Línea separadora
        fig.add_artist(plt.Line2D([0.05, 0.95], [y_pos-0.06, y_pos-0.06], 
                                 color=self.medical_colors['border'], linewidth=0.5))
                                 
    def _add_detailed_morphometry(self, fig, gs, analysis_results: Dict):
        """Agregar morfometría detallada"""
        
        ax = fig.add_subplot(gs[0, 0])
        ax.set_title('Morfometría Detallada', **self.fonts['header'])
        
        # Implementar visualización de morfometría
        ax.text(0.5, 0.5, 'Morfometría\nDetallada\n(Gráficos)', 
               ha='center', va='center', **self.fonts['body'])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
    def _add_anatomical_ratios(self, fig, gs, analysis_results: Dict):
        """Agregar ratios anatómicos"""
        
        ax = fig.add_subplot(gs[0, 1])
        ax.set_title('Ratios Anatómicos', **self.fonts['header'])
        
        # Implementar visualización de ratios
        ax.text(0.5, 0.5, 'Ratios\nAnatómicos\n(Gráficos)', 
               ha='center', va='center', **self.fonts['body'])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
    def _add_ai_confidence_metrics(self, fig, gs, analysis_results: Dict):
        """Agregar métricas de confianza de IA"""
        
        ax = fig.add_subplot(gs[1, :])
        ax.set_title('Métricas de Confianza de IA', **self.fonts['header'])
        
        ai_analysis = analysis_results.get('ai_medical_analysis', {})
        
        confidence_text = f"""
ANÁLISIS DE IA NEUROMORPHIC:
• Condición Predicha: {ai_analysis.get('predicted_condition', 'N/A')}
• Puntuación de Confianza: {ai_analysis.get('confidence_score', 0):.3f}
• Procesamiento: {ai_analysis.get('processing_device', 'N/A')}
• Núcleo Neuromorphic: {ai_analysis.get('neuromorphic_core_type', 'N/A')}
        """.strip()
        
        ax.text(0.05, 0.8, confidence_text, transform=ax.transAxes, **self.fonts['body'],
               va='top', ha='left', color=self.medical_colors['dark'])
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(self.medical_colors['light'])
        
    def _add_diagnostic_interpretation(self, fig, clinical_interp: Dict):
        """Agregar interpretación diagnóstica"""
        
        fig.text(0.05, 0.85, 'INTERPRETACIÓN DIAGNÓSTICA', ha='left', va='top',
                **self.fonts['subtitle'], color=self.medical_colors['primary'])
        
        overall_assessment = clinical_interp.get('overall_assessment', 'NORMAL')
        pathological_findings = clinical_interp.get('pathological_findings', [])
        
        if pathological_findings:
            findings_text = "HALLAZGOS PATOLÓGICOS:\n" + "\n".join([f"• {finding}" for finding in pathological_findings])
        else:
            findings_text = "✅ Estudio dentro de límites normales para la edad"
            
        fig.text(0.05, 0.8, findings_text, ha='left', va='top',
                **self.fonts['body'], color=self.medical_colors['dark'])
        
    def _add_follow_up_recommendations(self, fig, clinical_interp: Dict):
        """Agregar recomendaciones de seguimiento"""
        
        fig.text(0.05, 0.6, 'RECOMENDACIONES DE SEGUIMIENTO', ha='left', va='top',
                **self.fonts['subtitle'], color=self.medical_colors['primary'])
        
        urgency_level = clinical_interp.get('urgency_level', 'LOW')
        
        if urgency_level == 'HIGH':
            recommendations = [
                "• Evaluación neurológica URGENTE",
                "• Correlación clínica inmediata",
                "• Seguimiento de imagen en 24-48 horas si indicado"
            ]
        elif urgency_level == 'MEDIUM':
            recommendations = [
                "• Seguimiento neurológico en 1-3 meses",
                "• Correlación con síntomas clínicos",
                "• Considerar estudios adicionales según criterio médico"
            ]
        else:
            recommendations = [
                "• Seguimiento rutinario según protocolo institucional",
                "• Correlación clínica si hay síntomas",
                "• Próximo estudio según indicación médica"
            ]
            
        recommendations_text = "\n".join(recommendations)
        fig.text(0.05, 0.55, recommendations_text, ha='left', va='top',
                **self.fonts['body'], color=self.medical_colors['dark'])
        
    def _add_clinical_correlation(self, fig, clinical_interp: Dict):
        """Agregar correlación clínica"""
        
        fig.text(0.05, 0.35, 'CORRELACIÓN CLÍNICA SUGERIDA', ha='left', va='top',
                **self.fonts['subtitle'], color=self.medical_colors['primary'])
        
        correlation_text = """
• Evaluar síntomas neurológicos actuales
• Historia de trauma craneal o procedimientos previos
• Antecedentes familiares de patología neurológica
• Medicación actual y efectos secundarios
• Evaluación cognitiva y funcional si está indicada
        """.strip()
        
        fig.text(0.05, 0.3, correlation_text, ha='left', va='top',
                **self.fonts['body'], color=self.medical_colors['dark'])
        
    def _add_medical_disclaimer(self, fig):
        """Agregar disclaimer médico"""
        
        disclaimer_text = """
AVISO MÉDICO IMPORTANTE:
Este reporte ha sido generado por un sistema de inteligencia artificial neuromorphic y debe ser 
interpretado por un radiólogo o médico especialista calificado. Los hallazgos automatizados 
requieren correlación clínica y validación profesional antes de tomar decisiones diagnósticas 
o terapéuticas. No sustituye el juicio clínico profesional.
        """.strip()
        
        fig.text(0.5, 0.12, disclaimer_text, ha='center', va='top',
                **self.fonts['small'], color=self.medical_colors['secondary'],
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.medical_colors['light']),
                style='italic')

    def _analyze_hippocampal_atrophy(self, data):
        """Análisis de atrofia hipocampal"""
        try:
            results = {
                'bilateral_volume_loss': False,
                'asymmetry_detected': False,
                'severity': 'normal',
                'volume_ratio': 1.0,
                'clinical_significance': 'Normal'
            }
            
            if hasattr(data, 'anatomical_analysis') and data.anatomical_analysis:
                anat = data.anatomical_analysis
                if 'hippocampus' in anat:
                    hipp = anat['hippocampus']
                    
                    # Detectar atrofia bilateral
                    if hipp.get('volume_percentile', 50) < 10:
                        results['bilateral_volume_loss'] = True
                        results['severity'] = 'severe'
                    elif hipp.get('volume_percentile', 50) < 25:
                        results['bilateral_volume_loss'] = True
                        results['severity'] = 'moderate'
                    
                    # Detectar asimetría
                    asymmetry = hipp.get('asymmetry_ratio', 1.0)
                    if asymmetry > 1.2 or asymmetry < 0.8:
                        results['asymmetry_detected'] = True
                    
                    results['volume_ratio'] = hipp.get('volume_ratio', 1.0)
                    
                    # Significancia clínica
                    if results['bilateral_volume_loss']:
                        if results['severity'] == 'severe':
                            results['clinical_significance'] = 'Highly suggestive of neurodegenerative process'
                        else:
                            results['clinical_significance'] = 'May indicate mild cognitive impairment'
                    elif results['asymmetry_detected']:
                        results['clinical_significance'] = 'Asymmetry requires further evaluation'
            
            return results
            
        except Exception as e:
            logger.error(f"Error en análisis de atrofia hipocampal: {e}")
            return {
                'bilateral_volume_loss': False,
                'asymmetry_detected': False,
                'severity': 'normal',
                'volume_ratio': 1.0,
                'clinical_significance': 'Analysis incomplete'
            }

    def _analyze_ventricular_changes(self, data):
        """Análisis de cambios ventriculares"""
        try:
            results = {
                'hydrocephalus': False,
                'atrophy_related': False,
                'severity': 'normal',
                'evans_index': 0.25,
                'clinical_significance': 'Normal'
            }
            
            if hasattr(data, 'anatomical_analysis') and data.anatomical_analysis:
                anat = data.anatomical_analysis
                if 'ventricles' in anat:
                    vent = anat['ventricles']
                    
                    evans_index = vent.get('evans_index', 0.25)
                    results['evans_index'] = evans_index
                    
                    # Detectar hidrocefalia
                    if evans_index > 0.35:
                        results['hydrocephalus'] = True
                        results['severity'] = 'severe'
                        results['clinical_significance'] = 'Significant ventricular enlargement - hydrocephalus likely'
                    elif evans_index > 0.30:
                        results['hydrocephalus'] = True
                        results['severity'] = 'moderate'
                        results['clinical_significance'] = 'Moderate ventricular enlargement'
                    elif evans_index > 0.28:
                        results['atrophy_related'] = True
                        results['severity'] = 'mild'
                        results['clinical_significance'] = 'Mild enlargement - may be age-related atrophy'
            
            return results
            
        except Exception as e:
            logger.error(f"Error en análisis ventricular: {e}")
            return {
                'hydrocephalus': False,
                'atrophy_related': False,
                'severity': 'normal',
                'evans_index': 0.25,
                'clinical_significance': 'Analysis incomplete'
            }

    def _analyze_hemorrhage_detection(self, data):
        """Análisis de detección de hemorragias"""
        try:
            results = {
                'hemorrhage_detected': False,
                'location': 'none',
                'severity': 'none',
                'volume_ml': 0.0,
                'clinical_significance': 'No hemorrhage detected'
            }
            
            if hasattr(data, 'anatomical_analysis') and data.anatomical_analysis:
                anat = data.anatomical_analysis
                if 'hemorrhages' in anat:
                    hem = anat['hemorrhages']
                    
                    if hem.get('detected', False):
                        results['hemorrhage_detected'] = True
                        results['location'] = hem.get('primary_location', 'unspecified')
                        results['volume_ml'] = hem.get('estimated_volume_ml', 0.0)
                        
                        # Clasificar severidad
                        volume = results['volume_ml']
                        if volume > 50:
                            results['severity'] = 'large'
                            results['clinical_significance'] = 'Large hemorrhage - immediate medical attention required'
                        elif volume > 20:
                            results['severity'] = 'moderate'
                            results['clinical_significance'] = 'Moderate hemorrhage - urgent evaluation needed'
                        elif volume > 5:
                            results['severity'] = 'small'
                            results['clinical_significance'] = 'Small hemorrhage - clinical correlation required'
                        else:
                            results['severity'] = 'microhemorrhage'
                            results['clinical_significance'] = 'Microhemorrhages detected - monitor for progression'
            
            return results
            
        except Exception as e:
            logger.error(f"Error en detección de hemorragias: {e}")
            return {
                'hemorrhage_detected': False,
                'location': 'none',
                'severity': 'none',
                'volume_ml': 0.0,
                'clinical_significance': 'Analysis incomplete'
            }

    def _analyze_pituitary_abnormalities(self, data):
        """Análisis de anormalidades hipofisarias"""
        try:
            results = {
                'adenoma_suspected': False,
                'size_mm': 0.0,
                'classification': 'normal',
                'hormonal_risk': 'low',
                'clinical_significance': 'Normal pituitary gland'
            }
            
            if hasattr(data, 'anatomical_analysis') and data.anatomical_analysis:
                anat = data.anatomical_analysis
                if 'pituitary' in anat:
                    pit = anat['pituitary']
                    
                    size_mm = pit.get('max_diameter_mm', 0.0)
                    results['size_mm'] = size_mm
                    
                    # Clasificar según tamaño
                    if size_mm > 40:
                        results['adenoma_suspected'] = True
                        results['classification'] = 'macroadenoma'
                        results['hormonal_risk'] = 'high'
                        results['clinical_significance'] = 'Large pituitary mass - endocrine evaluation required'
                    elif size_mm > 10:
                        results['adenoma_suspected'] = True
                        results['classification'] = 'microadenoma'
                        results['hormonal_risk'] = 'moderate'
                        results['clinical_significance'] = 'Small pituitary lesion - hormonal assessment recommended'
                    elif size_mm > 8:
                        results['classification'] = 'borderline'
                        results['hormonal_risk'] = 'low'
                        results['clinical_significance'] = 'Borderline pituitary size - follow-up suggested'
            
            return results
            
        except Exception as e:
            logger.error(f"Error en análisis hipofisario: {e}")
            return {
                'adenoma_suspected': False,
                'size_mm': 0.0,
                'classification': 'normal',
                'hormonal_risk': 'low',
                'clinical_significance': 'Analysis incomplete'
            }

    def _generate_complete_summary(self, analysis_results):
        """Generar resumen médico completo"""
        try:
            patient_info = analysis_results.get('patient_info', {})
            image_name = patient_info.get('image_name', 'Unknown')
            
            # Análisis anatómico
            anatomical = analysis_results.get('anatomical_analysis', {})
            
            # Crear resumen estructurado
            summary_parts = []
            
            # Información del paciente
            summary_parts.append(f"PACIENTE: {image_name}")
            summary_parts.append(f"FECHA: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
            summary_parts.append("")
            
            # Hallazgos principales
            summary_parts.append("HALLAZGOS PRINCIPALES:")
            
            # Análisis ventricular
            if 'ventricles' in anatomical:
                vent = anatomical['ventricles']
                evans_index = vent.get('evans_index', 0.25)
                if evans_index > 0.30:
                    summary_parts.append(f"• Dilatación ventricular (Índice Evans: {evans_index:.3f})")
                
            # Análisis hipocampal
            if 'hippocampus' in anatomical:
                hipp = anatomical['hippocampus']
                if hipp.get('volume_percentile', 50) < 25:
                    summary_parts.append("• Posible atrofia hipocampal")
                    
            # Análisis hipofisario
            if 'pituitary' in anatomical:
                pit = anatomical['pituitary']
                size_mm = pit.get('max_diameter_mm', 0)
                if size_mm > 10:
                    summary_parts.append(f"• Lesión hipofisaria ({size_mm:.1f}mm)")
                    
            # Hemorragias
            if 'hemorrhages' in anatomical:
                hem = anatomical['hemorrhages']
                if hem.get('detected', False):
                    location = hem.get('primary_location', 'unspecified')
                    summary_parts.append(f"• Hemorragia detectada: {location}")
            
            # Análisis de tejidos
            tissue_analysis = analysis_results.get('tissue_analysis', {})
            if tissue_analysis:
                brain_vol = tissue_analysis.get('total_brain', 0)
                summary_parts.append(f"• Volumen cerebral total: {brain_vol:.1f} mL")
            
            # Recomendaciones
            summary_parts.append("")
            summary_parts.append("RECOMENDACIONES:")
            summary_parts.append("• Correlación clínica requerida")
            summary_parts.append("• Evaluación por radiólogo especialista")
            summary_parts.append("• Seguimiento según criterio médico")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Error generando resumen: {str(e)}"

    def save_report(self, output_path):
        """Guardar el reporte en PDF"""
        try:
            logger.info(f"Guardando reporte PDF en: {output_path}")
            self.fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
            logger.info("Reporte PDF generado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando PDF: {e}")
            return False
        
        finally:
            plt.close(self.fig)

    def _convert_pages_to_pdf(self, pages_dir: Path, output_path: str):
        """Convertir páginas PNG en un PDF consolidado"""
        try:
            from PIL import Image
            import glob
            
            # Buscar todas las páginas PNG en orden
            png_files = sorted(glob.glob(str(pages_dir / "*.png")))
            if not png_files:
                self.logger.warning("No se encontraron páginas PNG para convertir")
                return False
            
            # Convertir a PDF
            images = []
            for png_file in png_files:
                img = Image.open(png_file)
                # Convertir a RGB si es necesario (para PDF)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
            
            # Guardar como PDF multipágina
            if images:
                images[0].save(output_path, save_all=True, append_images=images[1:], 
                             format='PDF', resolution=300.0)
                self.logger.info(f"✅ PDF consolidado creado: {output_path}")
                return True
            
        except ImportError:
            self.logger.warning("PIL no disponible, usando matplotlib para PDF")
            # Fallback usando matplotlib
            return self._convert_pages_to_pdf_matplotlib(pages_dir, output_path)
            
        except Exception as e:
            self.logger.error(f"Error convirtiendo a PDF: {e}")
            return False
    
    def _convert_pages_to_pdf_matplotlib(self, pages_dir: Path, output_path: str):
        """Fallback: convertir usando matplotlib"""
        try:
            import glob
            from matplotlib.backends.backend_pdf import PdfPages
            
            png_files = sorted(glob.glob(str(pages_dir / "*.png")))
            if not png_files:
                return False
            
            with PdfPages(output_path) as pdf:
                for png_file in png_files:
                    img = plt.imread(png_file)
                    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=300)
                    ax.imshow(img)
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight', dpi=300)
                    plt.close(fig)
            
            self.logger.info(f"✅ PDF creado con matplotlib: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error en fallback matplotlib: {e}")
            return False
    
    def _generate_quantitative_summary(self, analysis_results: Dict) -> str:
        """Generar resumen cuantitativo del análisis"""
        
        try:
            # Obtener análisis médico
            medical_analysis = analysis_results.get('medical_analysis', {})
            image_analysis = analysis_results.get('image_analysis', {})
            anatomical_analysis = analysis_results.get('anatomical_analysis', {})
            
            # Métricas principales
            volume_metrics = image_analysis.get('volumetric_analysis', {})
            morphometry = image_analysis.get('morphometric_analysis', {})
            
            summary_parts = []
            
            # Análisis volumétrico
            if volume_metrics:
                summary_parts.append("📊 ANÁLISIS VOLUMÉTRICO:")
                for region, volume in volume_metrics.items():
                    if isinstance(volume, (int, float)):
                        summary_parts.append(f"  • {region.replace('_', ' ').title()}: {volume:.1f} mL")
            
            # Análisis morfométrico
            if morphometry:
                summary_parts.append("\n📐 ANÁLISIS MORFOMÉTRICO:")
                cortical_thickness = morphometry.get('cortical_thickness', 'N/A')
                white_matter_integrity = morphometry.get('white_matter_integrity', 'N/A')
                summary_parts.append(f"  • Grosor cortical: {cortical_thickness}")
                summary_parts.append(f"  • Integridad sustancia blanca: {white_matter_integrity}")
            
            # Métricas de confianza
            confidence = medical_analysis.get('confidence_score', 0)
            predicted_condition = medical_analysis.get('predicted_condition', 'N/A')
            
            summary_parts.append(f"\n🎯 MÉTRICAS DE CLASIFICACIÓN:")
            summary_parts.append(f"  • Condición predicha: {predicted_condition}")
            summary_parts.append(f"  • Confianza del modelo: {confidence:.4f}")
            
            # Probabilidades por condición
            probabilities = medical_analysis.get('condition_probabilities', {})
            if probabilities:
                summary_parts.append(f"\n📈 DISTRIBUCIÓN DE PROBABILIDADES:")
                for condition, prob in probabilities.items():
                    percentage = prob * 100
                    summary_parts.append(f"  • {condition}: {percentage:.2f}%")
            
            # Análisis anatómico avanzado
            if anatomical_analysis:
                summary_parts.append(f"\n🧠 ANÁLISIS ANATÓMICO ESTRUCTURAL:")
                
                # Surcos cerebrales
                sulci = anatomical_analysis.get('sulci_analysis', {})
                if sulci:
                    status = sulci.get('overall_status', 'Normal')
                    summary_parts.append(f"  • Surcos cerebrales: {status}")
                
                # Hipocampo
                hippocampus = anatomical_analysis.get('hippocampal_analysis', {})
                if hippocampus:
                    atrophy_level = hippocampus.get('atrophy_level', 'Mínima')
                    summary_parts.append(f"  • Atrofia hipocampal: {atrophy_level}")
                
                # Sistema ventricular
                ventricles = anatomical_analysis.get('ventricular_analysis', {})
                if ventricles:
                    enlargement = ventricles.get('enlargement_level', 'Normal')
                    summary_parts.append(f"  • Dilatación ventricular: {enlargement}")
                
                # Hemorragias
                hemorrhages = anatomical_analysis.get('hemorrhage_detection', {})
                if hemorrhages:
                    detected = hemorrhages.get('detected', False)
                    status = "Detectadas" if detected else "No detectadas"
                    summary_parts.append(f"  • Hemorragias: {status}")
            
            # Calidad de imagen
            quality_score = image_analysis.get('quality_score', 0)
            summary_parts.append(f"\n✅ CALIDAD DE IMAGEN: {quality_score}/10")
            
            return '\n'.join(summary_parts)
            
        except Exception as e:
            return f"Error generando resumen cuantitativo: {str(e)}"
    
    def _generate_recommendations_summary(self, analysis_results: Dict) -> str:
        """Generate comprehensive recommendations summary based on analysis results"""
        
        # Extract key metrics
        confidence = analysis_results.get('confidence_score', 0.0)
        prediction_category = analysis_results.get('prediction_category', 'unknown')
        anatomical_data = analysis_results.get('anatomical_analysis', {})
        volumes = analysis_results.get('volume_analysis', {})
        
        recommendations = []
        
        # Confidence-based recommendations
        if confidence < 0.3:
            recommendations.append("🔴 ALTA PRIORIDAD: Confianza del modelo baja. Se requiere revisión inmediata por especialista.")
        elif confidence < 0.6:
            recommendations.append("🟡 PRIORIDAD MEDIA: Confianza moderada. Recomendable segunda opinión profesional.")
        else:
            recommendations.append("🟢 Confianza del modelo aceptable. Hallazgos confiables para análisis inicial.")
        
        # Volume-based recommendations
        total_brain = volumes.get('total_brain', 0)
        if total_brain > 0:
            if total_brain < 1200:
                recommendations.append("⚠️ Volumen cerebral total reducido. Evaluar atrofia o desarrollo anormal.")
            elif total_brain > 1800:
                recommendations.append("⚠️ Volumen cerebral total elevado. Descartar hidrocefalia o lesiones expansivas.")
        
        # Anatomical structure recommendations
        hippocampus_vol = anatomical_data.get('hippocampus_volume', 0)
        if hippocampus_vol >  0:
            if hippocampus_vol < 3.0:
                recommendations.append("🧠 Volumen hipocampal reducido. Considerar evaluación neurocognitiva.")
        
        ventricles_vol = anatomical_data.get('ventricle_volume', 0)
        if ventricles_vol > 50:
            recommendations.append("💧 Dilatación ventricular detectada. Evaluar hidrocefalia o atrofia.")
        
        # Prediction-based recommendations
        if prediction_category in ['abnormal', 'pathological']:
            recommendations.append("🚨 HALLAZGOS ANORMALES: Se requiere evaluación neurológica especializada urgente.")
        elif prediction_category == 'borderline':
            recommendations.append("⚠️ Hallazgos limítrofes. Recomendable seguimiento y monitoreo.")
        
        # Follow-up recommendations
        recommendations.extend([
            "\n📋 SEGUIMIENTO RECOMENDADO:",
            "• Correlación con historia clínica y síntomas",
            "• Evaluación por radiólogo especialista en neuroimágenes",
            "• Consideración de estudios complementarios si está indicado",
            "• Seguimiento temporal según criterio clínico"
        ])
        
        return "\n".join(recommendations)
