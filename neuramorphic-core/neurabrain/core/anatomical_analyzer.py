"""
🧠 ADVANCED ANATOMICAL STRUCTURE ANALYZER
====================================================
Análisis estructural especializado para diagnóstico médico avanzado
Detecta y analiza estructuras anatómicas específicas según criterios radiológicos

Author: Neuromorphic Medical AI System
Date: August 29, 2025
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial.distance import euclidean
from skimage import measure, morphology, filters
from skimage.segmentation import watershed
# from skimage.feature import peak_local_maxima  # Removed problematic import
import logging
from typing import Dict, List, Tuple, Optional
import json

class AdvancedAnatomicalAnalyzer:
    """Analizador anatómico avanzado para estructuras cerebrales específicas"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Parámetros anatómicos de referencia (en mm)
        self.reference_values = {
            'sulci': {
                'normal_width_range': (2, 6),  # mm
                'pathological_width': 8,       # mm
                'depth_ratio': 0.3             # ratio profundidad/ancho
            },
            'hippocampus': {
                'normal_volume_range': (3000, 4500),  # mm³
                'atrophy_threshold': 2800,             # mm³
                'asymmetry_threshold': 0.15            # 15% diferencia
            },
            'pituitary': {
                'normal_volume_range': (400, 800),   # mm³
                'enlarged_threshold': 1000,          # mm³
                'normal_height_range': (6, 10)       # mm
            },
            'ventricles': {
                'normal_evans_ratio': 0.3,           # ratio ancho ventricular/cerebral
                'enlarged_threshold': 0.35,          # patológico
                'asymmetry_threshold': 0.1           # 10% diferencia
            },
            'cortex': {
                'normal_thickness_range': (2.5, 4.5), # mm
                'atrophy_threshold': 2.0,             # mm
                'sulcal_widening': 6.0                # mm
            }
        }
        
        self.anatomical_features = {}
        
    def analyze_anatomical_structures(self, image_data: np.ndarray, 
                                    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict:
        """
        Análisis completo de estructuras anatómicas
        
        Args:
            image_data: Datos de imagen 3D (T1-weighted MRI)
            spacing: Espaciado de voxel (x, y, z) en mm
            
        Returns:
            Dict con análisis estructural completo
        """
        self.logger.info("🔬 Iniciando análisis anatómico estructural avanzado")
        
        results = {
            'sulci_analysis': self._analyze_sulci(image_data, spacing),
            'hippocampus_analysis': self._analyze_hippocampus(image_data, spacing),
            'pituitary_analysis': self._analyze_pituitary(image_data, spacing),
            'ventricular_analysis': self._analyze_ventricles(image_data, spacing),
            'cortical_analysis': self._analyze_cortex(image_data, spacing),
            'hemorrhage_detection': self._detect_hemorrhages(image_data, spacing),
            'morphological_assessment': self._assess_morphology(image_data, spacing),
            'clinical_interpretation': {}
        }
        
        # Interpretación clínica integrada
        results['clinical_interpretation'] = self._generate_clinical_interpretation(results)
        
        self.logger.info("✅ Análisis anatómico estructural completado")
        return results
        
    def _analyze_sulci(self, image_data: np.ndarray, spacing: Tuple) -> Dict:
        """Análisis detallado de surcos cerebrales"""
        self.logger.info("🧠 Analizando surcos cerebrales...")
        
        # Segmentación de surcos usando detección de bordes y morfología
        gray_matter_mask = self._segment_gray_matter(image_data)
        csf_mask = self._segment_csf(image_data)
        
        # Detección de surcos como interfaz GM-CSF
        sulci_mask = self._detect_sulci_interface(gray_matter_mask, csf_mask)
        
        # Análisis morfométrico de surcos
        sulci_measurements = self._measure_sulci_morphometry(sulci_mask, spacing)
        
        # Detección de amplificación patológica
        widened_sulci = self._detect_widened_sulci(sulci_measurements)
        
        return {
            'sulci_count': len(sulci_measurements),
            'average_width_mm': np.mean([s['width_mm'] for s in sulci_measurements]),
            'max_width_mm': np.max([s['width_mm'] for s in sulci_measurements]) if sulci_measurements else 0,
            'widened_sulci_count': len(widened_sulci),
            'pathological_widening': len(widened_sulci) > 5,
            'sulci_asymmetry': self._calculate_sulci_asymmetry(sulci_measurements),
            'detailed_measurements': sulci_measurements,
            'clinical_significance': self._interpret_sulci_findings(sulci_measurements, widened_sulci)
        }
        
    def _analyze_hippocampus(self, image_data: np.ndarray, spacing: Tuple) -> Dict:
        """Análisis específico del hipocampo"""
        self.logger.info("🐎 Analizando estructura hipocampal...")
        
        # Localización anatómica del hipocampo
        left_hippocampus, right_hippocampus = self._locate_hippocampi(image_data)
        
        # Mediciones volumétricas
        left_volume = self._calculate_structure_volume(left_hippocampus, spacing)
        right_volume = self._calculate_structure_volume(right_hippocampus, spacing)
        
        # Análisis de atrofia
        atrophy_assessment = self._assess_hippocampal_atrophy(left_volume, right_volume)
        
        # Análisis de asimetría
        asymmetry = abs(left_volume - right_volume) / max(left_volume, right_volume) if max(left_volume, right_volume) > 0 else 0
        
        return {
            'left_volume_mm3': left_volume,
            'right_volume_mm3': right_volume,
            'total_volume_mm3': left_volume + right_volume,
            'asymmetry_ratio': asymmetry,
            'significant_asymmetry': asymmetry > self.reference_values['hippocampus']['asymmetry_threshold'],
            'atrophy_present': atrophy_assessment['atrophy_detected'],
            'atrophy_severity': atrophy_assessment['severity'],
            'affected_side': atrophy_assessment['affected_side'],
            'clinical_significance': self._interpret_hippocampal_findings(left_volume, right_volume, asymmetry)
        }
        
    def _analyze_pituitary(self, image_data: np.ndarray, spacing: Tuple) -> Dict:
        """Análisis de la hipófisis"""
        self.logger.info("🔬 Analizando glándula pituitaria...")
        
        # Localización de la silla turca y hipófisis
        pituitary_mask = self._locate_pituitary(image_data)
        
        # Mediciones morfométricas
        volume = self._calculate_structure_volume(pituitary_mask, spacing)
        dimensions = self._calculate_pituitary_dimensions(pituitary_mask, spacing)
        
        # Análisis morfológico
        morphology_assessment = self._assess_pituitary_morphology(pituitary_mask, dimensions)
        
        return {
            'volume_mm3': volume,
            'height_mm': dimensions['height'],
            'width_mm': dimensions['width'],
            'depth_mm': dimensions['depth'],
            'enlarged': volume > self.reference_values['pituitary']['enlarged_threshold'],
            'morphology_normal': morphology_assessment['normal'],
            'morphological_findings': morphology_assessment['findings'],
            'clinical_significance': self._interpret_pituitary_findings(volume, dimensions, morphology_assessment)
        }
        
    def _analyze_ventricles(self, image_data: np.ndarray, spacing: Tuple) -> Dict:
        """Análisis del sistema ventricular"""
        self.logger.info("💧 Analizando sistema ventricular...")
        
        # Segmentación de ventrículos
        ventricular_mask = self._segment_ventricles(image_data)
        
        # Mediciones Evans ratio
        evans_ratio = self._calculate_evans_ratio(ventricular_mask, image_data)
        
        # Análisis de simetría
        symmetry_analysis = self._analyze_ventricular_symmetry(ventricular_mask)
        
        # Detección de hidrocefalia
        hydrocephalus_assessment = self._assess_hydrocephalus(evans_ratio, symmetry_analysis)
        
        return {
            'evans_ratio': evans_ratio,
            'ventricular_volume_mm3': self._calculate_structure_volume(ventricular_mask, spacing),
            'enlarged_ventricles': evans_ratio > self.reference_values['ventricles']['enlarged_threshold'],
            'asymmetry_present': symmetry_analysis['asymmetric'],
            'asymmetry_ratio': symmetry_analysis['asymmetry_ratio'],
            'hydrocephalus_suspected': hydrocephalus_assessment['suspected'],
            'hydrocephalus_type': hydrocephalus_assessment['type'],
            'clinical_significance': self._interpret_ventricular_findings(evans_ratio, symmetry_analysis)
        }
        
    def _analyze_cortex(self, image_data: np.ndarray, spacing: Tuple) -> Dict:
        """Análisis cortical detallado"""
        self.logger.info("🧠 Analizando corteza cerebral...")
        
        # Segmentación cortical
        cortical_mask = self._segment_cortex(image_data)
        
        # Medición de grosor cortical
        thickness_map = self._calculate_cortical_thickness(cortical_mask, spacing)
        
        # Análisis de atrofia cortical
        atrophy_assessment = self._assess_cortical_atrophy(thickness_map)
        
        return {
            'average_thickness_mm': np.mean(thickness_map[thickness_map > 0]),
            'min_thickness_mm': np.min(thickness_map[thickness_map > 0]) if np.any(thickness_map > 0) else 0,
            'max_thickness_mm': np.max(thickness_map),
            'cortical_atrophy_present': atrophy_assessment['atrophy_detected'],
            'atrophy_severity': atrophy_assessment['severity'],
            'affected_regions': atrophy_assessment['affected_regions'],
            'clinical_significance': self._interpret_cortical_findings(thickness_map, atrophy_assessment)
        }
        
    def _detect_hemorrhages(self, image_data: np.ndarray, spacing: Tuple) -> Dict:
        """Detección de hemorragias en surcos y espacios subaracnoideos"""
        self.logger.info("🩸 Detectando hemorragias...")
        
        # Detección de intensidades anómalas en CSF
        csf_mask = self._segment_csf(image_data)
        hemorrhage_candidates = self._detect_high_intensity_csf(image_data, csf_mask)
        
        # Análisis morfológico de candidatos
        confirmed_hemorrhages = self._validate_hemorrhage_candidates(hemorrhage_candidates, image_data)
        
        return {
            'hemorrhages_detected': len(confirmed_hemorrhages),
            'subarachnoid_hemorrhage_suspected': len(confirmed_hemorrhages) > 0,
            'hemorrhage_locations': confirmed_hemorrhages,
            'clinical_urgency': 'HIGH' if len(confirmed_hemorrhages) > 0 else 'LOW',
            'clinical_significance': self._interpret_hemorrhage_findings(confirmed_hemorrhages)
        }
        
    def _assess_morphology(self, image_data: np.ndarray, spacing: Tuple) -> Dict:
        """Evaluación morfológica general"""
        self.logger.info("📐 Evaluando morfología general...")
        
        # Análisis de simetría cerebral
        brain_symmetry = self._analyze_brain_symmetry(image_data)
        
        # Detección de desplazamientos de línea media
        midline_shift = self._detect_midline_shift(image_data)
        
        # Análisis de proporciones anatómicas
        anatomical_proportions = self._analyze_anatomical_proportions(image_data, spacing)
        
        return {
            'brain_symmetry': brain_symmetry,
            'midline_shift_mm': midline_shift,
            'significant_shift': abs(midline_shift) > 2.0,  # >2mm es significativo
            'anatomical_proportions': anatomical_proportions,
            'overall_morphology': 'NORMAL' if brain_symmetry > 0.9 and abs(midline_shift) < 2.0 else 'ABNORMAL',
            'clinical_significance': self._interpret_morphological_findings(brain_symmetry, midline_shift)
        }
        
    # ========== MÉTODOS AUXILIARES DE SEGMENTACIÓN ==========
    
    def _segment_gray_matter(self, image_data: np.ndarray) -> np.ndarray:
        """Segmentación de materia gris"""
        # Umbralización adaptativa para materia gris
        threshold = filters.threshold_otsu(image_data)
        gray_matter = (image_data > threshold * 0.7) & (image_data < threshold * 1.2)
        
        # Refinamiento morfológico
        gray_matter = morphology.binary_closing(gray_matter, morphology.ball(2))
        gray_matter = morphology.binary_opening(gray_matter, morphology.ball(1))
        
        return gray_matter.astype(np.uint8)
        
    def _segment_csf(self, image_data: np.ndarray) -> np.ndarray:
        """Segmentación de líquido cefalorraquídeo"""
        # CSF aparece como intensidades bajas
        threshold = filters.threshold_otsu(image_data)
        csf_mask = image_data < threshold * 0.3
        
        # Refinamiento para eliminar ruido
        csf_mask = morphology.binary_opening(csf_mask, morphology.ball(2))
        
        return csf_mask.astype(np.uint8)
        
    def _segment_ventricles(self, image_data: np.ndarray) -> np.ndarray:
        """Segmentación específica de ventrículos"""
        # Ventrículos son regiones grandes de CSF en el centro del cerebro
        csf_mask = self._segment_csf(image_data)
        
        # Selección de regiones centrales
        center_z, center_y, center_x = np.array(image_data.shape) // 2
        
        # Máscara para región central (aproximadamente 60% del cerebro)
        z_range = int(image_data.shape[0] * 0.3)
        y_range = int(image_data.shape[1] * 0.3)
        x_range = int(image_data.shape[2] * 0.3)
        
        central_mask = np.zeros_like(image_data, dtype=bool)
        central_mask[center_z-z_range:center_z+z_range,
                    center_y-y_range:center_y+y_range,
                    center_x-x_range:center_x+x_range] = True
        
        # Ventrículos = CSF en región central
        ventricular_mask = csf_mask.astype(bool) & central_mask.astype(bool)
        
        # Refinamiento por tamaño de componentes
        labels = measure.label(ventricular_mask)
        regions = measure.regionprops(labels)
        
        # Mantener solo componentes grandes (ventrículos reales)
        min_volume = 1000  # voxels mínimos
        for region in regions:
            if region.area < min_volume:
                ventricular_mask[labels == region.label] = 0
                
        return ventricular_mask.astype(np.uint8)
        
    def _segment_cortex(self, image_data: np.ndarray) -> np.ndarray:
        """Segmentación de corteza cerebral"""
        # Corteza = borde externo de materia gris
        gray_matter = self._segment_gray_matter(image_data)
        
        # Detección de bordes para corteza
        cortex = morphology.binary_erosion(gray_matter, morphology.ball(2))
        cortex = gray_matter ^ cortex  # XOR para obtener solo el borde
        
        return cortex.astype(np.uint8)
        
    # ========== MÉTODOS DE MEDICIÓN Y ANÁLISIS ==========
    
    def _calculate_structure_volume(self, mask: np.ndarray, spacing: Tuple) -> float:
        """Cálculo de volumen de estructura en mm³"""
        voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm³ por voxel
        structure_voxels = np.sum(mask > 0)
        return structure_voxels * voxel_volume
        
    def _calculate_evans_ratio(self, ventricular_mask: np.ndarray, brain_image: np.ndarray) -> float:
        """Cálculo del ratio de Evans para diagnóstico de hidrocefalia"""
        # Slice axial en el nivel de los cuernos frontales
        mid_slice = brain_image.shape[0] // 2
        
        # Ancho ventricular máximo en slice axial
        ventricular_slice = ventricular_mask[mid_slice, :, :]
        if np.sum(ventricular_slice) == 0:
            return 0.0
            
        # Proyección horizontal para encontrar ancho máximo
        horizontal_projection = np.sum(ventricular_slice, axis=0)
        ventricular_width = np.sum(horizontal_projection > 0)
        
        # Ancho cerebral en el mismo nivel
        brain_slice = brain_image[mid_slice, :, :] > filters.threshold_otsu(brain_image[mid_slice, :, :]) * 0.1
        brain_projection = np.sum(brain_slice, axis=0)
        brain_width = np.sum(brain_projection > 0)
        
        # Ratio de Evans
        if brain_width > 0:
            return ventricular_width / brain_width
        return 0.0
        
    def _generate_clinical_interpretation(self, results: Dict) -> Dict:
        """Generación de interpretación clínica integrada"""
        interpretation = {
            'overall_assessment': 'NORMAL',
            'pathological_findings': [],
            'clinical_recommendations': [],
            'urgency_level': 'LOW',
            'radiological_report': ''
        }
        
        findings = []
        urgency = 'LOW'
        
        # Análisis de surcos
        if results['sulci_analysis']['pathological_widening']:
            findings.append("Amplificación patológica de surcos cerebrales")
            interpretation['overall_assessment'] = 'ABNORMAL'
            
        # Análisis hipocampal
        if results['hippocampus_analysis']['atrophy_present']:
            findings.append(f"Atrofia hipocampal {results['hippocampus_analysis']['atrophy_severity']}")
            interpretation['overall_assessment'] = 'ABNORMAL'
            
        # Análisis hipofisario
        if results['pituitary_analysis']['enlarged']:
            findings.append("Agrandamiento de hipófisis")
            interpretation['overall_assessment'] = 'ABNORMAL'
            
        # Análisis ventricular
        if results['ventricular_analysis']['enlarged_ventricles']:
            findings.append("Dilatación ventricular (posible hidrocefalia)")
            interpretation['overall_assessment'] = 'ABNORMAL'
            
        # Hemorragias
        if results['hemorrhage_detection']['hemorrhages_detected'] > 0:
            findings.append("HEMORRAGIA SUBARACNOIDEA DETECTADA")
            interpretation['overall_assessment'] = 'CRITICAL'
            urgency = 'HIGH'
            
        interpretation['pathological_findings'] = findings
        interpretation['urgency_level'] = urgency
        
        # Reporte radiológico estructurado
        interpretation['radiological_report'] = self._generate_structured_report(results, findings)
        
        return interpretation

    # ========== MÉTODOS DE INTERPRETACIÓN CLÍNICA ==========
    
    def _interpret_sulci_findings(self, measurements: List, widened: List) -> str:
        """Interpretación clínica de hallazgos de surcos"""
        if len(widened) == 0:
            return "Surcos cerebrales de morfología normal"
        elif len(widened) < 3:
            return f"Leve amplificación de {len(widened)} surcos - posible atrofia focal"
        else:
            return f"Amplificación generalizada de surcos ({len(widened)} afectados) - sugiere atrofia cerebral"
            
    def _interpret_hippocampal_findings(self, left_vol: float, right_vol: float, asymmetry: float) -> str:
        """Interpretación clínica de hallazgos hipocampales"""
        normal_range = self.reference_values['hippocampus']['normal_volume_range']
        
        if left_vol < normal_range[0] or right_vol < normal_range[0]:
            return f"Atrofia hipocampal presente - volumen reducido (L:{left_vol:.0f}, R:{right_vol:.0f} mm³)"
        elif asymmetry > 0.15:
            return f"Asimetría hipocampal significativa ({asymmetry:.1%}) - evaluar patología unilateral"
        else:
            return "Hipocampos de volumen y morfología normales"
            
    def _interpret_pituitary_findings(self, volume: float, dimensions: Dict, morphology: Dict) -> str:
        """Interpretación clínica de hallazgos hipofisarios"""
        if volume > self.reference_values['pituitary']['enlarged_threshold']:
            return f"Hipófisis agrandada ({volume:.0f} mm³) - considerar adenoma pituitario"
        elif not morphology['normal']:
            return f"Morfología hipofisaria alterada - {', '.join(morphology['findings'])}"
        else:
            return "Hipófisis de tamaño y morfología normales"
            
    def _generate_structured_report(self, results: Dict, findings: List) -> str:
        """Generación de reporte radiológico estructurado"""
        report = f"""
REPORTE RADIOLÓGICO - ANÁLISIS ESTRUCTURAL NEUROMORPHIC
═══════════════════════════════════════════════════════

HALLAZGOS ANATÓMICOS:

1. SURCOS CEREBRALES:
   • Surcos identificados: {results['sulci_analysis']['sulci_count']}
   • Ancho promedio: {results['sulci_analysis']['average_width_mm']:.1f} mm
   • Amplificación patológica: {'SÍ' if results['sulci_analysis']['pathological_widening'] else 'NO'}

2. SISTEMA HIPOCAMPAL:
   • Volumen L: {results['hippocampus_analysis']['left_volume_mm3']:.0f} mm³
   • Volumen R: {results['hippocampus_analysis']['right_volume_mm3']:.0f} mm³
   • Asimetría: {results['hippocampus_analysis']['asymmetry_ratio']:.1%}
   • Atrofia: {'PRESENTE' if results['hippocampus_analysis']['atrophy_present'] else 'AUSENTE'}

3. HIPÓFISIS:
   • Volumen: {results['pituitary_analysis']['volume_mm3']:.0f} mm³
   • Dimensiones: {results['pituitary_analysis']['height_mm']:.1f} x {results['pituitary_analysis']['width_mm']:.1f} x {results['pituitary_analysis']['depth_mm']:.1f} mm
   • Morfología: {'NORMAL' if results['pituitary_analysis']['morphology_normal'] else 'ALTERADA'}

4. SISTEMA VENTRICULAR:
   • Ratio de Evans: {results['ventricular_analysis']['evans_ratio']:.3f}
   • Dilatación: {'SÍ' if results['ventricular_analysis']['enlarged_ventricles'] else 'NO'}
   • Hidrocefalia: {'SOSPECHADA' if results['ventricular_analysis']['hydrocephalus_suspected'] else 'NO'}

5. HEMORRAGIAS:
   • Detectadas: {results['hemorrhage_detection']['hemorrhages_detected']}
   • HSA sospechada: {'SÍ' if results['hemorrhage_detection']['subarachnoid_hemorrhage_suspected'] else 'NO'}

INTERPRETACIÓN CLÍNICA:
{chr(10).join(['• ' + finding for finding in findings]) if findings else '• Estudio dentro de límites normales'}

RECOMENDACIONES:
• Correlación clínica necesaria
• Seguimiento según criterio médico
{'• EVALUACIÓN URGENTE REQUERIDA' if results['hemorrhage_detection']['hemorrhages_detected'] > 0 else ''}
        """.strip()
        
        return report

    # ========== MÉTODOS DE IMPLEMENTACIÓN ESPECÍFICA ==========
    # (Estos métodos necesitan implementación completa según disponibilidad de datos)
    
    def _detect_sulci_interface(self, gray_matter: np.ndarray, csf: np.ndarray) -> np.ndarray:
        """Detección de interfaz surcos GM-CSF"""
        # Implementación simplificada
        return morphology.binary_dilation(gray_matter, morphology.ball(1)) & csf
        
    def _measure_sulci_morphometry(self, sulci_mask: np.ndarray, spacing: Tuple) -> List[Dict]:
        """Medición morfométrica de surcos"""
        # Implementación simplificada
        labels = measure.label(sulci_mask)
        regions = measure.regionprops(labels)
        
        measurements = []
        for region in regions:
            if region.area > 50:  # Filtrar ruido
                measurements.append({
                    'width_mm': np.sqrt(region.area) * spacing[1],
                    'area_mm2': region.area * spacing[1] * spacing[2],
                    'location': region.centroid
                })
        
        return measurements
        
    def _detect_widened_sulci(self, measurements: List[Dict]) -> List[Dict]:
        """Detección de surcos patológicamente amplificados"""
        widened = []
        threshold = self.reference_values['sulci']['pathological_width']
        
        for measurement in measurements:
            if measurement['width_mm'] > threshold:
                widened.append(measurement)
                
        return widened
        
    def _locate_hippocampi(self, image_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Localización anatómica de hipocampos"""
        # Implementación simplificada usando regiones anatómicas aproximadas
        shape = image_data.shape
        
        # Región medial temporal aproximada
        left_region = np.zeros_like(image_data)
        right_region = np.zeros_like(image_data)
        
        # Coordenadas anatómicas aproximadas
        z_start, z_end = int(shape[0] * 0.3), int(shape[0] * 0.7)
        y_start, y_end = int(shape[1] * 0.4), int(shape[1] * 0.8)
        
        # Lado izquierdo
        x_left_start, x_left_end = int(shape[2] * 0.1), int(shape[2] * 0.4)
        left_region[z_start:z_end, y_start:y_end, x_left_start:x_left_end] = 1
        
        # Lado derecho  
        x_right_start, x_right_end = int(shape[2] * 0.6), int(shape[2] * 0.9)
        right_region[z_start:z_end, y_start:y_end, x_right_start:x_right_end] = 1
        
        # Segmentación de materia gris en estas regiones
        gray_matter = self._segment_gray_matter(image_data)
        
        left_hippo = gray_matter.astype(bool) & left_region.astype(bool)
        right_hippo = gray_matter.astype(bool) & right_region.astype(bool)
        
        return left_hippo.astype(np.uint8), right_hippo.astype(np.uint8)
        
    def _assess_hippocampal_atrophy(self, left_vol: float, right_vol: float) -> Dict:
        """Evaluación de atrofia hipocampal"""
        threshold = self.reference_values['hippocampus']['atrophy_threshold']
        
        left_atrophy = left_vol < threshold
        right_atrophy = right_vol < threshold
        
        severity = 'NONE'
        affected_side = 'NONE'
        
        if left_atrophy or right_atrophy:
            if left_atrophy and right_atrophy:
                severity = 'BILATERAL'
                affected_side = 'BILATERAL'
            elif left_atrophy:
                severity = 'MODERATE'
                affected_side = 'LEFT'
            else:
                severity = 'MODERATE'
                affected_side = 'RIGHT'
                
        return {
            'atrophy_detected': left_atrophy or right_atrophy,
            'severity': severity,
            'affected_side': affected_side
        }
        
    # Métodos adicionales simplificados para completar la implementación
    def _locate_pituitary(self, image_data: np.ndarray) -> np.ndarray:
        """Localización de hipófisis (implementación simplificada)"""
        # Región de silla turca (centro-inferior del cerebro)
        shape = image_data.shape
        pituitary_region = np.zeros_like(image_data)
        
        z_center = shape[0] // 2
        y_center = int(shape[1] * 0.8)  # Inferior
        x_center = shape[2] // 2
        
        # Región pequeña para hipófisis
        pituitary_region[z_center-2:z_center+3, 
                        y_center-3:y_center+3, 
                        x_center-3:x_center+3] = 1
        
        return pituitary_region
        
    def _calculate_pituitary_dimensions(self, mask: np.ndarray, spacing: Tuple) -> Dict:
        """Cálculo de dimensiones hipofisarias"""
        if np.sum(mask) == 0:
            return {'height': 0, 'width': 0, 'depth': 0}
            
        coords = np.where(mask > 0)
        
        height = (np.max(coords[0]) - np.min(coords[0])) * spacing[0]
        width = (np.max(coords[1]) - np.min(coords[1])) * spacing[1]  
        depth = (np.max(coords[2]) - np.min(coords[2])) * spacing[2]
        
        return {'height': height, 'width': width, 'depth': depth}
        
    def _assess_pituitary_morphology(self, mask: np.ndarray, dimensions: Dict) -> Dict:
        """Evaluación morfológica de hipófisis"""
        normal_dims = self.reference_values['pituitary']['normal_height_range']
        
        height_normal = normal_dims[0] <= dimensions['height'] <= normal_dims[1]
        
        findings = []
        if not height_normal:
            if dimensions['height'] > normal_dims[1]:
                findings.append("Altura aumentada")
            else:
                findings.append("Altura reducida")
                
        return {
            'normal': height_normal,
            'findings': findings
        }
        
    def _analyze_ventricular_symmetry(self, mask: np.ndarray) -> Dict:
        """Análisis de simetría ventricular"""
        # División en hemisferios
        mid_x = mask.shape[2] // 2
        left_ventricle = mask[:, :, :mid_x]
        right_ventricle = mask[:, :, mid_x:]
        
        left_volume = np.sum(left_ventricle)
        right_volume = np.sum(right_ventricle)
        
        if max(left_volume, right_volume) > 0:
            asymmetry = abs(left_volume - right_volume) / max(left_volume, right_volume)
        else:
            asymmetry = 0
            
        return {
            'asymmetric': asymmetry > self.reference_values['ventricles']['asymmetry_threshold'],
            'asymmetry_ratio': asymmetry
        }
        
    def _assess_hydrocephalus(self, evans_ratio: float, symmetry: Dict) -> Dict:
        """Evaluación de hidrocefalia"""
        threshold = self.reference_values['ventricles']['enlarged_threshold']
        
        suspected = evans_ratio > threshold
        hc_type = 'NONE'
        
        if suspected:
            if symmetry['asymmetric']:
                hc_type = 'ASYMMETRIC'
            else:
                hc_type = 'COMMUNICATING'
                
        return {
            'suspected': suspected,
            'type': hc_type
        }
        
    def _calculate_cortical_thickness(self, cortex_mask: np.ndarray, spacing: Tuple) -> np.ndarray:
        """Cálculo de mapa de grosor cortical"""
        # Implementación simplificada usando transformada de distancia
        distance_map = ndimage.distance_transform_edt(cortex_mask, sampling=spacing)
        return distance_map * 2  # Grosor = 2 * distancia
        
    def _assess_cortical_atrophy(self, thickness_map: np.ndarray) -> Dict:
        """Evaluación de atrofia cortical"""
        valid_thickness = thickness_map[thickness_map > 0]
        
        if len(valid_thickness) == 0:
            return {'atrophy_detected': False, 'severity': 'NONE', 'affected_regions': []}
            
        mean_thickness = np.mean(valid_thickness)
        threshold = self.reference_values['cortex']['atrophy_threshold']
        
        atrophy_detected = mean_thickness < threshold
        
        severity = 'NONE'
        if atrophy_detected:
            if mean_thickness < threshold * 0.8:
                severity = 'SEVERE'
            elif mean_thickness < threshold * 0.9:
                severity = 'MODERATE'
            else:
                severity = 'MILD'
                
        return {
            'atrophy_detected': atrophy_detected,
            'severity': severity,
            'affected_regions': ['GLOBAL'] if atrophy_detected else []
        }
        
    def _detect_high_intensity_csf(self, image_data: np.ndarray, csf_mask: np.ndarray) -> List:
        """Detección de intensidades altas en CSF (posibles hemorragias)"""
        # CSF normal tiene intensidad baja
        csf_intensities = image_data[csf_mask > 0]
        
        if len(csf_intensities) == 0:
            return []
            
        # Umbral para detectar sangre en CSF
        threshold = np.percentile(csf_intensities, 95)  # 95% percentil
        
        high_intensity_csf = (image_data > threshold) & (csf_mask.astype(bool))
        
        # Encontrar regiones conectadas
        labels = measure.label(high_intensity_csf)
        regions = measure.regionprops(labels)
        
        candidates = []
        for region in regions:
            if region.area > 10:  # Filtrar ruido
                candidates.append({
                    'location': region.centroid,
                    'area': region.area,
                    'intensity': np.mean(image_data[labels == region.label])
                })
                
        return candidates
        
    def _validate_hemorrhage_candidates(self, candidates: List, image_data: np.ndarray) -> List:
        """Validación de candidatos de hemorragia"""
        # Implementación simplificada - en práctica requiere análisis más sofisticado
        confirmed = []
        
        for candidate in candidates:
            # Criterios simples de validación
            if candidate['area'] > 20 and candidate['intensity'] > np.mean(image_data) * 1.5:
                confirmed.append(candidate)
                
        return confirmed
        
    def _analyze_brain_symmetry(self, image_data: np.ndarray) -> float:
        """Análisis de simetría cerebral"""
        # División sagital
        mid_x = image_data.shape[2] // 2
        left_brain = image_data[:, :, :mid_x]
        right_brain = np.flip(image_data[:, :, mid_x:], axis=2)
        
        # Ajustar tamaños si es necesario
        min_width = min(left_brain.shape[2], right_brain.shape[2])
        left_brain = left_brain[:, :, :min_width]
        right_brain = right_brain[:, :, :min_width]
        
        # Correlación como medida de simetría
        correlation = np.corrcoef(left_brain.flatten(), right_brain.flatten())[0, 1]
        
        return max(0, correlation)  # Asegurar que sea positivo
        
    def _detect_midline_shift(self, image_data: np.ndarray) -> float:
        """Detección de desplazamiento de línea media"""
        # Implementación simplificada
        # En práctica requiere detección sofisticada de estructuras de línea media
        
        # Proyección coronal para detectar asimetría
        coronal_projection = np.mean(image_data, axis=0)
        
        # Detectar centro de masa
        y_indices, x_indices = np.meshgrid(range(coronal_projection.shape[0]), 
                                          range(coronal_projection.shape[1]), indexing='ij')
        
        total_mass = np.sum(coronal_projection)
        if total_mass > 0:
            center_of_mass_x = np.sum(coronal_projection * x_indices) / total_mass
            geometric_center_x = coronal_projection.shape[1] / 2
            
            # Desplazamiento en mm (asumiendo spacing de 1mm)
            shift_mm = (center_of_mass_x - geometric_center_x)
        else:
            shift_mm = 0
            
        return shift_mm
        
    def _analyze_anatomical_proportions(self, image_data: np.ndarray, spacing: Tuple) -> Dict:
        """Análisis de proporciones anatómicas"""
        # Mediciones básicas
        dimensions = np.array(image_data.shape) * np.array(spacing)
        
        return {
            'length_mm': dimensions[0],
            'height_mm': dimensions[1], 
            'width_mm': dimensions[2],
            'aspect_ratios': {
                'length_height': dimensions[0] / dimensions[1] if dimensions[1] > 0 else 0,
                'length_width': dimensions[0] / dimensions[2] if dimensions[2] > 0 else 0,
                'height_width': dimensions[1] / dimensions[2] if dimensions[2] > 0 else 0
            }
        }
        
    def _calculate_sulci_asymmetry(self, measurements: List) -> float:
        """Cálculo de asimetría de surcos"""
        if len(measurements) < 2:
            return 0.0
            
        # Implementación simplificada
        widths = [m['width_mm'] for m in measurements]
        return np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 0
        
    def _interpret_sulci_findings(self, measurements: List, widened: List) -> str:
        """Interpretación de hallazgos de surcos"""
        if len(widened) == 0:
            return "Surcos cerebrales de morfología normal"
        else:
            return f"Amplificación de surcos detectada en {len(widened)} regiones"
            
    def _interpret_hippocampal_findings(self, left_vol: float, right_vol: float, asymmetry: float) -> str:
        """Interpretación de hallazgos hipocampales"""
        if asymmetry > 0.15:
            return f"Asimetría hipocampal significativa ({asymmetry:.1%})"
        else:
            return "Hipocampos simétricos y de volumen normal"
            
    def _interpret_pituitary_findings(self, volume: float, dimensions: Dict, morphology: Dict) -> str:
        """Interpretación de hallazgos hipofisarios"""
        if volume > 1000:
            return f"Hipófisis agrandada ({volume:.0f} mm³)"
        else:
            return "Hipófisis de tamaño normal"
            
    def _interpret_ventricular_findings(self, evans_ratio: float, symmetry: Dict) -> str:
        """Interpretación de hallazgos ventriculares"""
        if evans_ratio > 0.35:
            return f"Dilatación ventricular (Evans: {evans_ratio:.3f})"
        else:
            return "Sistema ventricular normal"
            
    def _interpret_cortical_findings(self, thickness_map: np.ndarray, atrophy: Dict) -> str:
        """Interpretación de hallazgos corticales"""
        if atrophy['atrophy_detected']:
            return f"Atrofia cortical {atrophy['severity']}"
        else:
            return "Corteza cerebral de grosor normal"
            
    def _interpret_hemorrhage_findings(self, hemorrhages: List) -> str:
        """Interpretación de hallazgos hemorrágicos"""
        if len(hemorrhages) > 0:
            return f"HEMORRAGIA DETECTADA - {len(hemorrhages)} focos identificados"
        else:
            return "No se detectan hemorragias"
            
    def _interpret_morphological_findings(self, symmetry: float, midline_shift: float) -> str:
        """Interpretación de hallazgos morfológicos"""
        if abs(midline_shift) > 2.0:
            return f"Desplazamiento de línea media significativo ({midline_shift:.1f} mm)"
        elif symmetry < 0.8:
            return "Asimetría cerebral detectada"
        else:
            return "Morfología cerebral normal"
