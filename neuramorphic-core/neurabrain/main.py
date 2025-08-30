#!/usr/bin/env python3
"""
Neuromorphic Medical AI Main Execution Script
Professional medical image analysis with neuromorphic AI
"""

import sys
import os
from pathlib import Path
import argparse
import logging

sys.path.append(str(Path(__file__).parent))

from core.medical_engine import MedicalAnalysisEngine

def setup_environment():
    """Setup environment and check dependencies"""
    
    try:
        import torch
        import numpy
        import nibabel
        import matplotlib
        import scipy
        import sklearn
        
        print("All required dependencies available")
        return True
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='Neuromorphic Medical AI Analysis System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--images', 
        nargs='+', 
        help='Medical image files to analyze (.nii.gz format)'
    )
    
    parser.add_argument(
        '--output-dir', 
        default='analysis_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--auto-scan',
        action='store_true',
        help='Automatically scan current directory for .nii.gz files'
    )
    
    parser.add_argument(
        '--enhanced-mode',
        action='store_true',
        help='Enable enhanced slice-by-slice analysis'
    )
    
    parser.add_argument(
        '--slice-axis',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='Axis for slice extraction (0=sagittal, 1=coronal, 2=axial)'
    )
    
    args = parser.parse_args()
    
    if not setup_environment():
        sys.exit(1)
    
    log_level = getattr(logging, args.log_level)
    
    analysis_engine = MedicalAnalysisEngine(
        output_dir=args.output_dir,
        log_level=log_level
    )
    
    image_files = []
    
    if args.auto_scan:
        current_dir = Path('.')
        image_files = list(current_dir.glob('*.nii.gz'))
        
        if not image_files:
            print("No .nii.gz files found in current directory")
            sys.exit(1)
            
        print(f"Found {len(image_files)} medical image files:")
        for img_file in image_files:
            print(f"  - {img_file}")
    
    elif args.images:
        for img_path in args.images:
            img_file = Path(img_path)
            if not img_file.exists():
                print(f"Error: File not found: {img_file}")
                sys.exit(1)
            if not str(img_file).endswith('.nii.gz'):
                print(f"Error: Unsupported format: {img_file}")
                print("Only .nii.gz files are supported")
                sys.exit(1)
            image_files.append(img_file)
    
    else:
        print("Error: No images specified")
        print("Use --images <file1> <file2> ... or --auto-scan")
        sys.exit(1)
    
    print("\nStarting Neuromorphic Medical AI Analysis")
    print("=" * 50)
    print(f"Images to analyze: {len(image_files)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Log level: {args.log_level}")
    print(f"Enhanced mode: {'ENABLED' if args.enhanced_mode else 'DISABLED'}")
    if args.enhanced_mode:
        axis_names = {0: 'Sagittal', 1: 'Coronal', 2: 'Axial'}
        print(f"Slice axis: {args.slice_axis} ({axis_names[args.slice_axis]})")
    print("=" * 50)
    
    try:
        if len(image_files) == 1:
            if args.enhanced_mode:
                print("🧠 Running ENHANCED slice-by-slice analysis...")
                result = analysis_engine.analyze_single_image_enhanced(
                    str(image_files[0]),
                    enable_slice_analysis=True,
                    slice_axis=args.slice_axis
                )
                print(f"\n✅ Enhanced analysis completed")
                print(f"📄 Enhanced report: {result['enhanced_report_path']}")
                print(f"📊 Standard report: {result['standard_analysis']['report_path']}")
                print(f"🖼️  Visualization: {result['standard_analysis']['visualization_path']}")
                
                # Print slice-by-slice summary
                enhanced_features = result.get('enhanced_features', {})
                if enhanced_features:
                    total_slices = enhanced_features.get('total_slices_analyzed', 0)
                    video_path = enhanced_features.get('video_generated')
                    print(f"🎬 Total slices processed: {total_slices}")
                    if video_path:
                        print(f"🎥 Video generated: {Path(video_path).name}")
                
            else:
                print("🔄 Running standard analysis...")
                result = analysis_engine.analyze_single_image(str(image_files[0]))
                print(f"\n✅ Single image analysis completed")
                print(f"📄 Report: {result['report_path']}")
                print(f"🖼️  Visualization: {result['visualization_path']}")
        else:
            if args.enhanced_mode:
                print("🧠 Running ENHANCED batch analysis...")
                results = []
                for i, image_file in enumerate(image_files, 1):
                    print(f"\n🔬 Processing {i}/{len(image_files)}: {image_file.name}")
                    result = analysis_engine.analyze_single_image_enhanced(
                        str(image_file),
                        enable_slice_analysis=True,
                        slice_axis=args.slice_axis
                    )
                    results.append(result)
                
                print(f"\n✅ Enhanced batch analysis completed")
                print(f"🎬 Successfully processed: {len(results)}/{len(image_files)} images with slice-by-slice analysis")
            else:
                print("🔄 Running standard batch analysis...")
                results = analysis_engine.analyze_multiple_images([str(f) for f in image_files])
                print(f"\n✅ Batch analysis completed")
                print(f"📊 Successfully processed: {len(results)}/{len(image_files)} images")
        
        summary = analysis_engine.get_analysis_summary()
        print(f"\nAnalysis Summary:")
        print(f"  Total analyses: {summary['total_analyses']}")
        print(f"  Average quality score: {summary['average_quality_score']:.2f}")
        print(f"  Neuromorphic core: {summary['neuromorphic_core_type']}")
        print(f"  Results directory: {summary['output_directory']}")
        print(f"  Log file: {summary['log_file']}")
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
