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
    print("=" * 50)
    
    try:
        if len(image_files) == 1:
            result = analysis_engine.analyze_single_image(str(image_files[0]))
            print(f"\nSingle image analysis completed")
            print(f"Report: {result['report_path']}")
            print(f"Visualization: {result['visualization_path']}")
        else:
            results = analysis_engine.analyze_multiple_images([str(f) for f in image_files])
            print(f"\nBatch analysis completed")
            print(f"Successfully processed: {len(results)}/{len(image_files)} images")
        
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
