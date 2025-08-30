#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Neuromorphic Medical AI - Cleanup Utility"
echo "=========================================="

echo "Cleaning up analysis results and temporary files..."

ITEMS_CLEANED=0

if ls analysis_results_* 1> /dev/null 2>&1; then
    echo "Removing analysis result directories..."
    rm -rf analysis_results_*
    ITEMS_CLEANED=$((ITEMS_CLEANED + 1))
fi

if ls analysis_results 1> /dev/null 2>&1; then
    echo "Removing default analysis results..."
    rm -rf analysis_results
    ITEMS_CLEANED=$((ITEMS_CLEANED + 1))
fi

if ls *.png 1> /dev/null 2>&1; then
    echo "Removing generated images..."
    rm -f *.png
    ITEMS_CLEANED=$((ITEMS_CLEANED + 1))
fi

if ls *.json 1> /dev/null 2>&1; then
    echo "Removing generated reports..."
    rm -f *.json
    ITEMS_CLEANED=$((ITEMS_CLEANED + 1))
fi

if ls *.log 1> /dev/null 2>&1; then
    echo "Removing log files..."
    rm -f *.log
    ITEMS_CLEANED=$((ITEMS_CLEANED + 1))
fi

if ls temp_* 1> /dev/null 2>&1; then
    echo "Removing temporary files..."
    rm -rf temp_*
    ITEMS_CLEANED=$((ITEMS_CLEANED + 1))
fi

if [ -d "__pycache__" ]; then
    echo "Removing Python cache..."
    rm -rf __pycache__
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    ITEMS_CLEANED=$((ITEMS_CLEANED + 1))
fi

if [ -d "core/__pycache__" ]; then
    echo "Removing core module cache..."
    rm -rf core/__pycache__
    ITEMS_CLEANED=$((ITEMS_CLEANED + 1))
fi

if ls .DS_Store 1> /dev/null 2>&1; then
    echo "Removing system files..."
    rm -f .DS_Store
    find . -name ".DS_Store" -delete
    ITEMS_CLEANED=$((ITEMS_CLEANED + 1))
fi

echo ""
if [ $ITEMS_CLEANED -gt 0 ]; then
    echo "Cleanup completed successfully!"
    echo "Removed $ITEMS_CLEANED category(ies) of files/directories"
else
    echo "No files to clean - workspace already clean"
fi

echo ""
echo "Remaining files in workspace:"
echo "$(find . -maxdepth 1 -type f -name "*.py" -o -name "*.sh" -o -name "*.txt" -o -name "*.md" -o -name "*.nii.gz" | sort)"
echo ""
echo "Workspace ready for fresh analysis"
echo "=========================================="
