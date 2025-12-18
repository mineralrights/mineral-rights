#!/bin/bash

echo "🧹 Cleaning up mineral-rights codebase..."

# Remove test and evaluation scripts
echo "Removing old test scripts..."
rm -f test_oil_gas_classifier.py
rm -f test_false_positive_fixes.py
rm -f analyze_false_positives.py
rm -f pattern_analysis.py
rm -f test_api.py
rm -f monitor_evaluation.py
rm -f evaluate_optimized_no_reservations.py
rm -f evaluate_full_dataset.py

# Remove Flask app (keeping Streamlit)
echo "Removing Flask app and templates..."
rm -f app.py
rm -rf templates/

# Remove experiments directory
echo "Removing experiments directory..."
rm -rf experiments/

# Remove evaluation results directories
echo "Removing evaluation results..."
rm -rf full_evaluation_results/
rm -rf optimized_evaluation_results/
rm -rf test_results/
rm -rf batch_results/
rm -rf ocr_results/

# Remove plotting and visualization files
echo "Removing plotting scripts and images..."
rm -f plot_optimization_results.py
rm -f *.png

# Remove presentation files
echo "Removing presentation files..."
rm -f presentation_summary.py
rm -f PRESENTATION_SUMMARY.md
rm -f OIL_GAS_UPDATES.md
rm -f executive_summary.md
rm -f technical_optimization_report.md

# Remove batch processing (uncomment if not needed)
# rm -f batch_processor.py
rm -f false_positive_documents.txt

# Remove system files
echo "Removing system files..."
find . -name ".DS_Store" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Clean up old results (keep recent ones)
echo "Cleaning old result files..."
cd results/ 2>/dev/null && ls -t *.json | tail -n +6 | xargs rm -f 2>/dev/null
cd ..

echo "✅ Cleanup complete!"
echo ""
echo "📊 Space saved:"
du -sh . 2>/dev/null || echo "Unable to calculate disk usage"
echo ""
echo "🎉 Streamlit app retained, Flask app removed!"
echo ""
echo "📋 Remaining core files:"
echo "   ✅ streamlit_app.py - Your web interface"
echo "   ✅ document_classifier.py - AI classification engine"
echo "   ✅ demo.py - Interactive demo"
echo "   ✅ requirements.txt - Dependencies"
echo "   ✅ data/ - Training data"
echo "   ✅ Deployment files (render.yaml, build.sh, etc.)"
echo ""
echo "🤔 Optional decision:"
echo "   - batch_processor.py (uncomment line 42 to remove if not needed)" 