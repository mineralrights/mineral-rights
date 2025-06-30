#!/bin/bash

# Mineral Rights Classification Demo Runner
# ========================================

echo "🏛️  Mineral Rights Classification Demo"
echo "======================================"
echo ""

# Check if Python script exists
if [ ! -f "demo.py" ]; then
    echo "❌ Error: demo.py not found in current directory"
    exit 1
fi

# Check if document classifier exists
if [ ! -f "document_classifier.py" ]; then
    echo "❌ Error: document_classifier.py not found in current directory"
    exit 1
fi

# Function to run demo with a specific document
run_demo() {
    local doc_path="$1"
    local doc_type="$2"
    
    echo "📄 Running demo with: $doc_path"
    echo "📋 Expected result: $doc_type"
    echo "⏳ Processing..."
    echo ""
    
    python3 demo.py "$doc_path"
    
    echo ""
    echo "─────────────────────────────────────────────────────────────"
    echo ""
}

# Show menu if no arguments provided
if [ $# -eq 0 ]; then
    echo "Choose a demo option:"
    echo ""
    echo "1) Document WITH mineral rights reservations"
    echo "2) Document WITHOUT mineral rights reservations"
    echo "3) Custom document path"
    echo "4) Run both examples (comparison)"
    echo ""
    read -p "Enter your choice (1-4): " choice
    
    case $choice in
        1)
            echo ""
            echo "🎯 Demo: Document WITH reservations"
            run_demo "data/reservs/Indiana Co. PA DB 550_322.pdf" "HAS RESERVATIONS"
            ;;
        2)
            echo ""
            echo "✅ Demo: Document WITHOUT reservations"
            run_demo "data/no-reservs/Butler DB 1895_80 - 4.23.2025.pdf" "NO RESERVATIONS"
            ;;
        3)
            echo ""
            read -p "Enter PDF path: " custom_path
            if [ -f "$custom_path" ]; then
                run_demo "$custom_path" "UNKNOWN"
            else
                echo "❌ File not found: $custom_path"
            fi
            ;;
        4)
            echo ""
            echo "🔄 Running comparison demo..."
            echo ""
            echo "═══════════════════════════════════════════════════════════"
            echo "📋 DEMO 1: Document WITH mineral rights reservations"
            echo "═══════════════════════════════════════════════════════════"
            run_demo "data/reservs/Indiana Co. PA DB 550_322.pdf" "HAS RESERVATIONS"
            
            echo ""
            echo "═══════════════════════════════════════════════════════════"
            echo "📋 DEMO 2: Document WITHOUT mineral rights reservations"
            echo "═══════════════════════════════════════════════════════════"
            run_demo "data/no-reservs/Butler DB 1895_80 - 4.23.2025.pdf" "NO RESERVATIONS"
            ;;
        *)
            echo "❌ Invalid choice"
            exit 1
            ;;
    esac
else
    # Run with provided arguments
    run_demo "$1" "UNKNOWN"
fi

echo "🎉 Demo completed!"
echo ""
echo "📁 Check the 'demo_results' folder for detailed output files:"
echo "   • demo_result_*.json (complete analysis data)"
echo "   • demo_text_*.txt (extracted text)"
echo "   • demo_summary_*.txt (human-readable report)" 