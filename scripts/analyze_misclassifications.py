#!/usr/bin/env python3
"""
Script to analyze misclassifications in mineral rights classification evaluation results.
Helps identify patterns that suggest the classification logic needs adjustment.
"""

import pandas as pd
import sys

def analyze_misclassifications(csv_file_path):
    """
    Analyze all misclassifications from the evaluation results.
    
    Args:
        csv_file_path (str): Path to the CSV file with evaluation results
    
    Returns:
        tuple: (false_negatives, false_positives, all_misclassified)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Identify misclassifications
        false_negatives = df[(df['true_label'] == 1) & (df['predicted_label'] == 0)]
        false_positives = df[(df['true_label'] == 0) & (df['predicted_label'] == 1)]
        all_misclassified = df[df['correct'] == False]
        
        return false_negatives, false_positives, all_misclassified
    
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return None, None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, None

def print_analysis(false_negatives, false_positives, all_misclassified):
    """Print detailed analysis of misclassifications."""
    
    print("🔍 MISCLASSIFICATION ANALYSIS")
    print("=" * 80)
    
    # False Negatives Analysis
    if false_negatives is not None and not false_negatives.empty:
        print(f"\n❌ FALSE NEGATIVES: {len(false_negatives)} documents")
        print("These actually HAVE mineral rights reservations but were classified as NO reservations:")
        print("-" * 60)
        
        for idx, row in false_negatives.iterrows():
            print(f"📄 {row['file_name']}")
            print(f"   Samples: {row['samples_used']} | Pages: {row['pages_processed']} | Confidence: {row['confidence']}")
            print()
        
        print("💡 ANALYSIS FOR FALSE NEGATIVES:")
        print("   These documents likely contain language like:")
        print("   - 'EXCEPTING AND RESERVING minerals'")
        print("   - 'RESERVING coal and mining rights'") 
        print("   - 'mineral rights' (without explicitly saying 'oil and gas')")
        print("   - General mineral language that should include oil/gas")
        print()
    
    # False Positives Analysis  
    if false_positives is not None and not false_positives.empty:
        print(f"\n❌ FALSE POSITIVES: {len(false_positives)} documents")
        print("These actually have NO mineral rights reservations but were classified as HAVING reservations:")
        print("-" * 60)
        
        for idx, row in false_positives.iterrows():
            print(f"📄 {row['file_name']}")
            print(f"   Samples: {row['samples_used']} | Pages: {row['pages_processed']} | Confidence: {row['confidence']}")
            print()
        
        print("💡 ANALYSIS FOR FALSE POSITIVES:")
        print("   These documents might contain misleading language like:")
        print("   - References to mineral rights in context (but not actually reserving them)")
        print("   - Historical mentions of previous reservations")
        print("   - Boilerplate language that doesn't actually reserve rights")
        print()

def suggest_prompt_improvements():
    """Suggest improvements to the classification prompt."""
    print("🚀 SUGGESTED PROMPT IMPROVEMENTS")
    print("=" * 80)
    print()
    print("Based on the Harrison document example, consider updating your classification logic:")
    print()
    print("CURRENT LOGIC (appears to be):")
    print("   ✅ Has reservations: Documents explicitly mentioning 'oil and gas' reservations")
    print("   ❌ No reservations: Everything else")
    print()
    print("SUGGESTED NEW LOGIC:")
    print("   ✅ Has reservations: Documents that reserve ANY of the following:")
    print("      • Oil and gas rights")
    print("      • Mineral rights (general)")
    print("      • Mining rights")
    print("      • Any minerals (unless ONLY coal is specified)")
    print("      • Subsurface rights")
    print()
    print("   ❌ No reservations: Documents that:")
    print("      • Have no mineral reservations at all")
    print("      • ONLY reserve coal specifically (and nothing else)")
    print("      • Only mention minerals in passing/historical context")
    print()
    print("EXAMPLE LANGUAGE THAT SHOULD BE 'HAS RESERVATIONS':")
    print('   • "EXCEPTING AND RESERVING the minerals"')
    print('   • "RESERVING unto grantors one-half of income from any minerals"')
    print('   • "EXCEPTING coal and mining rights" (includes other minerals)')
    print('   • "RESERVING all subsurface rights"')
    print()
    print("EXAMPLE LANGUAGE THAT SHOULD BE 'NO RESERVATIONS':")
    print('   • "EXCEPTING AND RESERVING ONLY the coal"')
    print('   • "Subject to coal rights previously reserved" (historical reference)')
    print('   • No mention of any mineral reservations')

def main():
    # Default CSV file path
    csv_file = "evaluation_results/oil_gas_results_20250629_162344.csv"
    
    # Allow command line argument for different file
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    print(f"Analyzing evaluation results from: {csv_file}")
    
    # Analyze misclassifications
    false_negatives, false_positives, all_misclassified = analyze_misclassifications(csv_file)
    
    if all_misclassified is None:
        return
    
    # Print analysis
    print_analysis(false_negatives, false_positives, all_misclassified)
    
    # Suggest improvements
    suggest_prompt_improvements()
    
    # Summary statistics
    if all_misclassified is not None:
        df = pd.read_csv(csv_file)
        total_docs = df.shape[0]
        accuracy = (total_docs - len(all_misclassified)) / total_docs * 100
        
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"   Total documents: {total_docs}")
        print(f"   Correctly classified: {total_docs - len(all_misclassified)}")
        print(f"   Misclassified: {len(all_misclassified)}")
        print(f"   Current accuracy: {accuracy:.1f}%")
        print(f"   False negatives: {len(false_negatives) if false_negatives is not None else 0}")
        print(f"   False positives: {len(false_positives) if false_positives is not None else 0}")

if __name__ == "__main__":
    main() 