#!/usr/bin/env python3
"""
Example: Batch Processing with CSV Export
========================================

This script demonstrates how to use the mineral rights classification system
for batch processing multiple documents and exporting results to CSV format,
similar to the Streamlit web application.
"""

import os
import pandas as pd
import csv
from datetime import datetime
from pathlib import Path
from document_classifier import DocumentProcessor

def process_documents_to_csv(pdf_paths, output_csv_path=None):
    """
    Process multiple PDF documents and export results to CSV
    
    Args:
        pdf_paths: List of paths to PDF files
        output_csv_path: Optional path for output CSV file
        
    Returns:
        DataFrame with results
    """
    
    # Initialize processor
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    processor = DocumentProcessor(api_key=api_key)
    
    results = []
    
    print(f"Processing {len(pdf_paths)} documents...")
    
    for i, pdf_path in enumerate(pdf_paths):
        try:
            print(f"Processing {pdf_path} ({i+1}/{len(pdf_paths)})...")
            
            # Process document
            result = processor.process_document(
                pdf_path,
                max_samples=5,
                confidence_threshold=0.7
            )
            
            # Extract detailed information for CSV
            classification = result['classification']
            confidence = result['confidence']
            
            # Get confidence level
            if confidence >= 0.8:
                confidence_level = "HIGH"
            elif confidence >= 0.6:
                confidence_level = "MEDIUM"
            else:
                confidence_level = "LOW"
            
            # Get the best reasoning from detailed samples
            best_reasoning = ""
            if 'detailed_samples' in result and result['detailed_samples']:
                best_sample = max(result['detailed_samples'], 
                                key=lambda x: x.get('confidence_score', 0))
                best_reasoning = best_sample.get('reasoning', '')
            
            # Get recommendation
            if classification == 0:
                if confidence >= 0.8:
                    recommendation = "This document appears to be a clean transfer without mineral rights reservations. You can proceed with confidence, but always consult with a qualified attorney for final verification."
                else:
                    recommendation = "While our analysis suggests no mineral rights reservations, the confidence level warrants additional review. Consider having a legal professional examine the document for complete certainty."
            else:
                if confidence >= 0.8:
                    recommendation = "Strong evidence of mineral rights reservations detected. This document likely contains clauses that reserve mineral rights to the grantor or previous parties. Legal review is strongly recommended before proceeding."
                else:
                    recommendation = "Potential mineral rights reservations detected, but with moderate confidence. Professional legal review is essential to determine the exact nature and scope of any reservations."
            
            # Calculate additional metrics
            votes = result['votes']
            total_votes = sum(votes.values())
            no_reservation_votes = votes.get(0, 0)
            has_reservation_votes = votes.get(1, 0)
            vote_ratio = has_reservation_votes / total_votes if total_votes > 0 else 0
            
            # Get file size
            file_size = Path(pdf_path).stat().st_size
            
            # Prepare result row
            result_row = {
                'filename': Path(pdf_path).name,
                'file_path': pdf_path,
                'file_size_bytes': file_size,
                'processing_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'classification': 'Has Mineral Rights Reservations' if classification == 1 else 'No Mineral Rights Reservations',
                'classification_numeric': classification,
                'confidence_score': round(confidence, 4),
                'confidence_level': confidence_level,
                'recommendation': recommendation,
                'llm_explanation': best_reasoning,
                'pages_processed': result['pages_processed'],
                'samples_used': result['samples_used'],
                'total_votes': total_votes,
                'no_reservation_votes': no_reservation_votes,
                'has_reservation_votes': has_reservation_votes,
                'vote_ratio_reservations': round(vote_ratio, 4),
                'early_stopped': result.get('early_stopped', False),
                'text_characters_analyzed': len(result.get('ocr_text', '')),
                'processing_status': 'Success'
            }
            
            results.append(result_row)
            print(f"✅ {Path(pdf_path).name}: {result_row['classification']} (Confidence: {confidence:.1%})")
            
        except Exception as e:
            # Add error row
            error_row = {
                'filename': Path(pdf_path).name,
                'file_path': pdf_path,
                'file_size_bytes': Path(pdf_path).stat().st_size if Path(pdf_path).exists() else 0,
                'processing_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'classification': 'ERROR',
                'classification_numeric': -1,
                'confidence_score': 0.0,
                'confidence_level': 'ERROR',
                'recommendation': f'Processing failed: {str(e)}',
                'llm_explanation': f'Error occurred during processing: {str(e)}',
                'pages_processed': 0,
                'samples_used': 0,
                'total_votes': 0,
                'no_reservation_votes': 0,
                'has_reservation_votes': 0,
                'vote_ratio_reservations': 0.0,
                'early_stopped': False,
                'text_characters_analyzed': 0,
                'processing_status': f'Error: {str(e)}'
            }
            results.append(error_row)
            print(f"❌ {Path(pdf_path).name}: Processing failed - {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Generate output filename if not provided
    if output_csv_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_csv_path = f'mineral_rights_analysis_{timestamp}.csv'
    
    # Export to CSV
    df.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_ALL)
    
    print(f"\n📊 Processing Complete!")
    print(f"📋 Results exported to: {output_csv_path}")
    
    # Print summary statistics
    total_docs = len(df)
    successful = len(df[df['processing_status'] == 'Success'])
    has_reservations = len(df[df['classification_numeric'] == 1])
    high_confidence = len(df[df['confidence_level'] == 'HIGH'])
    
    print(f"\n📈 Summary Statistics:")
    print(f"   Total Documents: {total_docs}")
    print(f"   Successfully Processed: {successful}")
    print(f"   With Reservations: {has_reservations}")
    print(f"   High Confidence: {high_confidence}")
    
    return df

def main():
    """Example usage of batch processing"""
    
    # Example PDF paths - replace with your actual paths
    pdf_paths = [
        "data/reservs/Indiana Co. PA DB 550_322.pdf",
        "data/no-reservs/Adams Co. PA DB 2011_1.pdf",
        # Add more paths as needed
    ]
    
    # Filter to only existing files
    existing_paths = [path for path in pdf_paths if Path(path).exists()]
    
    if not existing_paths:
        print("❌ No valid PDF files found. Please check the paths in the script.")
        print("   Update the 'pdf_paths' list with actual file paths.")
        return
    
    print(f"Found {len(existing_paths)} valid PDF files")
    
    try:
        # Process documents and export to CSV
        results_df = process_documents_to_csv(existing_paths)
        
        # Display preview of results
        print(f"\n📋 Results Preview:")
        print(results_df[['filename', 'classification', 'confidence_level', 'confidence_score']].to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure ANTHROPIC_API_KEY is set in your environment")

if __name__ == "__main__":
    main() 