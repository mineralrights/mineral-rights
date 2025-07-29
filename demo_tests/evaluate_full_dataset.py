#!/usr/bin/env python3
"""
Full Dataset Evaluation
=======================

Comprehensive evaluation of the improved pipeline on the entire dataset.
Tests the chunk-by-chunk early stopping with improved prompt classification.
"""

import json
import time
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from document_classifier import DocumentProcessor

def evaluate_full_dataset(data_dirs: List[str], output_dir: str = "full_evaluation_results"):
    """Evaluate the pipeline on the complete dataset"""
    
    processor = DocumentProcessor()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(" FULL DATASET EVALUATION")
    print("=" * 60)
    print("Testing pipeline with:")
    print("  * Chunk-by-chunk early stopping")
    print("  * Enhanced prompt for false positive reduction")
    print("  * 8K token limit per page")
    print("=" * 60)
    
    # Collect all PDF files with their expected labels
    pdf_files = []
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if data_path.exists():
            pdfs = list(data_path.glob("*.pdf"))
            # Expected label: 1 if in 'reservs' folder, 0 if in 'no-reservs' folder
            # Fix: Check for exact folder name match, not substring
            folder_name = data_path.name
            if folder_name == 'reservs':
                expected_label = 1
            elif folder_name == 'no-reservs':
                expected_label = 0
            else:
                print(f"Warning: Unknown folder '{folder_name}', skipping...")
                continue
            pdf_files.extend([(pdf, expected_label, folder_name) for pdf in pdfs])
    
    print(f" DATASET OVERVIEW:")
    print(f"  Total documents: {len(pdf_files)}")
    print(f"  Expected reservations: {sum(1 for _, label, _ in pdf_files if label == 1)}")
    print(f"  Expected no reservations: {sum(1 for _, label, _ in pdf_files if label == 0)}")
    print()
    
    # Process each document
    all_results = []
    start_time = time.time()
    
    for i, (pdf_path, expected_label, folder_name) in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        print(f"Expected: {expected_label} ({'Has Reservations' if expected_label == 1 else 'No Reservations'})")
        
        try:
            # Process with improved pipeline
            result = processor.process_document(
                str(pdf_path),
                max_samples=5,  # Reasonable number of samples per chunk
                confidence_threshold=0.6,
                # Uses sequential_early_stop by default
            )
            
            predicted_label = result['classification']
            confidence = result['confidence']
            is_correct = (predicted_label == expected_label)
            
            # Calculate efficiency metrics
            pages_processed = result['pages_processed']
            total_pages = result['total_pages_in_document']
            efficiency_ratio = pages_processed / total_pages
            stopped_early = result.get('stopped_at_chunk') is not None
            
            print(f"  Predicted: {predicted_label} ({'Has Reservations' if predicted_label == 1 else 'No Reservations'})")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Pages: {pages_processed}/{total_pages} ({efficiency_ratio:.2f})")
            print(f"  Early stop: {'Yes' if stopped_early else 'No'}")
            print(f"  Result: {' CORRECT' if is_correct else ' INCORRECT'}")
            
            # Store comprehensive result
            result_data = {
                'document_name': pdf_path.name,
                'document_path': str(pdf_path),
                'folder_name': folder_name,
                'expected_label': expected_label,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'correct_prediction': is_correct,
                'pages_processed': pages_processed,
                'total_pages_in_document': total_pages,
                'efficiency_ratio': efficiency_ratio,
                'stopped_early': stopped_early,
                'stopped_at_page': result.get('stopped_at_chunk'),
                'samples_used': result['samples_used'],
                'ocr_text_length': result['ocr_text_length'],
                'processing_time': None,  # Will be calculated later
                'chunk_analysis': result.get('chunk_analysis', [])
            }
            
            all_results.append(result_data)
            
            # Save individual result
            with open(output_path / f"{pdf_path.stem}_result.json", "w") as f:
                json.dump({
                    **result_data,
                    'detailed_samples': result.get('detailed_samples', []),
                    'ocr_text': result['ocr_text']
                }, f, indent=2)
                
        except Exception as e:
            print(f"   ERROR: {e}")
            # Store error result
            all_results.append({
                'document_name': pdf_path.name,
                'document_path': str(pdf_path),
                'folder_name': folder_name,
                'expected_label': expected_label,
                'predicted_label': None,
                'confidence': 0.0,
                'correct_prediction': False,
                'error': str(e),
                'pages_processed': 0,
                'total_pages_in_document': 0,
                'efficiency_ratio': 0.0,
                'stopped_early': False,
                'stopped_at_page': None,
                'samples_used': 0,
                'ocr_text_length': 0,
                'processing_time': None
            })
            continue
    
    total_time = time.time() - start_time
    
    # Generate comprehensive evaluation report
    generate_comprehensive_report(all_results, output_path, total_time)
    
    return all_results

def generate_comprehensive_report(results: List[Dict], output_dir: Path, total_time: float):
    """Generate comprehensive evaluation report with detailed metrics"""
    
    # Filter out error results for accuracy calculations
    valid_results = [r for r in results if r['predicted_label'] is not None]
    error_results = [r for r in results if r['predicted_label'] is None]
    
    if not valid_results:
        print(" No valid results to evaluate!")
        return
    
    # Create summary DataFrame
    df = pd.DataFrame(valid_results)
    
    # Calculate comprehensive metrics
    total_docs = len(valid_results)
    correct_predictions = df['correct_prediction'].sum()
    accuracy = correct_predictions / total_docs
    
    # Confusion Matrix
    true_positives = len(df[(df['expected_label'] == 1) & (df['predicted_label'] == 1)])
    true_negatives = len(df[(df['expected_label'] == 0) & (df['predicted_label'] == 0)])
    false_positives = len(df[(df['expected_label'] == 0) & (df['predicted_label'] == 1)])
    false_negatives = len(df[(df['expected_label'] == 1) & (df['predicted_label'] == 0)])
    
    # Calculate metrics
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / max(0.001, precision + recall)
    specificity = true_negatives / max(1, true_negatives + false_positives)
    
    # Efficiency metrics
    avg_efficiency = df['efficiency_ratio'].mean()
    early_stops = df['stopped_early'].sum()
    early_stop_rate = early_stops / total_docs
    
    # Efficiency by class
    reservations_df = df[df['expected_label'] == 1]
    no_reservations_df = df[df['expected_label'] == 0]
    
    avg_efficiency_reservations = reservations_df['efficiency_ratio'].mean() if len(reservations_df) > 0 else 0
    avg_efficiency_no_reservations = no_reservations_df['efficiency_ratio'].mean() if len(no_reservations_df) > 0 else 0
    
    early_stops_reservations = reservations_df['stopped_early'].sum() if len(reservations_df) > 0 else 0
    early_stops_no_reservations = no_reservations_df['stopped_early'].sum() if len(no_reservations_df) > 0 else 0
    
    # Generate detailed report
    report = f"""
COMPREHENSIVE MINERAL RIGHTS CLASSIFICATION EVALUATION
=====================================================

PIPELINE CONFIGURATION:
- Method: Chunk-by-chunk early stopping
- Prompt: Enhanced with false positive reduction
- Token limit: 8K per page
- Confidence threshold: 0.6
- Max samples per chunk: 5

DATASET SUMMARY:
- Total Documents Processed: {len(results)}
- Valid Results: {total_docs}
- Processing Errors: {len(error_results)}
- Expected Reservations (Class 1): {len(reservations_df)}
- Expected No Reservations (Class 0): {len(no_reservations_df)}
- Total Processing Time: {total_time:.1f} seconds
- Average Time per Document: {total_time/len(results):.1f} seconds

ACCURACY METRICS:
- Overall Accuracy: {accuracy:.3f} ({correct_predictions}/{total_docs})
- Precision: {precision:.3f} (of predicted reservations, how many were correct)
- Recall: {recall:.3f} (of actual reservations, how many were found)
- Specificity: {specificity:.3f} (of actual no-reservations, how many were correctly identified)
- F1 Score: {f1_score:.3f}

CONFUSION MATRIX:
                    Predicted
                 No Res  Has Res
Actual No Res      {true_negatives:3d}     {false_positives:3d}
Actual Has Res     {false_negatives:3d}     {true_positives:3d}

EFFICIENCY METRICS:
- Average Efficiency: {avg_efficiency:.3f} (pages processed / total pages)
- Early Stops: {early_stops}/{total_docs} ({early_stop_rate:.3f})
- Average Pages Processed: {df['pages_processed'].mean():.1f}
- Average Total Pages: {df['total_pages_in_document'].mean():.1f}

EFFICIENCY BY CLASS:
- Documents WITH reservations:
  * Average efficiency: {avg_efficiency_reservations:.3f}
  * Early stops: {early_stops_reservations}/{len(reservations_df)} ({early_stops_reservations/max(1,len(reservations_df)):.3f})
  * (Should stop early when reservations found)

- Documents WITHOUT reservations:
  * Average efficiency: {avg_efficiency_no_reservations:.3f}
  * Early stops: {early_stops_no_reservations}/{len(no_reservations_df)} ({early_stops_no_reservations/max(1,len(no_reservations_df)):.3f})
  * (Should process all pages to confirm no reservations)

CONFIDENCE STATISTICS:
- Mean Confidence: {df['confidence'].mean():.3f}
- Median Confidence: {df['confidence'].median():.3f}
- Min Confidence: {df['confidence'].min():.3f}
- Max Confidence: {df['confidence'].max():.3f}

SAMPLING STATISTICS:
- Mean Samples Used: {df['samples_used'].mean():.1f}
- Total Samples Generated: {df['samples_used'].sum()}

OCR STATISTICS:
- Average OCR Text Length: {df['ocr_text_length'].mean():.0f} characters
- Min OCR Text Length: {df['ocr_text_length'].min()} characters
- Max OCR Text Length: {df['ocr_text_length'].max()} characters
"""
    
    # Add class-specific performance
    for class_label in [0, 1]:
        class_name = "No Reservations" if class_label == 0 else "Has Reservations"
        class_df = df[df['expected_label'] == class_label]
        if len(class_df) > 0:
            class_accuracy = class_df['correct_prediction'].mean()
            class_confidence = class_df['confidence'].mean()
            
            report += f"\n{class_name} (Class {class_label}) PERFORMANCE:\n"
            report += f"  - Count: {len(class_df)}\n"
            report += f"  - Accuracy: {class_accuracy:.3f}\n"
            report += f"  - Avg Confidence: {class_confidence:.3f}\n"
            report += f"  - Avg Pages Processed: {class_df['pages_processed'].mean():.1f}\n"
            report += f"  - Avg Efficiency: {class_df['efficiency_ratio'].mean():.3f}\n"
    
    # Add misclassification analysis
    report += f"\nMISCLASSIFICATION ANALYSIS ({total_docs - correct_predictions} total):\n"
    
    # False Positives
    false_pos_df = df[(df['expected_label'] == 0) & (df['predicted_label'] == 1)]
    if len(false_pos_df) > 0:
        report += f"\nFalse Positives ({len(false_pos_df)}) - Predicted reservations but actually none:\n"
        for _, row in false_pos_df.iterrows():
            report += f"  - {row['document_name']} (conf: {row['confidence']:.3f}, pages: {row['pages_processed']}/{row['total_pages_in_document']})\n"
    
    # False Negatives
    false_neg_df = df[(df['expected_label'] == 1) & (df['predicted_label'] == 0)]
    if len(false_neg_df) > 0:
        report += f"\nFalse Negatives ({len(false_neg_df)}) - Missed reservations:\n"
        for _, row in false_neg_df.iterrows():
            report += f"  - {row['document_name']} (conf: {row['confidence']:.3f}, pages: {row['pages_processed']}/{row['total_pages_in_document']})\n"
    
    # Processing errors
    if error_results:
        report += f"\nPROCESSING ERRORS ({len(error_results)}):\n"
        for error_result in error_results:
            report += f"  - {error_result['document_name']}: {error_result.get('error', 'Unknown error')}\n"
    
    # Confidence analysis
    correct_df = df[df['correct_prediction'] == True]
    incorrect_df = df[df['correct_prediction'] == False]
    
    if len(correct_df) > 0 and len(incorrect_df) > 0:
        report += f"\nCONFIDENCE ANALYSIS:\n"
        report += f"- Correct predictions avg confidence: {correct_df['confidence'].mean():.3f}\n"
        report += f"- Incorrect predictions avg confidence: {incorrect_df['confidence'].mean():.3f}\n"
        report += f"- Confidence difference: {correct_df['confidence'].mean() - incorrect_df['confidence'].mean():.3f}\n"
    
    # Save comprehensive report
    with open(output_dir / "comprehensive_evaluation_report.txt", "w") as f:
        f.write(report)
    
    # Save detailed CSV
    df.to_csv(output_dir / "detailed_results.csv", index=False)
    
    # Save summary statistics as JSON
    summary_stats = {
        'dataset_info': {
            'total_documents': len(results),
            'valid_results': total_docs,
            'processing_errors': len(error_results),
            'expected_reservations': len(reservations_df),
            'expected_no_reservations': len(no_reservations_df),
            'total_processing_time': total_time,
            'avg_time_per_document': total_time / len(results)
        },
        'accuracy_metrics': {
            'overall_accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'specificity': float(specificity)
        },
        'confusion_matrix': {
            'true_positives': int(true_positives),
            'true_negatives': int(true_negatives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives)
        },
        'efficiency_metrics': {
            'average_efficiency': float(avg_efficiency),
            'early_stops': int(early_stops),
            'early_stop_rate': float(early_stop_rate),
            'avg_efficiency_reservations': float(avg_efficiency_reservations),
            'avg_efficiency_no_reservations': float(avg_efficiency_no_reservations)
        },
        'confidence_stats': {
            'mean': float(df['confidence'].mean()),
            'median': float(df['confidence'].median()),
            'min': float(df['confidence'].min()),
            'max': float(df['confidence'].max())
        }
    }
    
    with open(output_dir / "summary_statistics.json", "w") as f:
        json.dump(summary_stats, f, indent=2)
    
    print(report)
    
    print(f"\n COMPREHENSIVE RESULTS SAVED:")
    print(f"  - {output_dir}/comprehensive_evaluation_report.txt")
    print(f"  - {output_dir}/detailed_results.csv")
    print(f"  - {output_dir}/summary_statistics.json")
    print(f"  - Individual results: {output_dir}/*_result.json")

if __name__ == "__main__":
    # Evaluate on the complete dataset
    results = evaluate_full_dataset([
        "data/reservs",
        "data/no-reservs"
    ])
    
    print(f"\n EVALUATION COMPLETE!")
    print(f"Processed {len(results)} documents with the pipeline.")
    print(f"Check the 'full_evaluation_results' directory for detailed analysis.") 