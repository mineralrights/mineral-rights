#!/usr/bin/env python3
"""
Oil and Gas Classification Evaluation Script
===========================================

Evaluates the updated oil and gas classifier on the entire dataset
and provides detailed performance metrics.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import pandas as pd
from document_classifier import DocumentProcessor

class OilGasEvaluator:
    """Evaluates oil and gas classification performance"""
    
    def __init__(self, api_key: str):
        """Initialize evaluator with API key"""
        self.processor = DocumentProcessor(api_key)
        self.results = []
        
    def evaluate_dataset(self, 
                        data_dir: str = "data",
                        max_samples: int = 5,  # Reduced for faster evaluation
                        confidence_threshold: float = 0.7,
                        save_detailed: bool = True) -> Dict:
        """
        Evaluate classifier on entire dataset
        
        Args:
            data_dir: Base data directory
            max_samples: Max samples per document for classification
            confidence_threshold: Confidence threshold for early stopping
            save_detailed: Whether to save detailed results for each document
        """
        
        print("🔍 EVALUATING OIL AND GAS CLASSIFIER")
        print("=" * 60)
        
        # Get file lists
        reservs_dir = Path(data_dir) / "reservs"
        no_reservs_dir = Path(data_dir) / "no-reservs"
        
        reservs_files = list(reservs_dir.glob("*.pdf"))
        no_reservs_files = list(no_reservs_dir.glob("*.pdf"))
        
        print(f"📊 Dataset Overview:")
        print(f"  - Documents WITH reservations: {len(reservs_files)}")
        print(f"  - Documents WITHOUT reservations: {len(no_reservs_files)}")
        print(f"  - Total documents: {len(reservs_files) + len(no_reservs_files)}")
        print(f"  - Max samples per document: {max_samples}")
        print(f"  - Confidence threshold: {confidence_threshold}")
        print()
        
        # Process all documents
        all_results = []
        start_time = time.time()
        
        # Process documents WITH reservations (expected label = 1, but now only if oil/gas)
        print("🔥 Processing documents WITH reservations...")
        for i, pdf_path in enumerate(reservs_files, 1):
            print(f"\n[{i}/{len(reservs_files)}] Processing: {pdf_path.name}")
            
            try:
                result = self.processor.process_document(
                    str(pdf_path),
                    max_samples=max_samples,
                    confidence_threshold=confidence_threshold
                )
                
                result_data = {
                    'file_path': str(pdf_path),
                    'file_name': pdf_path.name,
                    'true_label': 1,  # Originally had reservations
                    'predicted_label': result['classification'],
                    'confidence': result['confidence'],
                    'samples_used': result['samples_used'],
                    'early_stopped': result['early_stopped'],
                    'pages_processed': result['pages_processed'],
                    'ocr_text_length': result['ocr_text_length'],
                    'processing_time': time.time() - start_time,
                    'category': 'reservs',
                    'detailed_result': result if save_detailed else None
                }
                
                all_results.append(result_data)
                
                # Print quick result
                status = "✅ CORRECT" if result['classification'] == 1 else "❌ RECLASSIFIED"
                print(f"  Result: {result['classification']} ({status})")
                print(f"  Confidence: {result['confidence']:.3f}")
                
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                all_results.append({
                    'file_path': str(pdf_path),
                    'file_name': pdf_path.name,
                    'true_label': 1,
                    'predicted_label': -1,  # Error indicator
                    'confidence': 0.0,
                    'error': str(e),
                    'category': 'reservs'
                })
        
        # Process documents WITHOUT reservations (expected label = 0)
        print(f"\n🚫 Processing documents WITHOUT reservations...")
        for i, pdf_path in enumerate(no_reservs_files, 1):
            print(f"\n[{i}/{len(no_reservs_files)}] Processing: {pdf_path.name}")
            
            try:
                result = self.processor.process_document(
                    str(pdf_path),
                    max_samples=max_samples,
                    confidence_threshold=confidence_threshold
                )
                
                result_data = {
                    'file_path': str(pdf_path),
                    'file_name': pdf_path.name,
                    'true_label': 0,  # No reservations
                    'predicted_label': result['classification'],
                    'confidence': result['confidence'],
                    'samples_used': result['samples_used'],
                    'early_stopped': result['early_stopped'],
                    'pages_processed': result['pages_processed'],
                    'ocr_text_length': result['ocr_text_length'],
                    'processing_time': time.time() - start_time,
                    'category': 'no-reservs',
                    'detailed_result': result if save_detailed else None
                }
                
                all_results.append(result_data)
                
                # Print quick result
                status = "✅ CORRECT" if result['classification'] == 0 else "❌ FALSE POSITIVE"
                print(f"  Result: {result['classification']} ({status})")
                print(f"  Confidence: {result['confidence']:.3f}")
                
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                all_results.append({
                    'file_path': str(pdf_path),
                    'file_name': pdf_path.name,
                    'true_label': 0,
                    'predicted_label': -1,  # Error indicator
                    'confidence': 0.0,
                    'error': str(e),
                    'category': 'no-reservs'
                })
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_results)
        
        # Create evaluation summary
        evaluation_summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_documents': len(all_results),
                'reservs_documents': len(reservs_files),
                'no_reservs_documents': len(no_reservs_files),
                'data_directory': data_dir
            },
            'processing_info': {
                'max_samples_per_doc': max_samples,
                'confidence_threshold': confidence_threshold,
                'total_processing_time': total_time,
                'avg_time_per_doc': total_time / len(all_results) if all_results else 0
            },
            'performance_metrics': metrics,
            'detailed_results': all_results
        }
        
        # Save results
        self.save_evaluation_results(evaluation_summary)
        
        # Print summary
        self.print_evaluation_summary(evaluation_summary)
        
        return evaluation_summary
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate classification performance metrics"""
        
        # Filter out error cases
        valid_results = [r for r in results if r['predicted_label'] != -1]
        
        if not valid_results:
            return {'error': 'No valid results to calculate metrics'}
        
        # True positives, false positives, true negatives, false negatives
        tp = sum(1 for r in valid_results if r['true_label'] == 1 and r['predicted_label'] == 1)
        fp = sum(1 for r in valid_results if r['true_label'] == 0 and r['predicted_label'] == 1)
        tn = sum(1 for r in valid_results if r['true_label'] == 0 and r['predicted_label'] == 0)
        fn = sum(1 for r in valid_results if r['true_label'] == 1 and r['predicted_label'] == 0)
        
        # Calculate metrics
        total = len(valid_results)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Specificity (true negative rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate reclassification stats (important for oil/gas focus)
        originally_reservs = [r for r in valid_results if r['category'] == 'reservs']
        reclassified_to_no_reservs = sum(1 for r in originally_reservs if r['predicted_label'] == 0)
        reclassification_rate = reclassified_to_no_reservs / len(originally_reservs) if originally_reservs else 0
        
        return {
            'confusion_matrix': {
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            },
            'performance_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'specificity': specificity
            },
            'oil_gas_specific_metrics': {
                'reclassification_rate': reclassification_rate,
                'documents_reclassified': reclassified_to_no_reservs,
                'originally_reservs_count': len(originally_reservs)
            },
            'processing_stats': {
                'total_documents': total,
                'successful_classifications': len(valid_results),
                'failed_classifications': len([r for r in results if r['predicted_label'] == -1]),
                'avg_confidence': sum(r['confidence'] for r in valid_results) / len(valid_results) if valid_results else 0,
                'avg_samples_used': sum(r['samples_used'] for r in valid_results) / len(valid_results) if valid_results else 0
            }
        }
    
    def save_evaluation_results(self, evaluation_summary: Dict):
        """Save evaluation results to files"""
        
        # Create results directory
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete evaluation
        complete_file = results_dir / f"oil_gas_evaluation_{timestamp}.json"
        with open(complete_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2, default=str)
        
        # Save summary CSV
        results_data = []
        for result in evaluation_summary['detailed_results']:
            if result['predicted_label'] != -1:  # Skip errors
                results_data.append({
                    'file_name': result['file_name'],
                    'category': result['category'],
                    'true_label': result['true_label'],
                    'predicted_label': result['predicted_label'],
                    'confidence': result['confidence'],
                    'correct': result['true_label'] == result['predicted_label'],
                    'samples_used': result['samples_used'],
                    'pages_processed': result.get('pages_processed', 0)
                })
        
        if results_data:
            df = pd.DataFrame(results_data)
            csv_file = results_dir / f"oil_gas_results_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"\n📊 Results saved:")
            print(f"  - Complete evaluation: {complete_file}")
            print(f"  - Summary CSV: {csv_file}")
    
    def print_evaluation_summary(self, evaluation_summary: Dict):
        """Print detailed evaluation summary"""
        
        metrics = evaluation_summary['performance_metrics']
        
        print(f"\n{'='*60}")
        print(f"🎯 OIL AND GAS CLASSIFICATION EVALUATION RESULTS")
        print(f"{'='*60}")
        
        # Dataset info
        dataset = evaluation_summary['dataset_info']
        print(f"📊 Dataset: {dataset['total_documents']} documents")
        print(f"   - Originally WITH reservations: {dataset['reservs_documents']}")
        print(f"   - Originally WITHOUT reservations: {dataset['no_reservs_documents']}")
        
        # Processing info
        processing = evaluation_summary['processing_info']
        print(f"\n⏱️  Processing: {processing['total_processing_time']:.1f}s total")
        print(f"   - Average per document: {processing['avg_time_per_doc']:.1f}s")
        print(f"   - Max samples per doc: {processing['max_samples_per_doc']}")
        
        # Performance metrics
        perf = metrics['performance_metrics']
        conf_matrix = metrics['confusion_matrix']
        
        print(f"\n🎯 PERFORMANCE METRICS:")
        print(f"   - Accuracy:    {perf['accuracy']:.3f} ({perf['accuracy']*100:.1f}%)")
        print(f"   - Precision:   {perf['precision']:.3f} ({perf['precision']*100:.1f}%)")
        print(f"   - Recall:      {perf['recall']:.3f} ({perf['recall']*100:.1f}%)")
        print(f"   - F1-Score:    {perf['f1_score']:.3f}")
        print(f"   - Specificity: {perf['specificity']:.3f} ({perf['specificity']*100:.1f}%)")
        
        print(f"\n📋 CONFUSION MATRIX:")
        print(f"   - True Positives (Oil/Gas Found):     {conf_matrix['true_positives']}")
        print(f"   - False Positives (Incorrectly Found): {conf_matrix['false_positives']}")
        print(f"   - True Negatives (Correctly No O/G):   {conf_matrix['true_negatives']}")
        print(f"   - False Negatives (Missed Oil/Gas):    {conf_matrix['false_negatives']}")
        
        # Oil/Gas specific metrics
        oil_gas = metrics['oil_gas_specific_metrics']
        print(f"\n🛢️  OIL & GAS SPECIFIC ANALYSIS:")
        print(f"   - Documents reclassified from reservations to no-reservations: {oil_gas['documents_reclassified']}/{oil_gas['originally_reservs_count']}")
        print(f"   - Reclassification rate: {oil_gas['reclassification_rate']:.3f} ({oil_gas['reclassification_rate']*100:.1f}%)")
        print(f"   - This indicates documents that had non-oil/gas reservations (e.g., coal only)")
        
        # Processing stats
        proc_stats = metrics['processing_stats']
        print(f"\n📈 PROCESSING STATISTICS:")
        print(f"   - Successful classifications: {proc_stats['successful_classifications']}/{proc_stats['total_documents']}")
        print(f"   - Average confidence: {proc_stats['avg_confidence']:.3f}")
        print(f"   - Average samples used: {proc_stats['avg_samples_used']:.1f}")
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS:")
        if oil_gas['reclassification_rate'] > 0.3:
            print(f"   ✅ High reclassification rate suggests the classifier is successfully")
            print(f"      distinguishing oil/gas from other mineral reservations")
        
        if perf['specificity'] > 0.8:
            print(f"   ✅ High specificity indicates good performance at avoiding false positives")
        
        if perf['precision'] > 0.8:
            print(f"   ✅ High precision means when it finds oil/gas reservations, it's usually correct")
        
        print(f"\n{'='*60}")

def main():
    """Run evaluation on the entire dataset"""
    
    # Get API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        print("   export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    # Initialize evaluator
    evaluator = OilGasEvaluator(api_key)
    
    # Run evaluation
    print("🚀 Starting oil and gas classification evaluation...")
    print("⚠️  This will process all 53 documents and may take 20-30 minutes")
    
    # Ask for confirmation
    response = input("\nProceed with full evaluation? (y/N): ").strip().lower()
    if response != 'y':
        print("Evaluation cancelled.")
        return
    
    try:
        # Run evaluation with conservative settings for speed
        evaluation_summary = evaluator.evaluate_dataset(
            data_dir="data",
            max_samples=3,  # Reduced for faster evaluation
            confidence_threshold=0.7,
            save_detailed=True
        )
        
        print("\n🎉 Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main() 