#!/usr/bin/env python3
"""
Comprehensive Oil and Gas Classification Test Suite
=================================================

This script thoroughly tests the core classification logic on the entire dataset
with focus on:
1. High specificity for no-reservations (detecting true negatives)
2. Overall performance metrics
3. Error analysis and diagnostic reporting
4. Comparison with previous baselines

Usage:
    python test_core_classification.py
"""

import os
import sys
import json
import time
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import traceback

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the classifier
try:
    # First try the direct import
    from src.mineral_rights.document_classifier import OilGasRightsClassifier
    print("✅ Successfully imported OilGasRightsClassifier")
except ImportError as e1:
    print(f"❌ Direct import failed: {e1}")
    try:
        # Try adding src to path
        sys.path.insert(0, str(project_root / "src"))
        from mineral_rights.document_classifier import OilGasRightsClassifier
        print("✅ Successfully imported OilGasRightsClassifier (method 2)")
    except ImportError as e2:
        print(f"❌ Module import failed: {e2}")
        try:
            # Try importing the file directly
            import importlib.util
            classifier_path = project_root / "src" / "mineral_rights" / "document_classifier.py"
            spec = importlib.util.spec_from_file_location("document_classifier", classifier_path)
            document_classifier = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(document_classifier)
            OilGasRightsClassifier = document_classifier.OilGasRightsClassifier
            print("✅ Successfully imported OilGasRightsClassifier (direct file import)")
        except Exception as e3:
            print(f"❌ All import methods failed:")
            print(f"   Method 1: {e1}")
            print(f"   Method 2: {e2}")
            print(f"   Method 3: {e3}")
            print(f"\nDebugging info:")
            print(f"   Project root: {project_root}")
            print(f"   Classifier file exists: {(project_root / 'src' / 'mineral_rights' / 'document_classifier.py').exists()}")
            print(f"   Current working directory: {os.getcwd()}")
            print(f"   Python path: {sys.path[:3]}...")
            sys.exit(1)


@dataclass
class TestResult:
    """Individual document test result"""
    filename: str
    true_label: int  # 0 = no-reservs, 1 = reservs
    predicted_label: int
    confidence: float
    processing_time: float
    samples_used: int
    early_stopped: bool
    stopped_at_page: Optional[int]
    pages_processed: int
    error: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for the classifier"""
    # Basic metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float  # True Negative Rate (critical for no-reservations)
    
    # Confusion matrix
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # Processing stats
    avg_confidence: float
    avg_processing_time: float
    avg_samples_used: float
    total_processing_time: float
    
    # Error analysis
    false_positive_rate: float
    false_negative_rate: float
    misclassification_rate: float


class CoreClassificationTester:
    """Comprehensive testing framework for oil and gas classification"""
    
    def __init__(self, api_key: str):
        self.classifier = OilGasRightsClassifier(api_key)
        self.results: List[TestResult] = []
        self.start_time = datetime.now()
        
    def get_dataset_files(self) -> Tuple[List[Path], List[Path]]:
        """Get all PDF files from reservs and no-reservs directories"""
        # Look for data directory relative to project root
        data_dir = project_root / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found at {data_dir}")
        
        reservs_files = list((data_dir / "reservs").glob("*.pdf"))
        no_reservs_files = list((data_dir / "no-reservs").glob("*.pdf"))
        
        print(f"📁 Dataset Overview:")
        print(f"   • Reservations (positive): {len(reservs_files)} files")
        print(f"   • No-reservations (negative): {len(no_reservs_files)} files")
        print(f"   • Total: {len(reservs_files) + len(no_reservs_files)} files")
        
        return reservs_files, no_reservs_files
    
    def test_single_document(
        self, 
        pdf_path: Path, 
        true_label: int,
        max_samples: int = 8,
        confidence_threshold: float = 0.80
    ) -> TestResult:
        """Test classification on a single document"""
        start_time = time.time()
        
        try:
            result = self.classifier.process_document(
                str(pdf_path),
                max_samples=max_samples,
                confidence_threshold=confidence_threshold
            )
            
            processing_time = time.time() - start_time
            
            return TestResult(
                filename=pdf_path.name,
                true_label=true_label,
                predicted_label=result['classification'],
                confidence=result['confidence'],
                processing_time=processing_time,
                samples_used=len(result.get('chunk_results', [])),
                early_stopped=result.get('early_stopped', False),
                stopped_at_page=result.get('stopped_at_page'),
                pages_processed=result.get('pages_processed', 0)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            return TestResult(
                filename=pdf_path.name,
                true_label=true_label,
                predicted_label=-1,  # Error indicator
                confidence=0.0,
                processing_time=processing_time,
                samples_used=0,
                early_stopped=False,
                stopped_at_page=None,
                pages_processed=0,
                error=error_msg
            )
    
    def run_full_evaluation(
        self, 
        max_samples: int = 8,
        confidence_threshold: float = 0.80,
        verbose: bool = True
    ) -> PerformanceMetrics:
        """Run evaluation on entire dataset"""
        
        print(f"🚀 Starting Comprehensive Classification Test")
        print(f"{'='*60}")
        print(f"Parameters:")
        print(f"   • Max samples per document: {max_samples}")
        print(f"   • Confidence threshold: {confidence_threshold}")
        print(f"   • Focus: HIGH SPECIFICITY (detecting no-reservations)")
        print()
        
        reservs_files, no_reservs_files = self.get_dataset_files()
        
        # Test no-reservations first (priority)
        print("🔍 Testing NO-RESERVATIONS documents (priority)...")
        for i, pdf_path in enumerate(no_reservs_files, 1):
            if verbose:
                print(f"   [{i:2d}/{len(no_reservs_files)}] {pdf_path.name}", end=" ... ")
            
            result = self.test_single_document(pdf_path, true_label=0, 
                                             max_samples=max_samples, 
                                             confidence_threshold=confidence_threshold)
            self.results.append(result)
            
            if verbose:
                if result.error:
                    print(f"❌ ERROR: {result.error}")
                else:
                    status = "✅ CORRECT" if result.predicted_label == 0 else "❌ FALSE POSITIVE"
                    print(f"{status} (conf: {result.confidence:.3f}, time: {result.processing_time:.1f}s)")
        
        print(f"\n🔥 Testing RESERVATIONS documents...")
        for i, pdf_path in enumerate(reservs_files, 1):
            if verbose:
                print(f"   [{i:2d}/{len(reservs_files)}] {pdf_path.name}", end=" ... ")
            
            result = self.test_single_document(pdf_path, true_label=1, 
                                             max_samples=max_samples, 
                                             confidence_threshold=confidence_threshold)
            self.results.append(result)
            
            if verbose:
                if result.error:
                    print(f"❌ ERROR: {result.error}")
                else:
                    status = "✅ CORRECT" if result.predicted_label == 1 else "❌ FALSE NEGATIVE"
                    print(f"{status} (conf: {result.confidence:.3f}, time: {result.processing_time:.1f}s)")
        
        # Calculate performance metrics
        metrics = self.calculate_metrics()
        
        print(f"\n{'='*60}")
        print(f"🎯 EVALUATION COMPLETE")
        print(f"   Total processing time: {metrics.total_processing_time:.1f} seconds")
        print(f"   Average time per document: {metrics.avg_processing_time:.1f} seconds")
        
        return metrics
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        # Filter out error cases
        valid_results = [r for r in self.results if r.error is None]
        
        if not valid_results:
            raise ValueError("No valid results to calculate metrics")
        
        # Confusion matrix
        tp = sum(1 for r in valid_results if r.true_label == 1 and r.predicted_label == 1)
        fp = sum(1 for r in valid_results if r.true_label == 0 and r.predicted_label == 1)
        tn = sum(1 for r in valid_results if r.true_label == 0 and r.predicted_label == 0)
        fn = sum(1 for r in valid_results if r.true_label == 1 and r.predicted_label == 0)
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Critical metric
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Error rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Processing stats
        confidences = [r.confidence for r in valid_results]
        times = [r.processing_time for r in valid_results]
        samples = [r.samples_used for r in valid_results]
        
        return PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            avg_confidence=statistics.mean(confidences) if confidences else 0,
            avg_processing_time=statistics.mean(times) if times else 0,
            avg_samples_used=statistics.mean(samples) if samples else 0,
            total_processing_time=sum(times),
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            misclassification_rate=1 - accuracy
        )
    
    def generate_detailed_report(self, metrics: PerformanceMetrics) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("COMPREHENSIVE OIL & GAS CLASSIFICATION TEST REPORT")
        report.append("=" * 60)
        report.append(f"Test Date: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Documents Tested: {len(self.results)}")
        report.append("")
        
        # Performance Summary
        report.append("🎯 PERFORMANCE SUMMARY")
        report.append("-" * 30)
        report.append(f"Overall Accuracy:     {metrics.accuracy:.1%}")
        report.append(f"Specificity (TNR):    {metrics.specificity:.1%} ⭐ KEY METRIC")
        report.append(f"Sensitivity (Recall): {metrics.recall:.1%}")
        report.append(f"Precision:            {metrics.precision:.1%}")
        report.append(f"F1-Score:            {metrics.f1_score:.1%}")
        report.append("")
        
        # Error Analysis (Critical for your use case)
        report.append("🚨 ERROR ANALYSIS (Focus: No-Reservations)")
        report.append("-" * 40)
        report.append(f"False Positive Rate:  {metrics.false_positive_rate:.1%} (wrongly flagged as reservations)")
        report.append(f"False Negative Rate:  {metrics.false_negative_rate:.1%} (missed actual reservations)")
        report.append(f"Misclassification:    {metrics.misclassification_rate:.1%}")
        report.append("")
        
        # Confusion Matrix
        report.append("📊 CONFUSION MATRIX")
        report.append("-" * 20)
        report.append("                    Predicted")
        report.append("                No-Res  Reserv")
        report.append(f"Actual  No-Res    {metrics.true_negatives:3d}     {metrics.false_positives:3d}")
        report.append(f"        Reserv    {metrics.false_negatives:3d}     {metrics.true_positives:3d}")
        report.append("")
        
        # Processing Stats
        report.append("⚡ PROCESSING STATISTICS")
        report.append("-" * 25)
        report.append(f"Total Processing Time:    {metrics.total_processing_time:.1f} seconds")
        report.append(f"Average Time per Doc:     {metrics.avg_processing_time:.1f} seconds")
        report.append(f"Average Confidence:       {metrics.avg_confidence:.3f}")
        report.append(f"Average Samples Used:     {metrics.avg_samples_used:.1f}")
        report.append("")
        
        # Error Details
        error_results = [r for r in self.results if r.error is not None]
        if error_results:
            report.append("❌ PROCESSING ERRORS")
            report.append("-" * 18)
            for err_result in error_results:
                report.append(f"• {err_result.filename}: {err_result.error}")
            report.append("")
        
        # Misclassification Details
        fp_results = [r for r in self.results if r.error is None and r.true_label == 0 and r.predicted_label == 1]
        fn_results = [r for r in self.results if r.error is None and r.true_label == 1 and r.predicted_label == 0]
        
        if fp_results:
            report.append("🔍 FALSE POSITIVES (Critical - No-reservs wrongly flagged)")
            report.append("-" * 55)
            for fp in fp_results:
                report.append(f"• {fp.filename} (confidence: {fp.confidence:.3f})")
            report.append("")
        
        if fn_results:
            report.append("🔍 FALSE NEGATIVES (Missed actual reservations)")
            report.append("-" * 45)
            for fn in fn_results:
                report.append(f"• {fn.filename} (confidence: {fn.confidence:.3f})")
            report.append("")
        
        # Benchmark Comparison
        report.append("📈 BENCHMARK COMPARISON")
        report.append("-" * 23)
        report.append("Target Metrics for Production:")
        report.append(f"  • Specificity (TNR):  ≥95% [Current: {metrics.specificity:.1%}]")
        report.append(f"  • Overall Accuracy:   ≥90% [Current: {metrics.accuracy:.1%}]")
        report.append(f"  • False Positive Rate: ≤5% [Current: {metrics.false_positive_rate:.1%}]")
        
        # Recommendations
        report.append("")
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 17)
        if metrics.specificity < 0.95:
            report.append("⚠️  Specificity below target - tune confidence threshold higher")
        if metrics.false_positive_rate > 0.05:
            report.append("⚠️  Too many false positives - review conservative prompt")
        if metrics.accuracy < 0.90:
            report.append("⚠️  Overall accuracy below target - investigate feature engineering")
        if metrics.specificity >= 0.95 and metrics.accuracy >= 0.90:
            report.append("✅ Performance meets production targets!")
        
        return "\n".join(report)
    
    def save_results(self, metrics: PerformanceMetrics, output_dir: str = "test_results"):
        """Save detailed results and report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        results_data = {
            "timestamp": self.start_time.isoformat(),
            "metrics": asdict(metrics),
            "detailed_results": [asdict(r) for r in self.results],
            "parameters": {
                "focus": "High specificity for no-reservations",
                "total_documents": len(self.results)
            }
        }
        
        json_path = output_path / f"core_classification_test_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save human-readable report
        report = self.generate_detailed_report(metrics)
        report_path = output_path / f"core_classification_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n💾 Results saved:")
        print(f"   • Detailed data: {json_path}")
        print(f"   • Human report:  {report_path}")


def main():
    """Main test execution"""
    # Get API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        return
    
    try:
        print("🧪 CORE CLASSIFICATION TESTING SUITE")
        print("=" * 50)
        print("🎯 PRIORITY: High specificity for no-reservations")
        print("📊 Testing entire dataset for comprehensive evaluation")
        print(f"📁 Project root: {project_root}")
        print()
        
        # Initialize tester
        tester = CoreClassificationTester(api_key)
        
        # Run full evaluation
        metrics = tester.run_full_evaluation(
            max_samples=8,           # Robust sampling
            confidence_threshold=0.80,  # Conservative threshold
            verbose=True
        )
        
        # Display results
        print(f"\n{tester.generate_detailed_report(metrics)}")
        
        # Save results
        tester.save_results(metrics)
        
        # Quick summary for immediate feedback
        print(f"\n🏆 QUICK SUMMARY:")
        print(f"   Accuracy: {metrics.accuracy:.1%}")
        print(f"   Specificity (key): {metrics.specificity:.1%}")
        print(f"   False Positives: {metrics.false_positives} documents")
        print(f"   Processing Time: {metrics.total_processing_time:.1f}s")
        
        if metrics.specificity >= 0.95 and metrics.accuracy >= 0.90:
            print(f"\n✅ CORE CLASSIFICATION ABILITY: EXCELLENT")
        elif metrics.specificity >= 0.90:
            print(f"\n⚠️  CORE CLASSIFICATION ABILITY: GOOD (room for improvement)")
        else:
            print(f"\n❌ CORE CLASSIFICATION ABILITY: NEEDS ATTENTION")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
