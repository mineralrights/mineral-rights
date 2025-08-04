#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def diagnose_classifier():
    """Diagnose what's working and what's broken"""
    
    print("🔍 DIAGNOSTIC: Testing OilGasRightsClassifier...")
    
    try:
        from mineral_rights.document_classifier import OilGasRightsClassifier
        print("✅ Successfully imported OilGasRightsClassifier")
    except Exception as e:
        print(f"❌ Failed to import OilGasRightsClassifier: {e}")
        return False
    
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not set")
            return False
        
        classifier = OilGasRightsClassifier(api_key)
        print("✅ Successfully initialized classifier")
        
        # Test if methods exist
        methods_to_check = ['create_classification_prompt', 'extract_classification', 'generate_sample', 'classify_document']
        for method in methods_to_check:
            if hasattr(classifier, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} MISSING")
        
        # Test sample generation
        print("\n🧪 Testing sample generation...")
        test_text = "EXCEPTING AND RESERVING all oil and gas rights"
        
        try:
            sample = classifier.generate_sample(test_text, high_recall_mode=True)
            print(f"✅ generate_sample worked")
            print(f"   Predicted class: {sample.predicted_class}")
            print(f"   Reasoning length: {len(sample.reasoning)}")
            print(f"   Raw response length: {len(sample.raw_response)}")
            
            if sample.predicted_class == 0 and not sample.reasoning:
                print("❌ generate_sample returns dummy values (BROKEN)")
                return False
            else:
                print("✅ generate_sample returns real values")
                
        except Exception as e:
            print(f"❌ generate_sample failed: {e}")
            return False
        
        # Test full classification
        print("\n🧪 Testing full classification...")
        try:
            result = classifier.classify_document(test_text, max_samples=1)
            print(f"✅ classify_document worked")
            print(f"   Final classification: {result.predicted_class}")
            print(f"   Confidence: {result.confidence}")
            
            if result.predicted_class == 1:
                print("✅ CORRECTLY detected reservations")
                return True
            else:
                print("❌ FAILED to detect reservations")
                return False
                
        except Exception as e:
            print(f"❌ classify_document failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ General error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = diagnose_classifier()
    if success:
        print("\n🎉 DIAGNOSIS: Classifier is working!")
    else:
        print("\n💥 DIAGNOSIS: Classifier is broken - need to fix methods") 