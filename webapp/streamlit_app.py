import streamlit as st
import os
import tempfile
import pandas as pd
import csv
from datetime import datetime
from document_classifier import DocumentProcessor

# Page config
st.set_page_config(
    page_title="Mineral Rights Analyzer",
    page_icon="⛏️",
    layout="wide"
)

# Custom CSS matching the Flask template design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@300;400;500;600&display=swap');
    
    /* Reset and base styles */
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 25%, #e0f2e0 50%, #f5fbf5 75%, #e8f5e8 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        color: #2c5530;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Hide Streamlit default elements */
    .stDeployButton {display: none;}
    .stDecoration {display: none;}
    header[data-testid="stHeader"] {display: none;}
    .stToolbar {display: none;}
    .stAppViewContainer > .main > div {padding-top: 0rem;}
    
    /* Main container */
    .main-container {
        max-width: 950px;
        margin: 20px auto;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        box-shadow: 0 30px 60px rgba(44, 85, 48, 0.12);
        overflow: hidden;
        border: 1px solid rgba(76, 175, 80, 0.1);
        transition: all 0.3s ease;
    }

    /* Header section */
    .header-section {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 25%, #81C784 50%, #66BB6A 75%, #4CAF50 100%);
        background-size: 300% 300%;
        animation: headerGradient 12s ease infinite;
        color: white;
        padding: 40px;
        text-align: center;
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    @keyframes headerGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .header-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="dots" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="10" cy="10" r="1.5" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23dots)"/></svg>');
        opacity: 0.4;
    }

    .header-icon {
        font-size: 4em;
        margin-bottom: 20px;
        display: inline-block;
        animation: bounce 2s ease-in-out infinite;
        position: relative;
        z-index: 2;
        text-align: center;
    }

    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }

    .header-title {
        font-size: 2.8em;
        margin-bottom: 15px;
        font-weight: 300;
        letter-spacing: -1px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: relative;
        z-index: 2;
        text-align: center;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
    }

    .header-subtitle {
        font-size: 1.2em;
        opacity: 0.95;
        line-height: 1.5;
        max-width: 650px;
        margin: 0 auto;
        font-weight: 300;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        position: relative;
        z-index: 2;
        text-align: center;
        width: 100%;
    }

    /* Content section */
    .content-section {
        padding: 40px;
    }

    /* Upload section */
    .upload-area {
        border: 3px dashed #81C784;
        border-radius: 16px;
        padding: 50px 40px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        background: linear-gradient(135deg, #f8fff8 0%, #ffffff 100%);
        position: relative;
        overflow: hidden;
        margin: 30px 0;
    }

    .upload-area::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(76, 175, 80, 0.08), transparent);
        transition: left 0.6s ease;
    }

    .upload-area:hover::before {
        left: 100%;
    }

    .upload-area:hover {
        border-color: #4CAF50;
        background: linear-gradient(135deg, #f0fff0 0%, #fafffe 100%);
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 15px 40px rgba(76, 175, 80, 0.15);
    }

    .upload-icon {
        font-size: 4em;
        color: #81C784;
        margin-bottom: 20px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        display: inline-block;
    }

    .upload-text {
        font-size: 1.5em;
        color: #2e7d32;
        margin-bottom: 12px;
        font-weight: 500;
    }

    .upload-subtext {
        color: #66BB6A;
        font-size: 1em;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        flex-wrap: wrap;
    }

    .feature-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(76, 175, 80, 0.1);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        color: #2e7d32;
        border: 1px solid rgba(76, 175, 80, 0.2);
    }

    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #f8fff8 0%, #ffffff 100%);
        border-radius: 16px;
        padding: 50px;
        margin: 40px 0;
        border: 1px solid #e8f5e8;
        box-shadow: 0 10px 30px rgba(76, 175, 80, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #4CAF50, #81C784, #4CAF50);
        background-size: 200% 100%;
        animation: shimmer 3s ease-in-out infinite;
    }

    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    .result-header {
        display: flex;
        align-items: center;
        margin-bottom: 35px;
        padding-bottom: 25px;
        border-bottom: 2px solid #e8f5e8;
    }

    .result-icon {
        font-size: 3.5em;
        margin-right: 25px;
        animation: resultPulse 2s ease-in-out infinite;
    }

    @keyframes resultPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    .result-title {
        font-size: 2.2em;
        font-weight: 600;
        color: #1b5e20;
        flex-grow: 1;
        line-height: 1.3;
    }

    .confidence-badge {
        display: inline-block;
        padding: 12px 24px;
        border-radius: 30px;
        font-size: 1em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        position: relative;
        overflow: hidden;
    }

    .confidence-high {
        background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%);
        color: #1b5e20;
        border: 2px solid #81c784;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
    }

    .confidence-medium {
        background: linear-gradient(135deg, #fff9c4 0%, #fff176 100%);
        color: #f57f17;
        border: 2px solid #ffeb3b;
        box-shadow: 0 4px 15px rgba(255, 235, 59, 0.2);
    }

    .confidence-low {
        background: linear-gradient(135deg, #ffcdd2 0%, #ef9a9a 100%);
        color: #c62828;
        border: 2px solid #f44336;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.2);
    }

    .explanation {
        line-height: 1.8;
        color: #2e7d32;
        margin: 30px 0;
        font-size: 1.3em;
        text-align: justify;
    }

    .recommendation {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0fff0 100%);
        border-left: 5px solid #4CAF50;
        padding: 30px 35px;
        border-radius: 0 12px 12px 0;
        margin: 30px 0;
        position: relative;
    }

    .recommendation::before {
        content: '💡';
        position: absolute;
        top: 15px;
        right: 20px;
        font-size: 1.5em;
        opacity: 0.7;
    }

    .recommendation strong {
        color: #1b5e20;
        font-size: 1.2em;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .recommendation-text {
        margin-top: 15px;
        font-size: 1.1em;
        line-height: 1.7;
        color: #2e7d32;
    }

    .details {
        background: #ffffff;
        border-radius: 12px;
        padding: 35px;
        margin: 35px 0;
        border: 2px solid #e8f5e8;
        box-shadow: inset 0 2px 4px rgba(76, 175, 80, 0.05);
    }

    .details h4 {
        color: #1b5e20;
        margin-bottom: 25px;
        font-size: 1.5em;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .detail-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 0;
        border-bottom: 1px solid #f0fff0;
        font-size: 1.1em;
        transition: all 0.2s ease;
    }

    .detail-item:last-child {
        border-bottom: none;
    }

    .detail-label {
        font-weight: 500;
        color: #2e7d32;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .detail-value {
        color: #1b5e20;
        font-weight: 600;
        background: rgba(76, 175, 80, 0.1);
        padding: 4px 12px;
        border-radius: 20px;
    }

    /* Streamlit button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
        color: white;
        border: none;
        padding: 16px 35px;
        border-radius: 12px;
        font-size: 1.1em;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
        position: relative;
        overflow: hidden;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(76, 175, 80, 0.4);
        background: linear-gradient(135deg, #66BB6A 0%, #4CAF50 100%);
    }

    /* File uploader styling */
    .stFileUploader > div > div {
        background: linear-gradient(135deg, #f8fff8 0%, #ffffff 100%);
        border: 3px dashed #81C784;
        border-radius: 16px;
        padding: 50px 40px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .stFileUploader > div > div::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(76, 175, 80, 0.08), transparent);
        transition: left 0.6s ease;
    }

    .stFileUploader > div > div:hover::before {
        left: 100%;
    }

    .stFileUploader > div > div:hover {
        border-color: #4CAF50;
        background: linear-gradient(135deg, #f0fff0 0%, #fafffe 100%);
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 15px 40px rgba(76, 175, 80, 0.15);
    }

    /* Style the file uploader label */
    .stFileUploader label {
        font-size: 1.5em !important;
        color: #2e7d32 !important;
        font-weight: 500 !important;
        margin-bottom: 20px !important;
    }

    /* Add custom content to file uploader */
    .stFileUploader > div > div::after {
        content: '🔒 Secure    📊 50MB Max    ⚡ Fast Processing';
        display: block;
        margin-top: 15px;
        font-size: 0.9em;
        color: #66BB6A;
        background: rgba(76, 175, 80, 0.1);
        padding: 8px 16px;
        border-radius: 20px;
        border: 1px solid rgba(76, 175, 80, 0.2);
        width: fit-content;
        margin-left: auto;
        margin-right: auto;
    }

    /* Style the browse files button */
    .stFileUploader button {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 30px !important;
        border-radius: 12px !important;
        font-size: 1em !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3) !important;
        margin-top: 20px !important;
    }

    .stFileUploader button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4) !important;
        background: linear-gradient(135deg, #66BB6A 0%, #4CAF50 100%) !important;
    }

    /* Processing spinner */
    .stSpinner > div {
        border-color: #4CAF50 !important;
    }

    /* Error styling */
    .stAlert > div {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border: 2px solid #ef9a9a;
        border-radius: 12px;
    }

    /* Success styling */
    .stSuccess > div {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border: 2px solid #81c784;
        border-radius: 12px;
        color: #1b5e20;
    }

    /* Info styling */
    .stInfo > div {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: 2px solid #64b5f6;
        border-radius: 12px;
    }

    /* Section titles */
    .section-title {
        font-size: 2em;
        color: #2e7d32;
        margin: 30px 0;
        font-weight: 600;
        text-align: center;
        position: relative;
    }

    .section-title::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #4CAF50, #81C784);
        border-radius: 2px;
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-container {
            margin: 15px;
            border-radius: 16px;
        }
        
        .header-section {
            padding: 30px 20px;
        }
        
        .header-title {
            font-size: 2.2em;
        }
        
        .header-subtitle {
            font-size: 1.1em;
        }
        
        .content-section {
            padding: 30px 20px;
        }
        
        .upload-area {
            padding: 40px 20px;
        }

        .result-header {
            flex-direction: column;
            text-align: center;
        }

        .result-icon {
            margin-right: 0;
            margin-bottom: 20px;
        }

        .detail-item {
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
    st.session_state.processing_error = None
    st.session_state.results_df = None

def initialize_processor():
    """Initialize the document processor"""
    try:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            st.error("❌ ANTHROPIC_API_KEY environment variable not set")
            st.info("Please set your Anthropic API key in the Streamlit Cloud settings.")
            return False
        
        st.session_state.processor = DocumentProcessor(api_key=api_key)
        return True
    except Exception as e:
        st.session_state.processing_error = str(e)
        st.error(f"❌ Failed to initialize: {e}")
        return False

def process_document(uploaded_file):
    """Process the uploaded document"""
    tmp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process document
        with st.spinner('🔍 Analyzing document...'):
            result = st.session_state.processor.process_document(
                tmp_path,
                max_samples=5,
                confidence_threshold=0.7
            )
        
        return result
        
    except Exception as e:
        raise e
    finally:
        # Always clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass  # Ignore cleanup errors

def get_confidence_level(confidence):
    """Get confidence level classification"""
    if confidence >= 0.8:
        return "HIGH", "confidence-high"
    elif confidence >= 0.6:
        return "MEDIUM", "confidence-medium"
    else:
        return "LOW", "confidence-low"

def get_recommendation(classification, confidence):
    """Get professional recommendation based on results"""
    if classification == 0:
        if confidence >= 0.8:
            return "This document appears to be a clean transfer without mineral rights reservations. You can proceed with confidence, but always consult with a qualified attorney for final verification."
        else:
            return "While our analysis suggests no mineral rights reservations, the confidence level warrants additional review. Consider having a legal professional examine the document for complete certainty."
    else:
        if confidence >= 0.8:
            return "Strong evidence of mineral rights reservations detected. This document likely contains clauses that reserve mineral rights to the grantor or previous parties. Legal review is strongly recommended before proceeding."
        else:
            return "Potential mineral rights reservations detected, but with moderate confidence. Professional legal review is essential to determine the exact nature and scope of any reservations."

def process_multiple_documents(uploaded_files):
    """Process multiple uploaded documents and return results as DataFrame"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        tmp_path = None
        try:
            status_text.text(f'Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...')
            
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Process document
            result = st.session_state.processor.process_document(
                tmp_path,
                max_samples=5,
                confidence_threshold=0.7
            )
            
            # Extract detailed information for CSV
            classification = result['classification']
            confidence = result['confidence']
            confidence_level, _ = get_confidence_level(confidence)
            
            # Get the best reasoning from detailed samples
            best_reasoning = ""
            if 'detailed_samples' in result and result['detailed_samples']:
                # Get the sample with highest confidence
                best_sample = max(result['detailed_samples'], key=lambda x: x.get('confidence_score', 0))
                best_reasoning = best_sample.get('reasoning', '')
            
            # Calculate additional metrics
            votes = result['votes']
            total_votes = sum(votes.values())
            no_reservation_votes = votes.get(0, 0)
            has_reservation_votes = votes.get(1, 0)
            vote_ratio = has_reservation_votes / total_votes if total_votes > 0 else 0
            
            # Prepare result row
            result_row = {
                'filename': uploaded_file.name,
                'file_size_bytes': uploaded_file.size,
                'processing_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'classification': 'Has Mineral Rights Reservations' if classification == 1 else 'No Mineral Rights Reservations',
                'classification_numeric': classification,
                'confidence_score': round(confidence, 4),
                'confidence_level': confidence_level,
                'recommendation': get_recommendation(classification, confidence),
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
            
        except Exception as e:
            # Add error row
            error_row = {
                'filename': uploaded_file.name,
                'file_size_bytes': uploaded_file.size,
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
            
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text('Processing complete!')
    return pd.DataFrame(results)

def create_csv_download(df):
    """Create CSV download with proper formatting"""
    # Create CSV string
    csv_buffer = df.to_csv(index=False, quoting=csv.QUOTE_ALL)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'mineral_rights_analysis_{timestamp}.csv'
    
    return csv_buffer, filename

# Main app
def main():
    # Create main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header section
    st.markdown('''
    <div class="header-section">
        <div class="header-icon">🏛️</div>
        <h1 class="header-title">Mineral Rights Document Analyzer</h1>
        <p class="header-subtitle">AI-powered batch analysis of deed documents to identify mineral rights reservations with detailed confidence scoring and expert recommendations.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Content section
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    
    # Initialize processor if not already done
    if st.session_state.processor is None and st.session_state.processing_error is None:
        with st.spinner('🔧 Initializing document processor...'):
            initialize_processor()
    
    # Show initialization error if any
    if st.session_state.processing_error:
        st.error(f"❌ Initialization failed: {st.session_state.processing_error}")
        if st.button("🔄 Retry Initialization"):
            st.session_state.processing_error = None
            st.session_state.processor = None
            st.rerun()
        st.markdown('</div></div>', unsafe_allow_html=True)
        return
    
    # Check if processor is ready
    if st.session_state.processor is None:
        st.warning("⏳ Initializing processor...")
        st.markdown('</div></div>', unsafe_allow_html=True)
        return
    
    # Document Upload Section
    st.markdown('<h2 class="section-title">Document Upload</h2>', unsafe_allow_html=True)
    
    st.info("📋 **Batch Processing Mode**: Upload multiple PDF files to analyze them all at once. Results will be provided as a downloadable CSV file with detailed analysis for each document.")
    
    # File uploader for multiple files
    uploaded_files = st.file_uploader(
        "📄 Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more legal documents to analyze for mineral rights reservations"
    )
    
    if uploaded_files:
        # Show file info
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded:")
        
        # Display file summary
        total_size = sum(file.size for file in uploaded_files)
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size:,} bytes)")
        st.write(f"**Total size:** {total_size:,} bytes")
        
        # Process button
        if st.button("🔍 Analyze All Documents", type="primary"):
            try:
                # Process all documents
                results_df = process_multiple_documents(uploaded_files)
                st.session_state.results_df = results_df
                
                # Show summary statistics
                st.markdown("### 📊 Analysis Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_docs = len(results_df)
                    st.metric("Total Documents", total_docs)
                
                with col2:
                    successful = len(results_df[results_df['processing_status'] == 'Success'])
                    st.metric("Successfully Processed", successful)
                
                with col3:
                    has_reservations = len(results_df[results_df['classification_numeric'] == 1])
                    st.metric("With Reservations", has_reservations)
                
                with col4:
                    high_confidence = len(results_df[results_df['confidence_level'] == 'HIGH'])
                    st.metric("High Confidence", high_confidence)
                
                # Show results preview
                st.markdown("### 📋 Results Preview")
                st.dataframe(
                    results_df[['filename', 'classification', 'confidence_level', 'confidence_score', 'processing_status']],
                    use_container_width=True
                )
                
                # Create download button
                csv_data, csv_filename = create_csv_download(results_df)
                
                st.download_button(
                    label="📥 Download Complete Results (CSV)",
                    data=csv_data,
                    file_name=csv_filename,
                    mime="text/csv",
                    type="primary"
                )
                
                st.markdown("### 📄 CSV File Contents")
                st.info("""
                The CSV file contains the following columns for each analyzed document:
                
                **Basic Information:**
                • `filename` - Original filename
                • `file_size_bytes` - File size in bytes
                • `processing_timestamp` - When the analysis was performed
                
                **Classification Results:**
                • `classification` - Human-readable classification result
                • `classification_numeric` - Numeric classification (0=No Reservations, 1=Has Reservations)
                • `confidence_score` - AI confidence score (0.0 to 1.0)
                • `confidence_level` - HIGH/MEDIUM/LOW confidence classification
                
                **Analysis Details:**
                • `recommendation` - Professional recommendation based on results
                • `llm_explanation` - Detailed AI reasoning for the classification
                • `pages_processed` - Number of document pages analyzed
                • `samples_used` - Number of AI samples used in analysis
                
                **Voting Information:**
                • `total_votes` - Total number of classification votes
                • `no_reservation_votes` - Votes for "no reservations"
                • `has_reservation_votes` - Votes for "has reservations"
                • `vote_ratio_reservations` - Ratio of reservation votes to total votes
                
                **Technical Details:**
                • `early_stopped` - Whether analysis stopped early due to high confidence
                • `text_characters_analyzed` - Number of text characters processed
                • `processing_status` - Success or error information
                """)
                
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")
                st.info("💡 Try uploading different PDFs or check if any files are corrupted.")
    
    # Show previous results if available
    if st.session_state.results_df is not None and not uploaded_files:
        st.markdown("### 📋 Previous Analysis Results")
        st.dataframe(
            st.session_state.results_df[['filename', 'classification', 'confidence_level', 'confidence_score', 'processing_status']],
            use_container_width=True
        )
        
        csv_data, csv_filename = create_csv_download(st.session_state.results_df)
        st.download_button(
            label="📥 Download Previous Results (CSV)",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv"
        )
        
        if st.button("🔄 Start New Analysis"):
            st.session_state.results_df = None
            st.rerun()
    
    # Close content section and main container
    st.markdown('</div></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 