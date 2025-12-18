# Mineral Rights Document Analyzer - Web App

A simple web interface for analyzing deed PDFs to determine whether they contain mineral rights reservations.

## Quick Start

1. **Install Dependencies** (if not already done):
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Web App**:
   ```bash
   python run_webapp.py
   ```

3. **Open Your Browser**:
   - Go to `http://localhost:5001`
   - The app will also be accessible from other devices on your network

## How to Use

1. **Upload a PDF**: 
   - Drag and drop a deed PDF onto the upload area, or
   - Click "Choose File" to browse and select a PDF

2. **Wait for Analysis**: 
   - The AI will analyze your document (this may take 1-3 minutes)
   - You'll see a processing indicator while it works

3. **Review Results**:
   - **Classification**: Whether mineral rights reservations were found
   - **Confidence Level**: How certain the AI is about its decision
   - **Explanation**: Plain English explanation of the findings
   - **Recommendation**: What you should do next
   - **Processing Details**: Technical information about the analysis

## Understanding the Results

### Classifications
- ✅ **No Mineral Rights Reservations Found**: The deed appears to transfer all rights including mineral rights
- ⚠️ **Mineral Rights Reservations Found**: The deed contains language reserving some or all mineral rights to the seller

### Confidence Levels
- 🟢 **High (80%+)**: Very confident in the result
- 🟡 **Medium (60-79%)**: Moderately confident, some ambiguity in the document
- 🔴 **Low (<60%)**: Low confidence, document language is unclear or complex

### Important Notes
- This tool provides AI-assisted analysis, not legal advice
- For legal certainty, always consult with a qualified attorney
- The tool works best with clear, well-scanned PDF documents
- Maximum file size: 50MB

## Features

- **Drag & Drop Interface**: Easy file upload
- **Real-time Processing**: See progress as your document is analyzed
- **Plain English Results**: No legal jargon, clear explanations
- **Confidence Scoring**: Know how certain the AI is about its findings
- **Mobile Friendly**: Works on phones and tablets
- **Secure**: Files are processed and immediately deleted

## Technical Details

- Uses Claude AI for document analysis and OCR
- Processes documents page-by-page with early stopping
- Self-consistent sampling for improved accuracy
- Confidence scoring based on multiple analysis factors

## Troubleshooting

**"Missing dependency" error**: Run `pip install -r requirements.txt`

**"API key not found" warning**: The API key is currently hardcoded in the classifier. For production use, set the `ANTHROPIC_API_KEY` environment variable.

**"Port already in use" error**: The app uses port 5001. If this port is busy, you can modify the port number in `app.py` and `run_webapp.py`.

**Slow processing**: Large or complex documents take longer. The system processes up to 8 AI samples per document for accuracy.

**Upload fails**: Ensure your file is a PDF and under 50MB.

## Support

For technical issues or questions about the mineral rights classification system, refer to the main project README.md file. 