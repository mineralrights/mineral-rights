// Test CSV generation with both old and new logic
const Papa = require('papaparse');

console.log('=== TESTING CSV GENERATION WITH BOTH LOGICS ===');

// Mock backend result
const mockBackendResult = {
  total_pages: 3,
  pages_with_reservations: 1,
  reservation_pages: [2],
  results: [
    { page_number: 1, has_reservations: false, confidence: 0.95, reasoning: 'No oil/gas keywords found' },
    { page_number: 2, has_reservations: true, confidence: 0.98, reasoning: 'Explicit oil and gas rights reservation language detected' },
    { page_number: 3, has_reservations: false, confidence: 0.92, reasoning: 'No substantive oil and gas reservation language found' }
  ],
  processing_method: 'page_by_page',
  filename: 'test_document.pdf'
};

// OLD LOGIC: Single row with all pages
console.log('\n=== OLD LOGIC (CURRENT DEPLOYED) ===');
const oldLogicRows = [];
if (mockBackendResult.results && Array.isArray(mockBackendResult.results)) {
  const row = {
    filename: mockBackendResult.filename || 'document.pdf',
    status: 'done',
    prediction: mockBackendResult.pages_with_reservations > 0 ? 'has_reservation' : 'no_reservation',
    confidence: mockBackendResult.pages_with_reservations > 0 ? 1.0 : 0.0,
    explanation: `Found mineral rights reservations on ${mockBackendResult.pages_with_reservations} pages: ${(mockBackendResult.reservation_pages || []).join(', ')}`,
    processingMode: 'page_by_page',
    pageResults: mockBackendResult.results, // ALL pages in one row
    totalPages: mockBackendResult.total_pages,
    pagesWithReservations: mockBackendResult.reservation_pages || []
  };
  oldLogicRows.push(row);
}

console.log('Old logic rows:', oldLogicRows.length);
console.log('Old logic pageResults length:', oldLogicRows[0].pageResults.length);

// Generate CSV with old logic
const oldCsvData = [];
oldLogicRows.forEach(row => {
  if (row.processingMode === 'page_by_page' && row.pageResults) {
    row.pageResults.forEach(page => {
      oldCsvData.push({
        'Original File': row.filename,
        'Deed Name': `page_${page.page_number}`,
        'Deed Number': page.page_number,
        Status: row.status,
        Prediction: page.has_reservations ? 'has_reservation' : 'no_reservation',
        Confidence: page.confidence ? (page.confidence * 100).toFixed(1) + '%' : '0%',
        'Page Range': `Page ${page.page_number}`,
        'Pages in Deed': 1,
        'Boundary Confidence': '',
        'Has Reservations': page.has_reservations ? 'YES' : 'NO',
        Explanation: page.reasoning || page.explanation || ''
      });
    });
  }
});

console.log('Old logic CSV rows:', oldCsvData.length);
console.log('Old logic CSV:');
console.log(Papa.unparse(oldCsvData));

// NEW LOGIC: Individual rows for each page
console.log('\n=== NEW LOGIC (AFTER DEPLOYMENT) ===');
const newLogicRows = [];
if (mockBackendResult.results && Array.isArray(mockBackendResult.results)) {
  mockBackendResult.results.forEach((page, index) => {
    const row = {
      filename: mockBackendResult.filename || 'document.pdf',
      status: 'done',
      prediction: page.has_reservations ? 'has_reservation' : 'no_reservation',
      confidence: page.confidence || 0,
      explanation: page.reasoning || page.explanation || '',
      processingMode: 'page_by_page',
      pageResults: [page], // Single page in each row
      totalPages: mockBackendResult.total_pages,
      pagesWithReservations: mockBackendResult.reservation_pages || []
    };
    newLogicRows.push(row);
  });
}

console.log('New logic rows:', newLogicRows.length);
console.log('New logic pageResults length per row:', newLogicRows[0].pageResults.length);

// Generate CSV with new logic
const newCsvData = [];
newLogicRows.forEach(row => {
  if (row.processingMode === 'page_by_page' && row.pageResults) {
    row.pageResults.forEach(page => {
      newCsvData.push({
        'Original File': row.filename,
        'Deed Name': `page_${page.page_number}`,
        'Deed Number': page.page_number,
        Status: row.status,
        Prediction: page.has_reservations ? 'has_reservation' : 'no_reservation',
        Confidence: page.confidence ? (page.confidence * 100).toFixed(1) + '%' : '0%',
        'Page Range': `Page ${page.page_number}`,
        'Pages in Deed': 1,
        'Boundary Confidence': '',
        'Has Reservations': page.has_reservations ? 'YES' : 'NO',
        Explanation: page.reasoning || page.explanation || ''
      });
    });
  }
});

console.log('New logic CSV rows:', newCsvData.length);
console.log('New logic CSV:');
console.log(Papa.unparse(newCsvData));

console.log('\n=== SUMMARY ===');
console.log('Both logics should produce the SAME CSV output (3 rows)');
console.log('The difference is in the UI table display:');
console.log('- OLD: 1 row in UI table, 3 rows in CSV');
console.log('- NEW: 3 rows in UI table, 3 rows in CSV');
