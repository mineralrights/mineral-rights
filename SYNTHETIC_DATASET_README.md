# Synthetic Multi-Deed Dataset Generator

This system generates a robust dataset for training deed detection models by combining single deeds into multi-deed PDFs with proper labeling for Google Cloud Document AI.

## Overview

The synthetic dataset generator addresses the challenge of limited multi-deed training data by:
- Randomly sampling from 90 single deeds (49 no-reservations + 41 with reservations)
- Creating multi-deed PDFs with 3-15 deeds per document
- Generating Google Cloud Document AI compatible JSON labels
- Providing train/test splits with known ground truth

## Quick Start

### Generate a Small Test Dataset
```bash
python scripts/run_dataset_generation.py --quick --validate
```

### Generate a Full Production Dataset
```bash
python scripts/run_dataset_generation.py --full --validate

```

(mineral) lauragomez@Mac mineral-rights % python scripts/run_dataset_generation.py --full --validate
🚀 Generating full dataset...
Running: /Users/lauragomez/miniconda3/envs/mineral/bin/python scripts/generate_synthetic_dataset.py --output_dir data/synthetic_dataset --num_train 100 --num_test 25 --seed 42
Loaded 49 no-reservs deeds
Loaded 41 reservs deeds
Total single deeds: 90
Generating 100 training documents and 25 test documents...
Generated 10/100 training documents
Generated 20/100 training documents
Generated 30/100 training documents
Generated 40/100 training documents
Generated 50/100 training documents
Generated 60/100 training documents
Generated 70/100 training documents
Generated 80/100 training documents
Generated 90/100 training documents
Generated 100/100 training documents
Generated 5/25 test documents
Generated 10/25 test documents
Generated 15/25 test documents
Generated 20/25 test documents
Generated 25/25 test documents

Dataset generation complete!
Training documents: 100/100
Test documents: 25/25
Dataset summary saved to: data/synthetic_dataset/dataset_summary.json

Synthetic dataset created successfully in: data/synthetic_dataset

Next steps:
1. Review the generated PDFs and labels
2. Upload to Google Cloud Document AI for training
3. Use the test set for model evaluation


🔍 Validating generated dataset...
Validating synthetic dataset...
Dataset summary found: {'train_documents': 100, 'test_documents': 25, 'total_documents': 125, 'source_single_deeds': 90, 'no_reservs_source': 49, 'reservs_source': 41}

Validation Results:
Valid training documents: 100
Valid test documents: 25
Total valid documents: 125

Dataset Statistics:
  Total pages: 3662
  Total deeds: 1081
  Documents with reservations: 121
  Average deeds per document: 8.6
  Average pages per document: 29.3

✅ Dataset validation passed!


✅ Dataset generation complete!
📁 Dataset location: data/synthetic_dataset
📊 Generated 100 training documents and 25 test documents

📋 Next steps:
1. Review the generated PDFs in the train/pdfs and test/pdfs directories
2. Check the JSON labels in the train/labels and test/labels directories
3. Upload to Google Cloud Document AI for training
4. Use the test set for model evaluation
(mineral) lauragomez@Mac mineral-rights % 


### Generate Custom Dataset
```bash
python scripts/run_dataset_generation.py --custom 100 25 --validate
```

## Dataset Structure

```
data/synthetic_dataset/
├── dataset_summary.json          # Dataset metadata and statistics
├── train/
│   ├── pdfs/                     # Multi-deed PDF files
│   │   ├── synthetic_train_001.pdf
│   │   └── ...
│   └── labels/                   # JSON labels for each PDF
│       ├── synthetic_train_001.json
│       └── ...
└── test/
    ├── pdfs/                     # Test multi-deed PDF files
    │   ├── synthetic_test_001.pdf
    │   └── ...
    └── labels/                   # Test JSON labels
        ├── synthetic_test_001.json
        └── ...
```

## Label Format

Each JSON label file contains:

```json
{
  "doc_id": "synthetic_train_001",
  "source_pdf": "train/pdfs/synthetic_train_001.pdf",
  "attributes": {
    "has_oil_gas_reservations": true,
    "reservation_count": 4,
    "total_deeds": 13,
    "total_pages": 49
  },
  "deed_count": 13,
  "page_starts": [1, 2, 8, 12, ...],
  "deeds": [
    {
      "index": 1,
      "page_start": 1,
      "page_end": 1,
      "source_file": "Carroll DB 130-50.pdf",
      "has_oil_gas_reservations": false,
      "page_count": 1
    },
    ...
  ],
  "validation": {
    "status": "synthetic",
    "issues": [],
    "n_starts_raw": 13,
    "n_starts_normalized": 13,
    "duplicates": {}
  },
  "qc": {
    "status": "synthetic",
    "issues": []
  },
  "metadata": {
    "generated_at": "2025-09-08T16:56:34.966341",
    "generator": "synthetic_dataset_generator",
    "seed": 42,
    "pdf_hash": "sha256_hash_here"
  }
}
```

## Scripts

### 1. `generate_synthetic_dataset.py`
Main dataset generation script with full control over parameters.

**Usage:**
```bash
python scripts/generate_synthetic_dataset.py \
    --output_dir data/synthetic_dataset \
    --num_train 80 \
    --num_test 20 \
    --seed 42
```

**Parameters:**
- `--output_dir`: Output directory for the dataset
- `--num_train`: Number of training documents to generate
- `--num_test`: Number of test documents to generate
- `--seed`: Random seed for reproducibility

### 2. `run_dataset_generation.py`
Simplified interface with preset configurations.

**Usage:**
```bash
# Quick test dataset (10 train, 5 test)
python scripts/run_dataset_generation.py --quick

# Full production dataset (100 train, 25 test)
python scripts/run_dataset_generation.py --full

# Custom dataset
python scripts/run_dataset_generation.py --custom 80 20

# With validation
python scripts/run_dataset_generation.py --full --validate
```

### 3. `validate_synthetic_dataset.py`
Validates the generated dataset for correctness.

**Usage:**
```bash
python scripts/validate_synthetic_dataset.py --dataset_dir data/synthetic_dataset
```

**Validation checks:**
- PDF readability and page counts
- JSON label format compliance
- PDF-label consistency
- Document AI format compatibility

## Dataset Statistics

### Quick Test Dataset (10 train, 5 test)
- **Total documents:** 15
- **Total pages:** ~420
- **Total deeds:** ~120
- **Average deeds per document:** 8.0
- **Average pages per document:** 28.0
- **Documents with reservations:** ~13

### Full Production Dataset (100 train, 25 test)
- **Total documents:** 125
- **Total pages:** ~3,500
- **Total deeds:** ~1,000
- **Average deeds per document:** 8.0
- **Average pages per document:** 28.0
- **Documents with reservations:** ~100

## Source Data

The generator uses:
- **49 single deeds** from `data/no-reservs/` (no oil/gas reservations)
- **41 single deeds** from `data/reservs/` (with oil/gas reservations)
- **Total:** 90 single deeds for sampling

## Google Cloud Document AI Integration

The generated labels are compatible with Google Cloud Document AI training:

1. **Upload PDFs and labels** to your Document AI dataset
2. **Train a custom model** for deed boundary detection
3. **Use the test set** for model evaluation
4. **Deploy the model** for production use

### Label Format Compatibility
- Page-based deed boundaries
- Document-level attributes
- Deed-level metadata
- Validation and QC information

## Quality Assurance

### Built-in Validation
- PDF readability checks
- Label format validation
- PDF-label consistency verification
- Page boundary accuracy

### Reproducibility
- Fixed random seed (default: 42)
- Deterministic sampling
- Consistent output across runs

### Error Handling
- Graceful handling of corrupted PDFs
- Detailed error reporting
- Validation summaries

## Advanced Usage

### Custom Deed Counts
Modify the `sample_deeds_for_document()` method in `generate_synthetic_dataset.py`:

```python
def sample_deeds_for_document(self, min_deeds: int = 5, max_deeds: int = 20) -> List[Tuple[Path, int, bool]]:
    # Customize deed count range
    num_deeds = random.randint(min_deeds, max_deeds)
    # ... rest of the method
```

### Custom Label Format
Extend the `generate_document_ai_label()` method to add custom fields:

```python
def generate_document_ai_label(self, doc_id: str, deed_info: List[Tuple[Path, int, bool]], 
                             output_pdf_path: Path) -> Dict[str, Any]:
    # ... existing code ...
    
    # Add custom fields
    label["custom_attributes"] = {
        "generation_method": "synthetic",
        "complexity_score": self.calculate_complexity(deed_info)
    }
    
    return label
```

## Troubleshooting

### Common Issues

1. **"Source data directories not found"**
   - Ensure `data/no-reservs/` and `data/reservs/` directories exist
   - Check that they contain PDF files

2. **"Cannot read PDF" errors**
   - Some PDFs may be corrupted
   - The generator will skip invalid PDFs and continue

3. **Validation failures**
   - Check PDF-label consistency
   - Verify JSON format compliance
   - Review error messages for specific issues

### Performance Tips

1. **Large datasets:** Use `--full` for production datasets
2. **Quick testing:** Use `--quick` for development
3. **Memory usage:** Process in batches for very large datasets
4. **Storage:** Ensure sufficient disk space for generated PDFs

## Next Steps

1. **Review generated samples** to ensure quality
2. **Upload to Google Cloud Document AI** for training
3. **Train your custom model** using the labeled data
4. **Evaluate on test set** to measure performance
5. **Deploy for production** deed detection

## Support

For issues or questions:
1. Check the validation output for specific errors
2. Review the generated dataset summary
3. Examine sample PDFs and labels manually
4. Adjust parameters as needed for your use case
