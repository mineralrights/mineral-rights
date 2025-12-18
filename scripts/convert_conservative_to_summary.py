#!/usr/bin/env python3
"""
Turn the simple conservative-run JSON into the richer summary format so that
scripts/visualize_results.py can draw its dashboard.
"""
import json, time, uuid
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
import math, numpy as np

src = Path("evaluation_results/conservative_run_20250728_235554.json")
dst = Path("evaluation_results") / f"oil_gas_evaluation_{datetime.now():%Y%m%d_%H%M%S}.json"

records = json.loads(src.read_text())
for r in records:
    r["true_label"] = 1 if r["folder"] == "reservs" else 0
    r["predicted_label"] = r["prediction"]       # rename to match visualiser needs
    r["file_name"] = r["file"]
    r["category"] = r["folder"]

y_true = [r["true_label"] for r in records]
y_pred = [r["predicted_label"] for r in records]
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

num_reservs = sum(y_true)
reclassified = sum(
    1 for r in records if r["true_label"] == 1 and r["predicted_label"] == 0
)

# ------------------------------------------------------------------
#  compute global aggregates
avg_conf      = sum(r["confidence"] for r in records) / len(records)
avg_samples   = 0.0          # we do not have this info → set to 0.0
avg_time      = 0.0          # idem
success_rate  = 1.0          # every JSON entry is a successful classification
# ------------------------------------------------------------------

summary = {
    "timestamp": datetime.now().isoformat(),
    "dataset_info": {
        "total_documents": len(records),
        "reservs_documents": sum(y_true),
        "no_reservs_documents": len(records) - sum(y_true),
        "data_directory": "N/A"
    },
    "processing_info": {
        "max_samples_per_doc": "N/A",
        "confidence_threshold": "N/A",
        "total_processing_time": "N/A",
        "avg_time_per_doc": "N/A"
    },
    "performance_metrics": {
        "confusion_matrix": {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn)
        },
        "performance_metrics": {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "specificity": tn / (tn + fp) if (tn + fp) else 0
        },
        "oil_gas_specific_metrics": {
            "reclassification_rate": reclassified / num_reservs if num_reservs else 0.0,
            "documents_reclassified": reclassified,
            "originally_reservs_count": num_reservs
        },
        "processing_stats": {
            "total_documents": len(records),
            "successful_classifications": len(records),
            "failed_classifications": 0,
            "avg_confidence": avg_conf,
            "avg_samples_used": avg_samples,      # numeric, not NaN / "N/A"
            "avg_time_per_doc": avg_time          # numeric
        }
    },
    "detailed_results": records
}

dst.write_text(json.dumps(summary, indent=2))
print(f"✓ summary written → {dst}") 