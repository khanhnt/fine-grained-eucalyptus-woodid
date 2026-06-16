# Supplementary Files

This directory stores manuscript supplementary CSV files that are small enough to keep in the code repository.

## Training logs

`training_logs/` contains one per-epoch CSV file for each model/seed combination used in the Split B benchmark. Each file contains:

```text
epoch, train_loss, val_loss, val_acc, val_f1_macro, lr, elapsed_sec
```

The expected set is 12 files: four architectures times three seeds.

## Leakage audit reports

The pHash split-audit CSV files are stored under:

```text
manifests/leakage_audit_reports/
```

They include cross-partition pHash near-duplicate pair listings for Split A and Split B at Hamming-distance thresholds 5 and 10, plus a compact summary comparing the two splits.

## Files not included here

Large files such as raw images, trained checkpoints, generated figures, and contact-sheet images are not committed to this repository. The external OOD image source is not redistributed here; users should obtain it from the original dataset cited in the manuscript.
