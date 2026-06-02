# Code Availability Text

Suggested manuscript wording:

> The dataset release, metadata, and split manifests are provided separately with the IC4SD-Wood-Eucalyptus Data in Brief resources. The companion research-code repository contains the scripts used to reproduce the four-model benchmark, repeated-seed Split B experiments, leakage audits, calibration/OOD analysis, t-SNE, Grad-CAM, and manuscript tables/figures. Raw images and trained checkpoints are not stored in the code repository because of size; the scripts materialize the required ImageFolder structure from the published split manifests.

Avoid claiming that OOD results are closed-set accuracy. Use wording such as:

> The external *Eucalyptus globulus* experiment was used as an unseen-species stress test, not as a closed-set accuracy evaluation.
