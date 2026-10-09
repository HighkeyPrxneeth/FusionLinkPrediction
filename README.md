# Heuristic Fusion for Link Prediction

A coursework study of whether neighborhood scores add information to a Variational Graph Autoencoder. The frozen protocol uses Cora, CiteSeer and PubMed, with five runs each and a separate graph-training, checkpoint-selection, fusion-fitting and test partition.

The complete 15-run benchmark, 12 protocol tests, numerical demo audit and browser checks passed. Fusion improves mean AP over VGAE on all three datasets; GAE remains the strongest main method. See [implementation status](docs/IMPLEMENTATION_STATUS.md) for the measured conclusion and remaining team review.

## Demonstrate the project

```powershell
.\scripts\start_demo.ps1
```

Open **http://127.0.0.1:8768**. Compare all three networks, inspect any saved test pair, visualize its training neighborhood, and examine its numerical fusion contributions. The demo reads audited results and does not retrain. See [the walkthrough](docs/DEMO_GUIDE.md) for a five-minute presentation and viva answers. The server stays on this computer and can be stopped with Ctrl+C.

The report is `output/pdf/SNA_Project_Report.pdf`, with editable text in `REPORT.md`. Team details are intentionally blank in `docs/team.json`. The submission bundle is `output/SNA_Project_Submission.zip`.

## Setup and execution

Create the dedicated conda environment and install the pinned dependencies:

```powershell
.\scripts\setup.ps1
conda activate D:\Projects\SNA-Project\.conda
python -m pytest -q
.\scripts\run_all.ps1
```

The scripts explicitly use the project environment, so activation is optional when invoking them. For direct commands, use `.\.conda\python.exe` if conda activation is unavailable.

Run a short pilot in a separate artifact folder:

```powershell
.\.conda\python.exe -m sna_project.run --datasets Cora --runs 0 --epochs 20 --artifact-folder artifacts_pilot --resume
```

Run or resume the complete protocol:

```powershell
.\.conda\python.exe -m sna_project.run --resume
.\.conda\python.exe -m sna_project.export
```

Interrupt safely; encoder checkpoints are written every ten epochs, and completed methods/runs are reused. A changed source, configuration, dataset or device requires a new artifact folder. Do not edit completed results to make them pass the fingerprint check.

## Protocol and interpretation

All methods use only the 80% training graph. Five percent of original edges select encoder checkpoints, another five percent fit fusion, and ten percent form the final test. Each evaluation partition contains an equal number of true non-edges. Training negatives exclude original positive edges and reserved evaluation pairs. All nodes and features remain available: this is static transductive missing-edge recovery, not future-link prediction.

The methods are CN, Jaccard, Adamic-Adar, Resource Allocation, Preferential Attachment, DeepWalk, GAE, VGAE, heuristics-only logistic regression, VGAE-score-only logistic regression, full fusion, and five leave-one-feature-out variants. AUC and AP are reported as means and sample standard deviations across five paired runs.

The fusion contribution is relative to the selected VGAE base model. Do not claim this is the first combination of graph embeddings with structural features. Balanced candidate-set scores do not establish population probability calibration or all-pairs recommendation performance.

## Outputs

- `artifacts/<dataset>/run_<id>/`: split manifests, seeds, checkpoints, embeddings, predictions, training history, coefficients, timings and metrics.
- `data/processed/<dataset>/statistics.json`: actual graph statistics, source URLs and hashes.
- `results/tables/`: per-run metrics, aggregate comparison, paired differences, subgroup counts, ablations, coefficients and timings.
- `results/figures/`: comparisons, preselected-run ROC/PR curves, ablations, coefficient plots and runtime.
- `results/export_manifest.json`: completeness and output hashes; reported metrics are recomputed from predictions.
- `docs/sources/`: cached public literature sources and retrieval provenance.
- `PROJECT_PLAN.md`: design, scope, assumptions and coursework mapping.

No partial export is labeled complete. `--allow-partial` is available for development only. GPU reductions may vary slightly despite fixed seeds; the environment record documents this limitation. DeepWalk uses one worker and Skip-gram hierarchical softmax.

## Rebuild the report and submission bundle

```powershell
.\.conda\python.exe scripts/audit_demo.py
.\.conda\python.exe scripts/build_report.py
.\.conda\python.exe scripts/build_submission.py
```

Report generation requires all 15 completed, exported experiments. Check the regenerated PDF before submission. The bundle excludes the environment, raw dataset downloads, archived paper full texts and walk text corpora. It retains splits, predictions, embeddings and checkpoints so the demo can work after installing dependencies. `requirements-lock.txt` and `conda-explicit-win64.txt` record this machine's actual installation; `scripts/setup.ps1` installs the selected CUDA PyTorch version before the project requirements.
