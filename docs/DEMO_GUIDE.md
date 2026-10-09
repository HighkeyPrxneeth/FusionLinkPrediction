# Demonstrating the project

Start the demo without retraining:

```powershell
.\scripts\start_demo.ps1
```

Open **http://127.0.0.1:8768** on this computer. Keep the server running during the demonstration. Stop it with Ctrl+C when finished. The demo uses the project conda interpreter and reads the saved benchmark artifacts; it needs no API keys or internet connection after setup.

## A five-minute walkthrough

1. **State the question.** “Does adding five simple neighborhood scores improve the VGAE link-ranking score?” Explain that the experiment hides known citations and tries to recover them.
2. **Compare results.** Show Cora, CiteSeer and PubMed. Switch between AP and ROC AUC. Explain that means and sample standard deviations use five runs, with matching graphs and candidate pairs for every method. Read the actual ranking rather than promising fusion wins.
3. **Explain the controls.** Point out GAE and DeepWalk, then heuristics-only regression and VGAE-score-only regression. Improvement over VGAE alone does not establish that the variational model is necessary.
4. **Inspect a prediction.** Select a hidden real citation. Show its neighborhood, raw heuristics and six numerical fusion contributions. The candidate edge itself is absent from the visible graph.
5. **Show a failure.** Select “Highest-scoring non-edge” or “Lowest-scoring hidden link.” Acknowledge that prediction is imperfect. These choices are deliberately labeled extreme examples, not representative averages.
6. **Show zero overlap.** Select a no-common-neighbor example. Four overlap scores are zero; endpoint degrees and learned representations can still vary.
7. **Describe the information boundary.** Open “How it works”: 80% graph training, 5% checkpoint selection, 5% fusion fitting, 10% test. Explain that no hidden edge enters message passing or heuristic computation.
8. **Close with the actual conclusion.** Use the result tables, feature-removal effects and limitations. This is a controlled extension of VGAE, not the invention of embedding/heuristic fusion or a future-citation system.

## Common viva questions

**Why is this social network analysis?**

It studies relational structure using overlap, degree, neighborhood weighting and graph representations. The chosen benchmarks are citation networks, a form of information network; they are not direct friendship graphs. The methods also apply to social and collaboration graphs, but that generalization has not been experimentally established here.

**What is new relative to the base paper?**

A separate logistic model adds five structural scores to the VGAE decoder score, and controlled ablations investigate their value. Related literature already contains hybrid methods, so novelty is relative to the selected base model.

**Why four partitions instead of the original 85/5/10?**

Checkpoint selection and fitting a supervised fusion model need separate roles. Holding out 5% for each preserves an untouched 10% test partition while ensuring every method sees the same 80% training graph.

**What is the difference between GAE and VGAE?**

GAE produces a deterministic representation. VGAE predicts a mean and standard deviation, samples latent representations during training and includes a KL regularizer. Both use mean/deterministic embedding dot products at evaluation.

**Why use mean embeddings at evaluation?**

They give a stable deterministic ranking score instead of introducing sampling noise into the comparison. This study does not evaluate posterior predictive uncertainty.

**Does a fusion score of 0.9 mean a 90% probability of a real citation?**

Only within the fitted balanced candidate distribution can it be interpreted as a model probability. Population prevalence is much lower, and calibration/all-pairs recommendation were not evaluated. Call it a fusion score in the demonstration.

**Why AP as well as AUC?**

AUC assesses positive-versus-negative ranking and handles tied scores. AP weights precision by recall increments, emphasizing positive retrieval. AP depends on prevalence; subgroup counts and prevalence must accompany it.

**Why not claim that a large coefficient proves a heuristic is important?**

The features are correlated and scaled. Coefficients explain this fitted model, while removal-and-refit experiments test incremental predictive contribution. Neither proves causation.

**How do you prevent leakage?**

Canonical unordered pairs prevent reverse-edge leakage. All methods use only saved training edges. Scalers/classifiers fit on fusion pairs, checkpoints use selection pairs, and test labels are used only for final metrics. Negatives exclude the full original positive graph and reserved pairs. The exporter audits hashes, labels, predictions and scaler fitting.

**Are you exactly reproducing the original VGAE numbers?**

No. Architecture and base methods are reimplemented, but the sampled objective, four-way split and checkpoint policy are documented deviations. The existing methods are compared under our common protocol; published numbers are context only.

**What would you do next?**

Register a new experiment using harder negatives and a temporal or true social/coauthorship graph, then examine encoder budget and objective sensitivity. Preserve the present results and do not select a variant using the current test sets.

## Before submission

Fill the blank fields in `docs/team.json`, then regenerate the report with the project interpreter. The report generator reads measured tables and refuses incomplete experiments. Follow the PDF workflow for regeneration and inspect the rendered pages. Keep the complete artifacts for the demo; the submission ZIP intentionally excludes the conda environment, data downloads and walk text corpora.
