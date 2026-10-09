"""Build the editable report and typeset PDF from audited measurements."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
NAVY, BLUE, GRAY = colors.HexColor('#173042'), colors.HexColor('#315de0'), colors.HexColor('#526a7a')
FONT_ROOT = Path(matplotlib.get_data_path()) / 'fonts/ttf'
for name, file in [('Body', 'DejaVuSans.ttf'), ('Bold', 'DejaVuSans-Bold.ttf'), ('Mono', 'DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_ROOT / file)))
styles = {
    'body': ParagraphStyle('body', fontName='Body', fontSize=9.8, leading=15, spaceAfter=9, textColor=NAVY),
    'h1': ParagraphStyle('h1', fontName='Bold', fontSize=19, leading=24, spaceBefore=14, spaceAfter=14, textColor=NAVY, keepWithNext=True),
    'h2': ParagraphStyle('h2', fontName='Bold', fontSize=12, leading=17, spaceBefore=12, spaceAfter=8, textColor=BLUE, keepWithNext=True),
    'small': ParagraphStyle('small', fontName='Body', fontSize=8, leading=11, spaceAfter=8, textColor=GRAY),
    'table': ParagraphStyle('table', fontName='Body', fontSize=8, leading=11, textColor=NAVY),
    'equation': ParagraphStyle('equation', fontName='Mono', fontSize=8.5, leading=13, spaceBefore=4, spaceAfter=10, leftIndent=12, textColor=NAVY),
    'title': ParagraphStyle('title', fontName='Bold', fontSize=29, leading=37, spaceAfter=18, textColor=NAVY),
}
story, markdown = [], []


def p(text, style='body'):
    story.append(Paragraph(html.escape(text).replace('\n', '<br/>'), styles[style]))
    markdown.append(text + '\n')


def heading(text, level=1, newpage=False):
    if newpage:
        story.append(PageBreak())
    story.append(Paragraph(html.escape(text), styles['h1' if level == 1 else 'h2']))
    markdown.append('#' * (level + 1) + ' ' + text + '\n')


def table(headers, rows, widths=None):
    content = [[Paragraph(html.escape(str(v)), styles['table']) for v in row] for row in [headers] + rows]
    t = Table(content, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eff7')),
                          ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 7),
                          ('RIGHTPADDING', (0, 0), (-1, -1), 7), ('TOPPADDING', (0, 0), (-1, -1), 7),
                          ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                          ('LINEBELOW', (0, 0), (-1, 0), .7, BLUE),
                          ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#d8e2e9'))]))
    story.extend([t, Spacer(1, 12)])
    markdown.extend(['| ' + ' | '.join(map(str, headers)) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'])
    markdown.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
    markdown.append('')


def figure(filename, caption, width=480):
    path = ROOT / 'results/figures' / filename
    iw, ih = ImageReader(str(path)).getSize()
    image = Image(str(path), width=width, height=width * ih / iw)
    story.append(KeepTogether([image, Spacer(1, 5), Paragraph(html.escape(caption), styles['small'])]))
    markdown.append(f'![{caption}]({path.relative_to(ROOT).as_posix()})\n')


def equation(text):
    p(text, 'equation')


def diagram():
    draw = Drawing(485, 205)
    boxes = [(0, 145, 485, 45, 'Canonical undirected graph + row-normalized attributes'),
             (0, 77, 145, 48, '80% graph: GAE / VGAE'), (168, 77, 145, 48, '80% graph: heuristics'),
             (337, 77, 148, 48, '80% graph: DeepWalk'),
             (45, 7, 395, 48, '5% checkpoint selection → 5% fusion fitting → 10% test')]
    for x, y, w, h, label in boxes:
        draw.add(Rect(x, y, w, h, rx=5, fillColor=colors.HexColor('#edf2fb'), strokeColor=colors.HexColor('#b6c9dc')))
        draw.add(String(x + w / 2, y + h / 2, label, fontName='Body', fontSize=8.2, textAnchor='middle', fillColor=NAVY))
    for x in [73, 241, 410]:
        draw.add(Line(x, 145, x, 125, strokeColor=BLUE))
        draw.add(Polygon([x-3,130,x,125,x+3,130], fillColor=BLUE, strokeColor=BLUE))
        draw.add(Line(x, 77, x, 55, strokeColor=BLUE))
    story.extend([draw, Spacer(1, 10)])
    markdown.append('```mermaid\nflowchart TD\n A[Canonical graph and features] --> B[80% training graph]\n B --> C[GAE / VGAE embeddings]\n B --> D[Five neighborhood heuristics]\n B --> E[DeepWalk]\n C --> F[5% checkpoint selection]\n F --> G[5% fusion fitting]\n D --> G\n G --> H[10% held-out test: AUC and AP]\n E --> H\n```\n')


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#d8e2e9'))
    canvas.line(54, 39, 541, 39)
    canvas.setFont('Body', 7.5)
    canvas.setFillColor(GRAY)
    canvas.drawString(54, 26, 'SNA coursework · Heuristic fusion for link prediction')
    canvas.drawRightString(541, 26, str(doc.page))
    canvas.restoreState()


def main():
    manifest = json.loads((ROOT / 'results/export_manifest.json').read_text())
    if not manifest['complete'] or manifest['completed'] != 15:
        raise RuntimeError('Final report requires all 15 audited runs')
    summary = pd.read_csv(ROOT / 'results/tables/summary.csv')
    paired = pd.read_csv(ROOT / 'results/tables/paired_differences.csv')
    raw = pd.read_csv(ROOT / 'results/tables/per_run_metrics.csv')
    timings = pd.read_csv(ROOT / 'results/tables/timings.csv')
    env = json.loads((ROOT / 'artifacts/environment.json').read_text())
    hardware = json.loads((ROOT / 'artifacts/hardware.json').read_text(encoding='utf-8-sig')) if (ROOT / 'artifacts/hardware.json').exists() else {}
    config = json.loads((ROOT / 'artifacts/config.json').read_text())
    team = json.loads((ROOT / 'docs/team.json').read_text())
    notes = json.loads((ROOT / 'docs/literature_notes.json').read_text())
    datasets = config['datasets']
    stats = {d: json.loads((ROOT / 'data/processed' / d / 'statistics.json').read_text()) for d in datasets}
    score = lambda d, m: summary[(summary.dataset == d) & (summary.method == m)].iloc[0]
    comparison = paired[(paired.baseline == 'VGAE') & (paired.metric == 'ap')].set_index('dataset')
    gains = ', '.join(f'{d} {100*comparison.loc[d, "mean_difference"]:+.2f}' for d in datasets)
    positives = sum(comparison.mean_difference > 0)
    OUT.mkdir(parents=True, exist_ok=True)
    title = 'Fusing Neighborhood Heuristics with Variational Graph Autoencoders for Link Prediction'
    markdown.append('# ' + title + '\n')
    story.append(Spacer(1, 50))
    p('SOCIAL NETWORK ANALYSIS · COURSEWORK PROJECT', 'small')
    story.append(Paragraph(title, styles['title']))
    p('A reproducible study on Cora, CiteSeer and PubMed', 'h2')
    p('Three benchmarks · Five runs per benchmark · Held-out-pair demo', 'body')
    story.append(Spacer(1, 24))
    table(['Team no.', 'Member name', 'Registration number'],
          [[team['team_number'] if i == 0 else '', member['name'], member['registration_number']] for i, member in enumerate(team['members'])], [85, 230, 170])
    p('Base paper: Thomas N. Kipf and Max Welling, “Variational Graph Auto-Encoders,” 2016 [1].', 'body')
    p('Team identity fields are intentionally blank and can be filled in docs/team.json before regenerating this report.', 'small')
    p('Experiment status: all 15 runs completed and audited. Source and data fingerprints, saved predictions, split invariants and fusion preprocessing are retained in the reproducibility package.', 'body')
    p('Prepared: ' + datetime.now(timezone.utc).strftime('%d %B %Y') + '. This report contains measured local experiments; literature results are identified separately.', 'small')

    heading('1. Abstract', newpage=True)
    abstract = ('Link prediction estimates missing relationships from observed network structure. This project tests whether interpretable neighborhood information improves a Variational Graph Autoencoder on three citation benchmarks: Cora, CiteSeer and PubMed. We combine its embedding dot product with Common Neighbors, Jaccard, Adamic-Adar, Resource Allocation and Preferential Attachment using logistic regression. A fixed, explicitly documented four-part edge protocol separates graph training, checkpoint selection, fusion fitting and final testing. Every method uses the same training graph and balanced test candidates. Independent implementations of DeepWalk, deterministic graph autoencoders and standalone heuristics provide controlled comparisons. Five runs per dataset measure ROC AUC and average precision, while feature-removal ablations and common-neighbor subgroups examine where fusion helps. Saved splits, predictions, model checkpoints and environment records support reproducibility. The local demo exposes both results and individual score calculations. Findings are limited to static, transductive missing-edge recovery; they establish neither future-link forecasting nor calibrated probabilities across all possible node pairs.')
    if len(abstract.split()) != 150:
        raise RuntimeError(f'Abstract must be 150 words; got {len(abstract.split())}')
    p(abstract)
    p('Keywords: link prediction; social network analysis; variational graph autoencoder; neighborhood heuristics; reproducibility.', 'small')
    heading('2. Introduction', level=2)
    paragraphs = [
        'Networks represent entities as nodes and relationships as edges. A social friendship network connects people, a collaboration network connects authors, and a citation network connects papers through references. Although their meanings differ, all exhibit structural patterns such as shared neighborhoods, unequal degree distributions and clusters of related entities. Social network analysis studies these patterns to understand how relationships are organized. Link prediction turns that structural information into a measurable task: given a partially observed network, rank pairs that may have a missing relationship. This can support recommendation, data completion and the prioritization of further observation. The present study uses citation networks as accessible attributed benchmarks; it does not equate citations with friendships.',
        'Simple neighborhood methods provide a useful starting point. Common Neighbors counts shared adjacent nodes, Jaccard adjusts overlap for neighborhood size, Adamic-Adar and Resource Allocation discount high-degree intermediaries, and Preferential Attachment uses endpoint degrees. These features connect prediction to recognizable network mechanisms, including triadic closure and hub connectivity. Their limitation is equally concrete: when two endpoints share no observed neighbor, four of the five measures produce identical zero scores. Missing edges, disconnected components and sparse observations can therefore hide the very evidence on which these methods rely. Degree-based scores remain available, but popularity alone may create many false positives.',
        'Representation learning supplies another view. DeepWalk uses random walks and a Skip-gram objective to embed nodes whose structural contexts are similar [2]. A graph autoencoder uses graph convolutions and node attributes to learn representations that reconstruct observed adjacency. VGAE adds a Gaussian latent-variable model to that framework [1]. Its inner-product decoder is compact and easy to implement, yet the dot product does not explicitly expose pairwise neighborhood overlap. The question addressed here is whether a small supervised fusion model can recover useful local information that the VGAE score leaves unresolved. A deterministic GAE comparator also tests whether the variational model is the strongest simple encoder under the chosen training budget.',
        'Combining structural predictors with embeddings is established research. The journal review includes attribute-preserving autoencoders, topology-aware attention, motif representations and stacking methods [3]-[12]. Our contribution is therefore an extension relative to the selected VGAE base paper and a controlled experimental study, rather than a claim to invent hybrid link prediction. The fusion classifier uses six input values and a separate fitting partition. This keeps its calculation inspectable and permits matched controls: a model using only the five heuristics and another using only the VGAE score. Leave-one-heuristic-out variants examine incremental effects without requiring a new encoder architecture.',
        'Credible comparison requires clear information boundaries. Randomly hiding an edge is insufficient if a later transformation places it back into the encoder graph or if a scaler is fitted using test examples. We explicitly fix the training graph for every method, reserve separate edges for checkpoint selection and fusion fitting, and evaluate on the same test pairs. Negative pairs are sampled from the complement of the original graph, preventing a hidden true edge from being mislabeled during training. All nodes and attributes remain visible, so this is transductive recovery of randomly missing undirected edges. It is not a chronological forecast and does not address unseen-node generalization.',
        'The project evaluates three benchmarks and five paired runs per benchmark using ROC AUC and average precision. The run variation includes both edge splits and random initialization. Tables report means and sample standard deviations; subgroup analysis retains counts and prevalence so that changes in candidate difficulty remain visible. The demo lets a viewer inspect a held-out pair, its visible neighborhood and the numerical fusion decomposition. The report presents results whether or not fusion wins, because a reliable answer about the contribution of local structure is more useful than a favorable score produced by inconsistent evaluation.'
    ]
    for paragraph in paragraphs:
        p(paragraph)

    heading('3. Base paper and project contribution', newpage=True)
    heading('3.1 Major contributions of the base paper', 2)
    for text in [
        '1. A graph latent-variable model with a factorized Gaussian approximate posterior, parameterized by graph convolutions.',
        '2. An inner-product Bernoulli decoder that relates latent node representations to adjacency reconstruction.',
        '3. Variational training with a reconstruction term and a prior-matching KL term, alongside a deterministic GAE alternative.',
        '4. Link-prediction experiments on attributed citation graphs and comparisons with structural representation baselines.']:
        p(text)
    p('These describe the base architecture. Our sampled loss, graph partition and checkpoint policy are deliberately documented deviations, so the experiment is a controlled reimplementation rather than an exact numerical reproduction of the original table.', 'small')
    heading('3.2 Contributions relative to the base paper', 2)
    p('1. A post-encoder fusion model combines the VGAE score and five structural heuristics. The encoder is unchanged and the fusion model is fitted on a distinct hidden-edge partition.')
    p('2. A reproducible comparison includes matched GAE/DeepWalk baselines, single-source fusion controls, five feature-removal ablations, zero-common-neighbor analysis, prediction-level audits and an interactive demonstration.')
    p('Scope of novelty: this is a coursework extension of VGAE. Related journal methods already combine graph representations with structural information. A positive result supports the tested mechanism on these graphs; it does not establish state-of-the-art performance or universal benefit.')

    heading('4. Literature review: ten recent journal papers', newpage=True)
    p('The assignment specifies 2022 onwards. The following ten entries are journal publications within that range, and each description is based on accessible full text. Surveys and conference preprints from the preliminary shortlist are not counted. The notes distinguish reported advantages from protocol limitations identified in this review.')
    references = []
    for index, note in enumerate(notes, start=3):
        meta = json.loads((ROOT / 'docs/sources' / (note['key'] + '_metadata.json')).read_text())['message']
        authors = ', '.join((a.get('given', '') + ' ' + a.get('family', '')).strip() for a in meta.get('author', []))
        year = meta['published']['date-parts'][0][0]
        venue = meta['container-title'][0]
        title_paper = meta['title'][0]
        year_display = '2025 issue; online 2024' if note['key'] == 'se4lp' else str(year)
        heading(f'[{index}] {title_paper}', 2)
        p(f'{authors}. {venue}, {year_display}. DOI: {meta["DOI"]}.', 'small')
        for label, key in [('Methodology', 'method'), ('Datasets', 'datasets'), ('Performance metrics', 'metrics'), ('Advantages', 'advantage'), ('Limitations and relevance', 'limitation')]:
            p(label + ': ' + note[key])
        p('Evidence: ' + note['evidence'], 'small')
        volume = meta.get('volume', '')
        pages = meta.get('page', meta.get('article-number', ''))
        references.append(f'[{index}] {authors}, “{title_paper},” {venue}, vol. {volume}, {pages}, {year_display}, doi: {meta["DOI"]}. https://doi.org/{meta["DOI"]}')

    heading('5. Proposed methodology', newpage=True)
    p('The pipeline begins with public Planetoid data. Reverse copies are collapsed into canonical unordered edges and graph self-loops are removed. All node attributes are row normalized. A reproducible partition assigns approximately 80% of unique edges to graph training, 5% to checkpoint selection, 5% to fusion fitting and 10% to testing, with integer rounding recorded in each manifest. Balanced non-edge candidates are sampled separately for the three evaluation partitions. The same 80% graph remains visible at training and evaluation for all methods.')
    p('GAE and VGAE train for 200 epochs, selecting the highest selection-set AP among checkpoints evaluated every ten epochs. DeepWalk learns from training-graph random walks. Neighborhood heuristics are computed only on that graph. The selected VGAE mean embeddings score the fusion and test candidates; the fusion partition fits a feature scaler and regularized logistic classifier. Once fitted, these transformations score the untouched test partition. All methods, controls and ablations share the same candidate labels within a run.')
    diagram()
    heading('5.1 Module responsibilities', 2)
    table(['Module', 'Input and responsibility'], [
        ['Data', 'Public attributes and edges → normalized features, canonical graph, graph statistics and source hashes.'],
        ['Split', 'Graph and seed → four disjoint positive partitions, disjoint evaluation negatives and invariants.'],
        ['Encoders', 'Training graph and attributes → selected GAE/VGAE node embeddings.'],
        ['Structural scoring', 'Training graph and candidate pairs → five raw neighborhood/degree scores.'],
        ['DeepWalk', 'Training graph → streamed walks, Skip-gram embeddings and dot-product scores.'],
        ['Fusion', 'Hidden fusion pairs → fitted scaler/logistic model; test pairs → frozen predictions.'],
        ['Export and demo', 'Predictions and manifests → audited tables, plots and inspectable held-out examples.']], [95, 390])
    heading('5.2 Neighborhood heuristics', 2)
    p('Let Γ(u) be the neighbors of u in the training graph and d(u)=|Γ(u)|. Each candidate has endpoints u<v. The following scores quantify different aspects of local structure:')
    for text in ['CN(u,v) = |Γ(u) ∩ Γ(v)|', 'JC(u,v) = |Γ(u) ∩ Γ(v)| / |Γ(u) ∪ Γ(v)|',
                 'AA(u,v) = Σ[w ∈ Γ(u) ∩ Γ(v)] 1 / log d(w)',
                 'RA(u,v) = Σ[w ∈ Γ(u) ∩ Γ(v)] 1 / d(w)', 'PA(u,v) = d(u) × d(v)']:
        equation(text)
    p('Jaccard is defined as zero when the union is empty. For distinct endpoints, a common neighbor necessarily has degree at least two, making the Adamic-Adar logarithm well defined. CN captures overlap, JC controls for neighborhood size, AA/RA discount popular intermediaries, and PA retains degree information even when overlap is zero. Tests compare these formulas with hand calculations and NetworkX.')
    heading('5.3 Graph encoding and variational objective', 2)
    equation('Â = D̃^(-1/2) (A_train + I) D̃^(-1/2)')
    equation('H = ReLU(Â X W0); μ = Â H Wμ; log σ = Â H Wσ')
    equation('z_i = μ_i + σ_i ⊙ ε_i,  ε_i ~ Normal(0,I)')
    equation('s_VGAE(u,v) = μ_uᵀ μ_v;  p(u,v) = sigmoid(s(u,v))')
    p('The shared hidden layer has 32 units and each latent representation has 16 units. GAE uses a deterministic output head. VGAE has separate mean and log-standard-deviation heads and samples z during training; evaluation uses μ. Encoder self-loops are normalization operations, not candidate edges. Cached adjacency is safe because every encoder invocation uses the same fixed training graph.')
    equation('KL_mean = (1 / 2N) Σ_i Σ_j [μ_ij² + σ_ij² − 1 − 2 log σ_ij]')
    equation('L = mean_pos softplus(−z_uᵀz_v)')
    equation('    + mean_neg softplus(z_uᵀz_v) + KL_mean / N')
    p('GAE omits the KL term. Every epoch samples as many distinct non-edges as training positives, excluding all original positive edges and all reserved evaluation negatives. The two reconstruction means are balanced sampled losses. KL_mean/N follows the documented PyG-style scaling; it is not claimed to be the exact original dense weighted-adjacency objective. Log-standard-deviation, rather than log-variance, is used consistently in both sampling and KL calculation.')
    heading('5.4 Fusion and controls', 2)
    equation('f(u,v) = [s_VGAE, log(1+CN), JC, log(1+AA),')
    equation('          log(1+RA), log(1+PA)]')
    equation('f̂_j = (f_j − mean_fusion,j) / scale_fusion,j')
    equation('p_fusion(u,v) = sigmoid(b + Σ_j β_j f̂_j)')
    p('Log transforms reduce the dynamic range of count/degree scores; Jaccard is already bounded. Scaling and logistic regression are fitted only on fusion candidates. Logistic regression uses L2 regularization, C=1, the lbfgs solver and a 1,000-iteration limit. The six-feature model is accompanied by heuristics-only regression, VGAE-score-only regression and five versions dropping one heuristic. Each ablation refits its scaler and classifier on the same fusion pairs. Coefficients and score contributions explain the fitted calculation but are not causal importance measures.')
    heading('5.5 Difference from existing work', 2)
    p('Compared with base VGAE, this pipeline adds a small supervised pair-scoring stage without modifying its encoder. Compared with motif, subgraph and topology-aware neural methods, it avoids a new graph architecture and uses six fixed scalar features. Compared with temporal stacking, it uses a single partially observed static graph. The emphasis is a clearly separated fitting protocol and matched controls, rather than architecture novelty. Fusion consumes labeled hidden pairs, and therefore its additional supervision is an explicit part of the comparison.')

    heading('6. Experimental setup', newpage=True)
    table(['Parameter', 'Setting'], [
        ['Graph protocol', 'Undirected, unweighted, transductive; 80/5/5/10 positive edges.'],
        ['Candidate negatives', '1:1 with positives; complement of original graph; disjoint reserved pairs.'],
        ['Runs', 'Five split/initialization runs per dataset; run IDs 0-4.'],
        ['GCN encoder', 'Input feature dimension → 32 hidden → 16 latent; ReLU; no dropout.'],
        ['Optimizer / budget', 'Adam; learning rate 0.01; 200 epochs; best selection AP every 10 epochs.'],
        ['VGAE inference', 'Mean embeddings; raw dot-product ranking score.'],
        ['DeepWalk', '128 dimensions; 10 walks/node; length 80; context 10; 1 epoch.'],
        ['DeepWalk objective', 'Skip-gram hierarchical softmax; negative=0; sample=0.001; one worker.'],
        ['Fusion', 'StandardScaler + L2 logistic regression; C=1; lbfgs; max_iter=1000.'],
        ['Randomness', 'Separate stable split, initialization, negative-sampling and walk seeds.']], [130, 355])
    p('Run seeds derive from base_seed + 1000×run_id, with distinct offsets for the six random processes. Their realized values are saved per run. Paired comparisons use identical candidate pairs and graph splits for all methods. A separate 20-epoch Cora pilot checked execution; it is excluded from final tables, and its test scores did not choose model hyperparameters. The complete protocol uses the predeclared 200-epoch configuration.')
    heading('6.1 Actual environment and tools', 2)
    table(['Item', 'Recorded environment'], [
        ['Conda environment', '.conda under the project root; all experiments use its Python interpreter.'],
        ['Python / operating system', env['python'] + ' / ' + env['platform']],
        ['CPU / memory', str(hardware.get('cpu', env['processor'])) + ' / ' + str(hardware.get('ram_gb', 'not recorded')) + ' GB RAM; ' + str(env['cpu_count']) + ' logical processors.'],
        ['GPU / CUDA', str(env['gpu']) + ' / CUDA ' + str(env['cuda_runtime'])],
        ['PyTorch / PyG', env['packages']['torch'] + ' / ' + env['packages']['torch-geometric']],
        ['NumPy / SciPy', env['packages']['numpy'] + ' / ' + env['packages']['scipy']],
        ['scikit-learn / pandas', env['packages']['scikit-learn'] + ' / ' + env['packages']['pandas']],
        ['Gensim / Numba', env['packages']['gensim'] + ' / ' + env['packages']['numba']],
        ['Other tools', 'NetworkX: graph audits; Matplotlib: figures; ReportLab: PDF; standard-library HTTP server: demo.']], [130, 355])
    p('The installed GPU wheel was obtained from a compatible local package cache. The environment is new and separate from other projects. Exact package versions are in requirements-lock.txt and conda-explicit-win64.txt. Seeds and deterministic-algorithm warnings improve reproducibility, but GPU reductions may differ slightly across hardware/software. A CPU interruption test verifies exact restoration on the test graph; it does not claim cross-device bitwise identity.')
    p('Cora, CiteSeer and PubMed run 0 were executed sequentially. The remaining four independent PubMed trials used separate processes concurrently to reduce turnaround time. Each DeepWalk trial retained one training worker and each run kept its own directory, seeds and checkpoints. This scheduling change did not alter the model or split protocol. Runtime comparisons are descriptive and reflect concurrent workload.')

    heading('7. Benchmarks and evaluation metrics', newpage=True)
    table(['Dataset', 'Nodes', 'Unique edges', 'Features', 'Components', 'Mean degree'],
          [[d, f'{stats[d]["nodes"]:,}', f'{stats[d]["unique_edges"]:,}', stats[d]['features'], stats[d]['components'], f'{stats[d]["mean_degree"]:.3f}'] for d in datasets], [65, 65, 85, 70, 85, 115])
    for d in datasets:
        s = stats[d]
        p(f'{d}: papers are nodes and citation relationships are symmetrized edges. Data source: {s["source"]}. The measured density is {s["density"]:.6f}, average clustering is {s["average_clustering"]:.4f}, and there are {s["isolates"]} isolated nodes after canonicalization.', 'body')
    p('Raw-file hashes, retrieval timestamps, normalized-feature hashes and canonical-edge hashes are recorded in each statistics file. Published tables often list different edge counts because of dataset versions, loops, duplicate edges or component filtering. We compare locally rerun methods on our actual canonical graph; printed paper numbers are not treated as matched baselines.')
    table(['Dataset', 'Training edges', 'Selection positives', 'Fusion positives', 'Test positives / negatives'],
          [[d] + [json.loads((ROOT / f'artifacts/{d}/run_0/split_manifest.json').read_text())['partitions'][k]['count'] for k in ['train_pos','selection_pos','fusion_pos']] + [str(json.loads((ROOT / f'artifacts/{d}/run_0/split_manifest.json').read_text())['partitions']['test_pos']['count']) + ' / same'] for d in datasets], [65, 95, 110, 110, 105])
    heading('7.1 ROC AUC', 2)
    equation('AUC = P(s_positive > s_negative) + 0.5 P(equal scores)')
    p('AUC measures pairwise ranking between a randomly selected positive and negative candidate, giving half credit to ties. This matters for discrete heuristics, which can assign zero to many pairs. It is computed from saved predictions with scikit-learn roc_auc_score.')
    heading('7.2 Average precision', 2)
    equation('AP = Σ_k (Recall_k − Recall_(k−1)) × Precision_k')
    p('AP is computed with average_precision_score using threshold-wise recall increments. It is not trapezoidal PR area. Balanced full-test prevalence is 0.5; subgroup prevalence can be very different, so each subgroup reports its counts and positive fraction. A group containing only one class has undefined AUC/AP for this comparative analysis and is recorded as NA. Scores cannot be compared as if they came from identical prevalence or candidate difficulty.')
    p('Across runs, means and sample standard deviations use ddof=1. Each paired difference subtracts a baseline score from fusion within the same run. Five runs are a small sample, and overlapping splits from one graph do not represent independent real-world networks. We report wins and per-run differences without claiming a universal or statistically established improvement.')

    heading('8. Results and analysis', newpage=True)
    p(f'All 15 predeclared experiments completed. Fusion minus VGAE AP differences, in percentage points: {gains}. The mean difference is positive on {positives} of three datasets. This is an observation under the frozen protocol; the stronger baselines and controls below determine how far the claim can go.')
    methods = ['CN','JC','AA','RA','PA','DeepWalk','GAE','VGAE','Heuristics LR','VGAE score LR','Fusion']
    for d in datasets:
        heading(f'8.1 {d}: controlled comparison', 2)
        table(['Method', 'ROC AUC (%) mean ± SD', 'AP (%) mean ± SD'],
              [[m, f'{100*score(d,m).auc_mean:.2f} ± {100*score(d,m).auc_std:.2f}', f'{100*score(d,m).ap_mean:.2f} ± {100*score(d,m).ap_std:.2f}'] for m in methods], [150, 168, 167])
        best = max(methods, key=lambda m: score(d,m).ap_mean)
        p(f'The highest mean AP among the main methods is {best} ({100*score(d,best).ap_mean:.2f}%). Fusion AP is {100*score(d,"Fusion").ap_mean:.2f}%; GAE AP is {100*score(d,"GAE").ap_mean:.2f}%; DeepWalk AP is {100*score(d,"DeepWalk").ap_mean:.2f}%. Improvement relative to VGAE alone therefore must not be confused with being the strongest method in the benchmark.')
        figure(d.lower() + '_comparison.png', f'{d}: matched candidate sets, five-run means and sample SD.')
    heading('8.2 Paired differences and single-source controls', 2)
    part = paired[paired.metric == 'ap']
    table(['Dataset', 'Baseline', 'Fusion − baseline AP (pp)', 'Wins / runs'],
          [[r.dataset, r.baseline, f'{100*r.mean_difference:+.2f} ± {100*r.std_difference:.2f}', f'{r.wins}/{r.runs}'] for r in part.itertuples()], [65, 130, 195, 95])
    p('The VGAE-score-only regression controls for fitting a new intercept and scale. A positive coefficient leaves ranking unchanged, while a negative coefficient reverses it; this is not an additional graph model. The heuristics-only model tests whether embeddings add anything to the same labeled fitting set. Comparison against GAE and DeepWalk is essential because a gain over a relatively weak VGAE does not imply the variational representation is necessary.')
    heading('8.3 Feature-removal ablation', 2)
    for d in datasets:
        full = score(d,'Fusion').ap_mean
        rows = []
        for feature in ['CN','JC','AA','RA','PA']:
            variant = score(d, 'Fusion minus ' + feature)
            rows.append([feature, f'{100*variant.ap_mean:.2f} ± {100*variant.ap_std:.2f}', f'{100*(full-variant.ap_mean):+.2f}'])
        heading(d, 2)
        table(['Removed feature', 'Variant AP (%)', 'Full − variant AP (pp)'], rows, [160, 160, 165])
        figure(d.lower() + '_ablation.png', f'{d}: each reduced model refits on the same fusion partition.')
    p('A positive full-minus-variant value supports incremental benefit within the fitted feature set. A negative value means removal improves the observed mean. CN, AA and RA are strongly related, so small removal effects may indicate redundancy rather than irrelevance. These post-evaluation diagnostics were not used to replace the frozen full-fusion model with a test-selected variant.')
    heading('8.4 Common-neighbor subgroup analysis', 2)
    subgroup_rows = []
    for d in datasets:
        for group in ['CN=0','CN>0']:
            subset = raw[(raw.dataset == d) & (raw.group == group) & (raw.method == 'Fusion')]
            v = raw[(raw.dataset == d) & (raw.group == group) & (raw.method == 'VGAE')]
            subgroup_rows.append([d, group, f'{subset["count"].mean():.1f}', f'{subset.prevalence.mean():.3f}',
                                  f'{100*subset.ap.mean():.2f}' if subset.ap.notna().any() else 'NA',
                                  f'{100*v.ap.mean():.2f}' if v.ap.notna().any() else 'NA'])
    table(['Dataset','Group','Mean count','Prevalence','Fusion AP (%)','VGAE AP (%)'], subgroup_rows, [65,60,80,80,100,100])
    p('Counts and prevalence are mean run values, and subgroup AP is averaged only where defined. The complete per-run table retains the exact counts and NA flags. CN>0 candidates often have much higher positive prevalence than CN=0 candidates; this difference alone can raise AP. Compare methods within a group, rather than treating between-group AP as a pure measure of model difficulty. PA can still vary in the zero-overlap group, whereas CN, JC, AA and RA are identically zero there.')
    heading('8.5 Curves, coefficients and runtime', 2)
    figure('cora_curves.png', 'Cora ROC and PR curves from preselected run 0; this is an illustrative run, not the best run.')
    figure('cora_coefficients.png', 'Cora standardized fusion coefficients: mean ± sample SD; correlated features prevent causal interpretation.')
    p('The same curve and coefficient figures are exported for CiteSeer and PubMed. The demo exposes any run and any saved test-pair index, including high-scoring non-edges and low-scoring true hidden links. Its neighborhood diagram uses only training edges and caps displayed neighbor nodes; heuristic values always use the full training graph.')
    runtime_rows = []
    for d in datasets:
        t = timings[timings.dataset == d]
        values = t.groupby('stage').seconds.mean()
        runtime_rows.append([d] + [f'{values[k]:.1f}' for k in ['GAE','VGAE','DeepWalk','heuristics','fusion_fits']])
    table(['Dataset','GAE (s)','VGAE (s)','DeepWalk (s)','Heuristics (s)','Fusion fits (s)'], runtime_rows, [65,80,80,90,85,85])
    figure('runtime.png', 'Measured mean stage runtime per run on this laptop. Concurrent workload and thermal state can affect timings.')
    p('Encoder runtime includes checkpoint writes and evaluation; DeepWalk includes corpus generation, vocabulary construction and training. Heuristics cover fusion/test candidates rather than all node pairs. Fusion runtime includes all controls and ablations. These are measured wall-clock stages, not hardware-independent complexity estimates.')
    heading('8.6 Comparison with existing works and published context', 2)
    p('The same-dataset comparison requirement is met by independently running DeepWalk [2] and the GAE/VGAE methods from [1] in the matched local protocol. The main tables compare their actual saved predictions against fusion on all three datasets. Raw heuristics provide additional existing-method baselines. Published VGAE and recent journal numbers are contextual because graph versions, training-visible edges, objective, negative sampling and dispersion conventions differ.')
    table(['Historical method [1]', 'Cora AUC / AP (%)', 'CiteSeer AUC / AP (%)', 'PubMed AUC / AP (%)'],
          [['DeepWalk','83.1 / 85.0','80.5 / 83.6','84.4 / 84.1'],['GAE','91.0 / 92.0','89.5 / 89.9','96.4 / 96.5'],['VGAE','91.4 / 92.6','90.8 / 92.0','94.4 / 94.7']], [105,127,127,126])
    p('Historical values are quoted from the base paper table and are not our measurements. Its original 85/5/10 split and ten initialization runs on a fixed split differ from our four-way split and five changing split/initialization runs. Its reported uncertainty uses a different convention. We neither claim exact reproduction nor subtract these values to assert an improvement.')

    heading('9. Conclusion, limitations and future work', newpage=True)
    p(f'The project delivers a complete, audited link-prediction experiment and local demonstration. Fusion has a positive mean AP difference relative to VGAE on {positives}/3 datasets ({gains} percentage points). The per-dataset rankings and paired controls qualify that finding: adding structural scores to VGAE is not proof that VGAE is the best representation, nor that every heuristic contributes independently. The practical contribution is an inspectable post-encoder scoring method evaluated with consistent graph exposure and preserved artifacts.')
    for text in [
        'Static evaluation: uniformly removed undirected citations do not model future citation arrival, realistic missingness or person-to-person interaction.',
        'Candidate sampling: balanced uniform negatives are easier and have a different prevalence than an all-pairs deployment. Logistic outputs are not population-calibrated probabilities.',
        'Transductive features: all node attributes remain visible. No new-node or temporal generalization is demonstrated.',
        'Fixed model budget: 200 epochs, a PyG-style sampled VGAE objective and no extensive search may disadvantage a baseline. A strong GAE or DeepWalk result is retained rather than hidden.',
        'Uncertainty: five runs and three related citation graphs support bounded descriptive conclusions. GPU operations and laptop workloads limit bitwise/timing reproducibility.',
        'Supervision and interpretation: fusion uses an additional labeled fitting partition. Coefficients describe a correlated feature model, not causal network mechanisms.']:
        p('• ' + text)
    p('Future extensions should begin with harder negative sampling, disjoint temporal data and a true social or coauthorship graph. A separately registered sensitivity study could examine stronger GAE/VGAE training and KL scaling without tuning the current test sets. Cross-fitted stacking can use labeled edges more efficiently, but it must preserve graph-exposure boundaries. Inductive pair-subgraph models are another comparison if the compute budget permits.')
    heading('10. Reproducibility and demonstration', 2)
    p('Start the local demo with scripts/start_demo.ps1 and open http://127.0.0.1:8768. It reads completed artifacts without retraining. The demo guide supplies a short walkthrough and answers for common viva questions. The team fields remain blank; edit docs/team.json and regenerate the report before submitting.')
    p('To rerun experiments, use scripts/run_all.ps1. It uses the dedicated conda interpreter and resumes completed work. A changed source/configuration requires a separate artifact folder, preserving prior results. Checkpoints retain model, optimizer and random state. Environment locks, provenance, per-run predictions and exported tables are included in the submission package. Raw datasets are public and can be downloaded again from the recorded source.')
    p('Audit gates: 12 protocol tests passed; all expected runs completed; metrics recomputed from predictions; positive/negative disjointness and graph hashes checked; saved scaler means compared with fusion-only features. The audit does not prove scientific superiority or validate unknown future data.')

    heading('11. References', newpage=True)
    p('[1] T. N. Kipf and M. Welling, “Variational Graph Auto-Encoders,” NIPS Workshop on Bayesian Deep Learning, 2016, arXiv:1611.07308. https://arxiv.org/abs/1611.07308', 'small')
    p('[2] B. Perozzi, R. Al-Rfou and S. Skiena, “DeepWalk: Online Learning of Social Representations,” Proc. ACM KDD, pp. 701-710, 2014, doi: 10.1145/2623330.2623732. https://arxiv.org/abs/1403.6652', 'small')
    for reference in references:
        p(reference, 'small')
    p('[13] PyTorch Geometric, “Planetoid dataset documentation” and “Autoencoder model source,” accessed 9 October 2026. https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html ; https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/autoencoder.html', 'small')
    p('[14] scikit-learn, “roc_auc_score” and “average_precision_score,” software documentation. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html', 'small')
    p('Experiment source SHA-256: ' + manifest['source_hash'], 'small')
    pdf = OUT / 'SNA_Project_Report.pdf'
    doc = SimpleDocTemplate(str(pdf), pagesize=(595.28,841.89), leftMargin=54, rightMargin=54, topMargin=48, bottomMargin=54,
                            title=title, author='SNA project team', pageCompression=1)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (ROOT / 'REPORT.md').write_text('\n'.join(markdown), encoding='utf-8')
    print('Created', pdf, 'and REPORT.md', flush=True)


if __name__ == '__main__':
    main()
