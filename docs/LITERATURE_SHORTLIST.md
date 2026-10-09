# Journal Literature for Heuristic Fusion and Graph Link Prediction

**Planning shortlist only.** The final reviewed roster is in `literature_notes.json` and the generated report. It uses ten full-text journal articles, replacing inaccessible candidates and the survey entries with empirical studies. The preliminary list below is retained as planning context.

The course wording is 2022 onwards, so 2022 is included. The ten entries below have identifiable journal publications within that range. They form a reading shortlist, not a completed literature review: exact dataset versions, performance metrics, numerical results, and limitations must be extracted from each full text before the final report. Publisher or author-hosted records were checked on 9 October 2026.

## Ten journal papers

| No. | Publication | Year and venue | Method and relevance |
| --- | --- | --- | --- |
| 1 | Salha-Galvan et al., [Modularity-aware graph autoencoders for joint community detection and link prediction](https://doi.org/10.1016/j.neunet.2022.06.021) | 2022, Neural Networks 153, 474–495 | Modularity-informed message passing and regularization extend GAE/VGAE. Useful example of a focused base-model extension. [Author version](https://arxiv.org/abs/2202.00961). |
| 2 | Wu et al., [Link Prediction on Complex Networks: An Experimental Survey](https://link.springer.com/article/10.1007/s41019-022-00188-2) | 2022, Data Science and Engineering 7, 253–278 | Compares method families across 36 networks. Supports network-dependent evaluation and a strong heuristic baseline roster. |
| 3 | Fang et al., [Evidential link prediction by exploiting the applicability of similarity indexes to nodes](https://doi.org/10.1016/j.eswa.2022.118397) | 2022, Expert Systems with Applications 210, 118397 | Combines similarity indices through evidence theory with node-dependent applicability. Close related work for fusion and a contrast to fixed logistic coefficients. |
| 4 | Liu et al., [Link prediction approach combined graph neural network with capsule network](https://doi.org/10.1016/j.eswa.2022.118737) | 2023, Expert Systems with Applications 212, 118737 | GNN node embeddings become pair feature maps for capsule classification. Discusses richer pair representations; too complex for the core implementation. |
| 5 | [“Stealing fire or stacking knowledge” by machine intelligence to model link prediction in complex networks](https://doi.org/10.1016/j.isci.2022.105697) | 2023, iScience 26(1), 105697 | Tests assumptions behind stacking link predictors. Directly relevant to the possibility that fusion offers no consistent benefit. |
| 6 | Arrar, Kamel and Lakhfif, [A comprehensive survey of link prediction methods](https://link.springer.com/article/10.1007/s11227-023-05591-8) | Online 2023; issue 2024, The Journal of Supercomputing 80, 3902–3942 | Organizes similarity, embedding, and learning methods. Literature taxonomy; this review does not report a new benchmark experiment. |
| 7 | Lu and Uddin, [A parameterised model for link prediction using node centrality and similarity measure based on graph embedding](https://doi.org/10.1016/j.neucom.2024.127820) | 2024, Neurocomputing 593, 127820 | NCSM combines centrality and similarity as edge features in a custom graph neural network. Important prior art constraining our novelty claim. |
| 8 | Si et al., [Inductive Subgraph Embedding for Link Prediction](https://link.springer.com/article/10.1007/s11036-024-02339-3) | Online 2024; issue 2025, Mobile Networks and Applications 30, 312–323 | SE4LP learns from local subgraphs with contrastive supervision. Uses citation benchmarks, but its Facebook page graph and evaluation protocol differ from the optional ego-style graph. |
| 9 | [TAGNN: topology-aware graph neural network framework for link prediction](https://www.nature.com/articles/s41598-026-48184-0) | 2026, Scientific Reports 16, 17292 | Structural information supplements GNN-based link representations. Relevant recent comparison of pair topology with node-centered encoding. |
| 10 | Baswani et al., [Link Prediction Using Node Influence Based on Graph Neural Networks](https://doi.org/10.1109/ACCESS.2026.3677526) | 2026, IEEE Access 14, 51732–51749 | Centrality and similarity features, a centrality-aware encoder, and multiple decoding branches. Further evidence that this feature combination is established. [Published author-hosted copy](https://shura.shu.ac.uk/37825/). |

A DOI containing 2022 does not by itself make an article's issue year 2022. Keep online and issue dates separately where they differ. Copy the full author list and bibliographic fields from the publication rather than guessing them from this shortened table.

## Reading and evidence matrix

For every entry, fill these fields in a working matrix:

1. Verified title, authors, online year, issue year, venue, DOI, publication type, and accessible full-text URL.
2. Problem setting: undirected/directed, static/temporal, transductive/inductive, attributed/featureless.
3. Methodology: architecture or algorithm, inputs, learning objective, and source section.
4. Dataset names, exact versions, node/edge counts, and source table.
5. Positive splits, negative sampling, graph visible at evaluation, and tuning policy.
6. Evaluation metrics with definitions and the relevant results table.
7. Advantages supported by evidence; distinguish the authors' claims from demonstrated comparisons.
8. Limitations acknowledged by the authors and separately labeled limitations identified by our team.
9. Relationship to our project and why its result is or is not directly comparable.

For a narrative survey, record original experimental datasets and performance as not applicable when none are provided. Discuss reviewed benchmarks and metrics as reviewed evidence. Do not manufacture a performance number just to fill a column.

Two entries are journal surveys. The literal rubric permits journal papers, but if the professor means ten original empirical research articles, replace those entries with additional primary studies. One relevant candidate is [Incorporating high-frequency information into edge convolution for link prediction in complex networks](https://www.nature.com/articles/s41598-024-56144-9), Scientific Reports, 2024; complete its metadata and full-text verification before counting it. A second replacement must also be verified. This shortlist should not be called complete under a ten-empirical-articles interpretation.

## Other references that do not count toward the ten journals

| Reference | Role |
| --- | --- |
| [Kipf and Welling, Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308), 2016 workshop paper | Base paper; architecture and historical results |
| [Perozzi, Al-Rfou and Skiena, DeepWalk](https://arxiv.org/abs/1403.6652), KDD 2014 | Second distinct implemented existing work |
| [Heuristic Learning with Graph Neural Networks: A Unified Framework for Link Prediction](https://arxiv.org/abs/2406.07979), 2024 | Additional related work on learning heuristic information; does not satisfy journal count |
| [PyG Planetoid documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html) | Benchmark identities and expected sizes |
| [PyG RandomLinkSplit source](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/random_link_split.html) | Verify exactly which graph reaches the encoder at evaluation |

Prioritize reading papers 1, 3, 5, and 7 for the motivation and novelty boundaries, then finish all ten before the report is finalized. Do not treat conference papers, preprints, a source list, or abstracts alone as ten completed journal reviews.
