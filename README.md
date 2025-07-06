# sbi_task5
Inference of protein secondary structure motifs

---

| Aspect               | Target                                                                                                                                                                                                        |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Simulator**        | Exact replication of the Task 5 HMM (fixed emissions + transitions; always start in “other”).                                                                                                                 |
| **Training data**    | 𝑁≈30 k synthetic sequences (lengths 50-300 aa, uniformly sampled). For each:  • observed amino-acid string  • ground-truth hidden state path  • exact posterior state probabilities from forward–backward.   |
| **Inference engine** | BayesFlow (summary net + conditioner flow) that maps a *variable-length* sequence to a distribution over a matching-length vector of helix–posterior logits.                                                  |
| **Metrics**          | • Average cross-entropy between predicted and exact posteriors on held-out synthetic set  • Calibration curve  • Helix/other classification accuracy (0.5 threshold)  • Qualitative overlay on human insulin. |
| **Deliverables**     | • Well-documented Python package  • Jupyter notebook demo  • Slides & short report (per course rules)                                                                                                         |
