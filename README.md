# sbi_task5
Inference of protein secondary structure motifs

---

| Aspect                      | Purpose                                                                                                                                                            |            |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| **Goal**                    | Estimate, for every residue in an amino-acid sequence, the posterior probability of being in the **α-helix (H) state** given the observed sequence.                |            |
| **Generative model**        | A 2-state HMM (`H` = helix, `C` = coil) with categorical emissions over the 20-letter protein alphabet.                                                            |            |
| **Inference target**        | A length-`L` vector **θ = (p₁,…,p\_L)** where pᵢ = P(stateᵢ = H                                                                                                    | sequence). |
| **Technique**               | Simulation-based inference (SBI) via **BayesFlow’s ContinuousApproximator** → learns a mapping *once* that amortises posterior estimation for arbitrary sequences. |            |
| **Baseline / sanity-check** | Classic forward-backward posterior from **`hmmlearn.CategoricalHMM`**. ([hmmlearn.readthedocs.io][1], [hmmlearn.readthedocs.io][1])                                |            |
