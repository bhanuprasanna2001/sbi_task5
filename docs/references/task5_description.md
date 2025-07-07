## Task 5 — Inference of protein secondary structure motifs

**Lecturer:** Daniel (<daniel.habermann@tu-dortmund.de>)

Proteins are long chains of amino acids that fold into specific shapes. One key level of organization is the *secondary structure*, where each amino acid is part of three local folding patterns (alpha‑helix, beta‑sheet or random coil [1]), which then further fold into three‑dimensional structures, defining the function of the protein.  
In this project, we focus specifically on predicting **alpha‑helix** patterns using a two‑state Hidden Markov Model (HMM) [2].  
The two states are **“alpha‑helix”** and **“other”** (encompassing beta‑sheets and coils).  
We assume *fixed emission and transition probabilities* derived from empirical data [3].

We define the following generative model for simulating amino‑acid sequences:

* The sequence always starts in the **“other”** state.  
* At each step we emit an amino acid according to the following **emission probability tables**.

### Emission probabilities

**Alpha‑helix**

| Amino acid | A | R | N | D | C | E | Q | G | H | I |
|------------|---|---|---|---|---|---|---|---|---|---|
| Probability| 12 % | 6 % | 3 % | 5 % | 1 % | 9 % | 5 % | 4 % | 2 % | 7 % |

| Amino acid | L | K | M | F | P | S | T | W | Y | V |
|------------|---|---|---|---|---|---|---|---|---|---|
| Probability| 12 % | 6 % | 3 % | 4 % | 2 % | 5 % | 4 % | 1 % | 3 % | 6 % |

**Other**

| Amino acid | A | R | N | D | C | E | Q | G | H | I |
|------------|---|---|---|---|---|---|---|---|---|---|
| Probability| 6 % | 5 % | 5 % | 6 % | 2 % | 5 % | 3 % | 9 % | 3 % | 5 % |

| Amino acid | L | K | M | F | P | S | T | W | Y | V |
|------------|---|---|---|---|---|---|---|---|---|---|
| Probability| 8 % | 6 % | 2 % | 4 % | 6 % | 7 % | 6 % | 1 % | 4 % | 7 % |

### Transition probabilities

* If the current state is **“alpha‑helix”**, the next state is  
  * “alpha‑helix” with probability **p = 90 %**  
  * “other” with probability **1 − p = 10 %**  

* If the current state is **“other”**, the next state is  
  * “alpha‑helix” with probability **p = 0.05**  
  * “other” with probability **1 − p = 95 %**

Using this simulator we can generate amino‑acid chains of arbitrary length.  
Additionally, the **Viterbi algorithm** [4] (e.g. via the *hmmlearn* Python package [5]) can infer state probabilities for a given amino‑acid sequence.  
Given pairs of sequences and state probabilities as training data, the goal is to train a **BayesFlow neural posterior density estimator**.  
Finally, compare the posterior state‑probability estimates for a *new* protein sequence to the known ground truth, e.g. the annotated secondary structure of **human insulin** [6].

---

### References

1. <https://old-ib.bioninja.com.au/higher-level/topic-7-nucleic-acids/73-translation/protein-structure.html>  
2. <https://scholar.harvard.edu/files/adegirmenci/files/hmm_adegirmenci_2014.pdf>  
3. <https://www.kaggle.com/datasets/alfrandom/protein-secondary-structure>  
4. <https://web.stanford.edu/~jurafsky/slp3/A.pdf>  
5. <https://pypi.org/project/hmmlearn/>  
6. <https://www.rcsb.org/3d-sequence/1A7F>
