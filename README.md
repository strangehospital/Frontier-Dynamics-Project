[README.md](https://github.com/user-attachments/files/25150380/README.md)
# Set Theoretic Learning Environment: STLE.v3

> **Enabling AI systems to reason explicitly about the boundaries of it's knowledge through formal dual-space representation**

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Status: | Open-source framework | Research-driven development | Production deployment: MarvinBot|

## What is STLE?

**Set Theoretic Learning Environment (STLE)**: Framework that enables AI systems to engage in principled reasoning about “unknown” information through a dual-space representation. To accomplish this, STLE models accessible (known) and inaccessible (unknown) data as complementary fuzzy subsets of a unified domain, with a membership function μ_x: D → [0,1] that quantifies the degree to which any data point belongs to the system's knowledge.

Fundamental Problem: Contemporary AI systems cannot explicitly represent the boundaries of it's knowledge. Neural networks will confidently classify everything, even data it's never encountered, leading to potentially false outcomes or hallucinations:

- **Show a model random noise? "Cat (92% confidence)"**
- **Feed it corrupted data? "High priority threat (87%)"**

Current AI can't say "I don't know." This makes it dangerous in production.

STLE fixes this by modeling both accessible (μ_x) and inaccessible (μ_y) data as complementary fuzzy sets over a universal domain D, with the guarantee that μ_x(r) + μ_y(r) = 1 for all data points r.

- **Known data: μ_x ≈ 0.85 → "I know this well"**
- **Novel data: μ_x ≈ 0.41 → "This is unfamiliar territory"**
- **Learning frontier: 0.3 < μ_x < 0.7 → "I'm partially uncertain — worth studying"**

## Update: STLE.v3 | Offical Paper Release | MarvinBot LIVE

- **Official Paper Release (April 10th 2026)**: _Set Theoretic Learning Environment for Large-Scale Continual Learning: Evidence Scaling in High-Dimensional Knowledge Bases_

- **GitHub Repository update (April 10th 2026)**: STLE.v3 implementation and documentation uploaded

- **MarvinBot**: Autonomous Learning Agent built on STLE.v3 is LIVE! Watch/Chat @ https://just-inquire.replit.app

## Theoretical Foundations

STLE is grounded in four formal axioms:
- **[A1] Coverage**: x ∪ y = D (every data point is accounted for)
- **[A2] Non-Empty Overlap**: x ∩ y ≠ ∅ (partial knowledge states exist)
- **[A3] Complementarity**: μ_x(r) + μ_y(r) = 1, ∀r ∈ D (knowledge and ignorance are complementary)
- **[A4] Continuity**: μ_x is continuous in the data space (smooth generalization)
  
These axioms establish STLE as a rigorous mathematical framework rather than an empirical heuristic. STLE.v3 formulation preserves all axioms while adding practical capabilities:
- Saturation resistance via evidence scaling
- Multi-domain support with per-domain normalizing flows
- Numerical stability through log-space computation
- Convergence guarantees via PAC-Bayes theory
  
### Key Properties:
- **Complementarity**: μ_x(r) + μ_y(r) = 1 (mathematically guaranteed)
- **Learning frontier**: The region 0.3 < μ_x < 0.7 identifies optimal targets for active learning
- **Out-of-distribution detection**: Low μ_x scores on unfamiliar data without requiring OOD training samples
- **Epistemic uncertainty quantification**: Explicit measure of model uncertainty distinct from data noise

STLE is currently deployed in MarvinBot, an autonomous continual learning system maintaining a knowledge base of 16,923 topics with over 3,200 completed study sessions.

Formal proofs are provided in STLE_v3.md and the offical paper; _Set Theoretic Learning Environment for Large-Scale Continual Learning: Evidence Scaling in High-Dimensional Knowledge Bases._

## Quick Start
Installation and Demo:
``` Python
git clone https://github.com/strangehospital/Frontier-Dynamics-Project
cd Frontier-Dynamics-Project
stle_v3_minimal_demo.py
```
Output: Five validation experiments demonstrating complementarity, OOD detection, frontier identification, Bayesian updates, and saturation resistance. 

- **In Depth**: Running stle_v3_experiments runs 6 experiments, including the same core 5 (from minimal demo) plus a more rigorous saturation resistance test at larger N values (up to 5,000). Also includes a convergence analysis experiment (Experiment 4) that trains at varying dataset sizes and tracks how epistemic uncertainty decreases. 

## Quick Start API
``` Python
from stle_v3_minimal_demo import MinimalSTLEv3

model = MinimalSTLEv3(input_dim=2, num_classes=2)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Predictions: {predictions['predictions']}")
print(f"Accessibility (μ_x): {predictions['mu_x']}")
print(f"Epistemic uncertainty: {predictions['epistemic']}")
```
Interpretation:

- μ_x ≈ 0.9: High accessibility (familiar, well-studied data)
- μ_x ≈ 0.5: Learning frontier (partial knowledge, active learning candidate)
- μ_x ≈ 0.2: Low accessibility (unfamiliar, out-of-distribution)

## Why STLE Matters:
Comparison with State-of-the-Art Methods

| Capability | **STLE** | Softmax | MC Dropout | Ensembles | Posterior Nets |
|-----------|:--------:|:-------:|:----------:|:---------:|:--------------:|
| **Epistemic Uncertainty** | ✅✅ | ❌ | ✅ | ✅ | ✅✅ |
| **Explicit Ignorance Modeling** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **OOD Detection (no OOD training)** | ✅ | ❌ | ⚠️ | ⚠️ | ⚠️ |
| **Complementarity Guarantee (μ_x + μ_y = 1)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Learning Frontier Identification** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Computational Cost** | 🟢 Low | 🟢 Low | 🟡 Medium | 🔴 High | 🟡 Medium |

## Validation Metrics
### Proof-of-Concept (STLE v2) Implementation 
MNIST-based validation:

- **OOD Detection AUROC**: 0.668 (without OOD training data)
- **Classification Accuracy**: 81.5% on test set
- **Complementarity Error**: 0.00 (machine precision)
- **Training Time**: < 1 second (400 samples)
- **Inference Speed**: < 1 ms per sample

### Production Deployment (MarvinBot: Autonomous Learning Agent)
16,923-topic knowledge base:

- **Held-out μ_x**: 0.855 ± 0.062 (known topics)
- **Novel topic μ_x**: 0.41 (out-of-distribution topics)
- **Domain Classification Accuracy**: 88.4% (four domains)
- **Knowledge Base Scale**: 16,923 topics across 23 domains
- **Continual Learning**: 3,200+ study sessions without catastrophic forgetting

## Real-World Applications
1. Safety-Critical Decision Systems
``` Python
Safety-Critical Decision Systems
diagnosis = model.predict(patient_scan)
if diagnosis['mu_x'] < 0.5:
defer_to_human_expert()
```
Use case: Medical diagnosis systems that explicitly identify unfamiliar cases requiring human review.

2. Autonomous Systems
``` Python
frontier_samples = X[(mu_x > 0.4) & (mu_x < 0.6)]
request_labels(frontier_samples)
```
Use case: Efficient data labeling by querying samples at the learning frontier (30% sample efficiency improvement over random sampling).

3. Self-Driving Vehicles
``` Python
if perception['mu_x'] < 0.6:
    engage_safe_mode()  # Don't act on unfamiliar scenarios
```
Use case: Ensures safety via explicit uncertainty in autonomous vehicles. 

## STLE.v3
### Core Innovation
Computes accessibility on-demand via an evidence-scaled multi-domain Dirichlet formulation, a strict generalization of STLE.v2, which preserves density-based lazy initialization while preventing saturation at scale:

- α_c = β + λ · N_c · p(z | domain_c)
- α_0 = Σ_c α_c
- μ_x = (α_0 - K) / α_0

### Parameters:
- β: Dirichlet prior (default 1.0, prevents zero-evidence collapse)
- λ: Evidence scale (typically 0.001, prevents saturation at large N)
- N_c: Training samples in domain c
- p(z|domain_c): Density under domain c's normalizing flow
- K: Number of domains

Key insight: Evidence scaling (λ) bounds μ_x independently of training set size, enabling discrimination between well-known and barely-known topics even at production scale.

### STLE v3 Theoretical Guarantees

- **Complementarity**: μ_x(r) + μ_y(r) = 1 for all r ∈ D (by construction)
- **PAC-Bayes Convergence**:|μ_x(r) - μ*_x(r)| ≤ O(1/√(λN))  with probability 1-δ
- **Monotonic Learning**: ∂μ_x/∂N_c ≥ 0 (more data increases accessibility)
- **Saturation Prevention**: μ_x < 1 - K/(Kβ + λ·N_total·max_c p(z|c)) for all finite N

## MarvinBot: STLE v3 in Practice

MarvinBot is an autonomous continual learning system that demonstrates STLE v3 at production scale. Unlike traditional chatbots or static models, MarvinBot operates continuously without human intervention:

- **Autonomous study cycle**: Independently selects topics to study based on μ_x scores
- **Knowledge integration**: Fetches content from Wikipedia, processes it through STLE's learning pipeline, and updates its knowledge representation
- **Meta-reasoning**: Uses accessibility scores to identify knowledge gaps and prioritize learning
- **Continual learning**: Grows its knowledge base without catastrophic forgetting

### Key Results
- **Knowledge base growth**: 16,923 topics studied over 3,200+ sessions
- **Accessibility calibration**: Mean μ_x = 0.855 on held-out known topics
- **OOD discrimination**: Mean μ_x = 0.41 on novel topics (appropriate uncertainty)
- **Domain separation**: 88.4% classification accuracy across trained domains
- **Saturation resistance**: Evidence scaling prevents μ_x collapse at scale

### Architecture Highlights
MarvinBot implements STLE v3, which extends the original framework with:

- Evidence-scaled Posterior Networks: Prevents saturation bug that caused μ_x → 1.0 at large N
- Multi-domain structure: Per-domain normalizing flows for specialized knowledge modeling
- Trainable projection: 384-D embeddings → 64-D latent space to overcome curse of dimensionality
- Numerical stability: Log-space computations with logsumexp for stable density evaluation

More on MarvinBot's architecture is detailed in the offical Set Theoretic Learning Environment Paper included in this repository.

## Repository Content
STLE v3 Core Implementation
- stle_v3_minimal_demo.py (23 KB) - NumPy implementation with zero dependencies
- stle_v3_core.py (22 KB) - Full PyTorch version with normalizing flows
- stle_v3_experiments (16 KB) - Automated validation suite (6 experiments)
- stle_v3_visualizations (11 KB) - Visualization generator

STLE v3 Documentation
- STLE_v3.md (27 KB) - Theoretical specification and proofs
- STLE_v3_Technical_Report.md (15 KB) - Validation results and analysis
- Set Theoretic Learning Environment Paper.md (43 KB) - Official STLE research paper (Includes v3 and MarvinBot Deployment)

STLE v2 Core Implementation
- stle_minimal_demo.py (17 KB) — NumPy implementation with zero dependencies
- stle_core.py (18 KB) — Full PyTorch version with normalizing flows
- stle_experiments.py (16 KB) — 5 experiments
- stle_visualizations.py (11 KB) — Publication-quality visualization generator

STLE v2 Documentation
- STLE_v2.md (48 KB) — Theoretical specification and proofs
- STLE_Technical_Report.md (18 KB) — Validation results and analysis
- Research.md (28 KB) — Development journey and breakthrough solutions

STLE v2 Visualizations
- stle_decision_boundary.png — Classification boundaries with accessibility overlay
- stle_ood_comparison.png — In-distribution vs. out-of-distribution detection
- stle_uncertainty_decomposition.png — Epistemic vs. aleatoric uncertainty
- stle_complementarity.png — Verification of μ_x + μ_y = 1

## Citation
If you use STLE in your research, please cite:
```bibtex
@article{musila2026stle,
title={Set Theoretic Learning Environment for Large-Scale Continual Learning:
Evidence Scaling in High-Dimensional Knowledge Bases},
author={Musila, Moses (u/Strange_Hospital7878)},
journal={arXiv preprint},
year={2026},
note={Version 3.0 - Production Deployment}
}
```

## Contact

- **GitHub**: https://github.com/strangehospital
- **Email**: mwmusila@outlook.com
- **Reddit**: u/Strange_Hospital7878 or u/CodenameZeroStroke

## License
MIT License

## Acknowledgments
Development tools: Claude Sonnet 4.5 (Anthropic), DeepSeek R1, Genspark AI

## Inspiration
A philosophical thought experiment on how a limited subjective experience functions in a unverifiable objective reality laid the foundations for the principle that: AI systems must understand the boundaries of their knowledge before they can reliably extend it.

---

<p align="center">
  <strong>"The boundary between knowledge and ignorance is no longer philosophical. It is μ_x = 0.5"</strong>
</p>

<p align="center">
  <a href="https://github.com/strangehospital/Frontier-Dynamics-Project/stargazers">⭐ Star this repo</a> • 
  <a href="https://github.com/strangehospital/Frontier-Dynamics-Project/issues">🐛 Report Issues</a>
</p>

---

Project Status: Production deployment (MarvinBot) | STLE. v3

Last Updated: April 10 2026
