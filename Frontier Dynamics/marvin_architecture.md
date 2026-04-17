# MarvinBot: Architecture Overview

**System**: Autonomous knowledge-acquisition agent  
**Framework**: STLE v3 (Evidence-Scaled Posterior Networks)  
**Status**: Production — running continuously since February 2026

---

## System Overview

MarvinBot is an autonomous learning system that studies topics from the open web, builds a knowledge graph, and tracks its own understanding using STLE v3's accessibility score μ_x.

The system operates in a continuous loop: select a topic → fetch content → process through ML pipeline → update μ_x → select next topic. No human intervention is required. The μ_x score drives every decision.

---

## STLE v3 Pipeline

All knowledge assessment flows through the STLE v3 pipeline:

```
Topic (name / description)
    ↓
SentenceTransformer (frozen, all-MiniLM-L6-v2, 384-D)
    ↓
Trainable Projection (384 → 256 → 64-D, BatchNorm + ReLU)
    ↓
Per-Domain RealNVP Normalizing Flows (4 coupling layers, 64-D)
    ↓
Evidence-Scaled Posterior Networks
    α_c = β + λ · N_c · p(z | domain_c)
    α_0 = Σ_c α_c
    μ_x = (α_0 - K) / α_0
    ↓
{μ_x, μ_y, confidence_level, dominant_domain}
```

### Pipeline Components

**Embedding**: all-MiniLM-L6-v2 produces 384-D dense embeddings from topic names and descriptions. Frozen during all training — serves as a fixed feature extractor.

**Projection**: A two-layer MLP (384 → 256 → 64) with batch normalization. Trained for domain separation, achieving 88.4% classification accuracy. Reduces dimensionality to a regime where density estimation is tractable.

**Per-Domain Flows**: Each of the 4 trained domains (General, Chemistry, Computer Science, History) has a dedicated RealNVP normalizing flow in the 64-D latent space. Each flow has 4 coupling layers with hidden dimension 64.

**Evidence Scaling**: The evidence scale λ ≈ 0.001 prevents the saturation bug that caused μ_x → 1.0 at large N in the original formula. Auto-calibrated via grid search targeting median μ_x ≈ 0.9 on training data.

---

## Two-Stage Training

The STLE model is trained in two stages:

**Stage 1 — Projection Training** (60 epochs): Train the projection layer and a temporary classification head using cross-entropy loss. The classification head shapes the latent space for domain separation, then is discarded.

**Stage 2 — Flow Training** (40 epochs): Freeze the projection, pre-compute latent representations, and train per-domain flows with negative log-likelihood loss. Noise augmentation anneals from 0.5 to 0.1 for robustness.

**Post-Training**: Unfreeze the projection and calibrate λ on the full training set.

---

## Knowledge Classification

Marvin classifies every topic into one of three knowledge states based on μ_x:

```
Known    (μ_x ≥ 0.70)  →  Marvin has studied this topic sufficiently
Frontier (0.30 ≤ μ_x < 0.70)  →  Partial knowledge — priority for study
Unknown  (μ_x < 0.30)  →  Outside current knowledge boundaries
```

These thresholds drive the autonomous study strategy:
- **Frontier topics** are prioritized for study (maximum information gain)
- **Known topics** are revisited when recency decay lowers their score
- **Unknown topics** are discovered through bridge connections to known domains

---

## Autonomous Learning Cycle

Marvin operates on a continuous ~30-second study cycle:

```
1. Select topic (strategy-driven: frontier-first, or exploration)
    ↓
2. Fetch content (Wikipedia, arXiv, Internet Archive)
    ↓
3. Process through STLE v3 pipeline → compute μ_x
    ↓
4. Update knowledge graph (topic scores, domain classifications)
    ↓
5. Evaluate: continue studying this topic, or move to next?
    ↓
6. Periodically: recalculate all μ_x scores (recency decay)
    ↓
7. Periodically: sleep cycle (consolidation, strategy evaluation)
    ↓
→ Return to step 1
```

---

## Data Sources

Marvin currently studies from three primary sources:

| Source | Content Type |
|--------|-------------|
| Wikipedia | Article text, summaries, category structure |
| arXiv | Paper abstracts and metadata |
| Internet Archive | Archived documents and texts |

Content is fetched via API, processed into embeddings, and evaluated through the STLE pipeline. No content is stored permanently — only the resulting μ_x scores and knowledge graph metadata.

---

## Key Design Decisions

**No LLM layer**: All intelligence is algorithmic. Marvin's study decisions, knowledge assessments, and chat responses come from the STLE pipeline and knowledge graph — not from prompting a language model. LLM integration is planned as a future "mouth" layer on top of the STLE "brain."

**Evidence scaling (λ)**: Without λ, the original formula saturates to μ_x ≈ 1.0 at Marvin's scale (N > 8,000). The evidence scale keeps scores bounded and discriminative at any knowledge base size.

**Dirichlet prior (β = 1.0)**: Prevents zero-evidence collapse. Every domain starts with a baseline concentration of 1.0, ensuring α_0 never approaches zero.

**Per-domain flows**: Rather than a single density model for all topics, each domain has its own normalizing flow. This preserves domain structure and improves discrimination between topics that belong to different fields.

---

## Production Statistics

| Specification | Value |
|---------------|-------|
| Knowledge base size | 16,923+ topics |
| Study sessions completed | 3,200+ |
| Trained STLE domains | 4 |
| Total domains in database | 23 |
| Embedding dimension | 384-D (frozen) |
| Latent dimension | 64-D (learned) |
| Evidence scale (λ) | ~0.001 |
| Study interval | ~30 seconds |
| Held-out μ_x | 0.855 ± 0.062 |
| OOD μ_x | ~0.41 |
| Domain accuracy | 88.4% |

---

## What's Next

**PAC-Bayes Training (Stage 3)**: Joint optimization of projection and flows with a provable generalization bound via weight-space KL divergence. Technical specification complete, implementation planned.

**Domain Expansion**: Train flows for all 23 domains in the knowledge base, not just the current 4.

**LLM Integration**: Add a language model layer that consults μ_x before generating responses — STLE as epistemic grounding for natural language generation.

---

*For the theoretical framework powering this system, see the [STLE v3 specification](../stle/v3/STLE_v3.md).*
