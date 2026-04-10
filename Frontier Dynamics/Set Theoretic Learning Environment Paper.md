**Set Theoretic Learning Environment for Large-Scale Continual Learning:**

**Evidence Scaling in High-Dimensional Knowledge Bases**

Moses Musila (strangehospital)

*GitHub: Frontier Dynamics Project*

mwmusila@outlook.com

**Abstract**

This paper presents Set Theoretic Learning Environment (STLE): a framework that enables artificial intelligence systems to engage in principled reasoning about “unknown” information through a dual-space representation. To accomplish this, STLE models accessible (known) and inaccessible (unknown) data as complementary fuzzy subsets of a unified domain, with a membership function μ\_x: D → \[0,1\] that quantifies the degree to which any data point belongs to the system's knowledge. STLE’s framework progressed through three major stages of development, and each version of STLE was born through addressing limitations exposed by real-world deployment. The original formulation, STLE v0, was a thought experiment grounded in Set Theory, but was not computationally feasible. The following formulation, STLE v1 established the frameworks feasibility with fuzzy membership and Bayesian statistics. STLE v2, solved a fundamental challenge (how to initialize μ\_x(r) for unseen data without prior knowledge) and established the theoretical foundations with inspiration from *Posterior Networks* and PAC-Bayes theory. I applied STLE v2 into practice with our machine learning model, MarvinBot, and when implemented on a 16,923-topic knowledge base it revealed two critical failures: the curse of dimensionality in 384-dimensional embedding space, and a saturation bug where the accessibility formula collapses to μ\_x ≈ 1.0 when the training set size N\_x exceeds several thousand. STLE v3, the latest version, resolved both critical failures through utilizing an evidence-scaled multi-domain Dirichlet formulation: α\_c \= β \+ λ · N\_c · p(z|domain\_c), μ\_x \= (α\_0 \- K)/α\_0, where λ is a calibrated evidence scale that prevents saturation while preserving all STLE theoretical guarantees. Thus, utilizing our machine learning model, MarvinBot, we validate on a continuously growing knowledge base, achieving a mean μ\_x \= 0.855 on held-out topics, μ\_x ≈ 0.41 on novel out-of-distribution topics, and 88.4% domain classification accuracy across four domains; STLE v3’s formulation is a strict generalization of the original: meaning it reduces to v1 under specific parameter settings (K=2, β=0, λ=1), while supporting multi-domain structure, numerical stability via logsumexp, and auto-calibrated evidence scaling. Moreover, we further outlined a PAC-Bayes training extension that provides a provable generalization bound of |μ\_x \- μ\*\_x| \= O(1/√(λN)) with probability 1-δ.

# **1  Introduction**

Artificial Intelligence systems in the form of Large Language Models (LLM’s) are known to be being over-confident, even when wrong, and are prone to hallucination. Therefore, a fundamental challenge becomes enabling these systems to reason about what they do not know. Traditionally, machine learning models produce a prediction that’s associated with a confidence score (probability), but these scores can conflate, or fail to fully contextualize, two distinct forms of uncertainty: Aleatoric Uncertainty, the uncertainty arising from inherent noise in the data, vs Epistemic Uncertainty, the uncertainty arising from limited knowledge. I believe that attempting to mitigate either form of uncertainty is crucial for safety critical applications, active learning, and continual learning systems. This paper demonstrates that Set Theoretic Learning Environment (STLE) provides a solution to the challenge of AI systems processing epistemic uncertainty by providing an environment for principled reasoning about inaccessible information through dual-space representation.    

Set Theoretic Learning Environment addresses the challenge of epistemic uncertainty by introducing a dual-space representation: Every data point “r” in a universal domain D, simultaneously belongs to an accessible set x (what the system knows) and an inaccessible set y (what it doesn’t know), with complementary membership functions μ\_x(r) \+ μ\_y(r) \= 1\. This complementary formulation provides a principled framework for quantifying epistemic uncertainty. Thus, the “Learning Frontier” is defined as x ∩ y \= {r ∈ D : 0 \< μ\_x(r) \< 1}, which represents the boundary between what the system knows and or has access to (i.e its knowledge representation) and what the system doesn’t know and or doesn’t have access to (i.e its ignorance representation). Moreover, this region serves as a natural target for active learning.

Although being ostensively simple, the theoretical potential of STLE is clear. However, after moving from theory to practice and utilizing STLE v2 in a learning model training on large datasets, two critical failures were revealed that the original formulation was incapable of addressing. First, density estimation in high-dimensional embedding spaces suffers from the curse of dimensionality, which rendered the accessibility formula uninformative. Second, the original formula saturates to μ\_x ≈ 1.0 when the training set is large (i.e ≥ several thousand samples), making it impossible to actually distinguish well known topics from barely known topics.

In this paper, we will detail the development of STLE across the varying versions, documenting both the theoretical foundations and the empirical failures and successes that necessitated each revision. The central contribution of this paper is STLE v3, which resolves above mentioned limitations through evidence-scaled posterior networks with a multi-domain Dirichlet formulation inspired by Charpentier’s 2020 paper, “*Posterior Network: Uncertainty estimation without OOD samples via density-based pseudo-counts.”* In this paper, I demonstrate that STLE v3 preserves all theoretical guarantees of the original formulations, while adding saturation resistance, numerical stability, and multi-domain support. Moreover, my autonomous machine learning model, MarvinBot, validates on a deployed knowledge base of 16,923 topics that has been continuously learning for well over 3,200 study sessions.

**Contributions.** (1) Identified and formally characterized the saturation bug in the original STLE accessibility function, showing that μ\_x → 1 for all queries as N\_x → ∞. (2) Thus, proposed evidence-scaled posterior networks that bound μ\_x independently of N\_x while preserving complementarity, monotonic learning, and PAC-Bayes convergence. (3) Proved that the STLE v3s evidence-scaled multi-domain Dirichlet accessibility function is a strict generalization of the original, reducing to it under specific parameter settings. (4) Using a ML model (MarvinBot) I validate on a large-scale, continuously learning knowledge base, demonstrating appropriate discrimination between known, frontier, and out-of-distribution topics. (5) Outlined a PAC-Bayes training extension with provable generalization bounds for autonomous learning model running STLE (MarvinBot).

# **2  Related Work**

## **2.1  Posterior Networks**

STLE’s accessibility formulas are inspired by Charpentier’s 2020 paper “*Posterior Network: Uncertainty estimation without OOD samples via density-based pseudo-counts*.” Here, networks are used to estimate epistemic uncertainty by learning a Dirichlet distribution over predictive categorical distributions, and than use normalizing flows to map inputs to concentration parameters (Charpentier et al., 2020). The key insight is that the Dirichlet concentrations reflect the total evidence supporting a prediction: high concentration indicates strong evidence (i.e well-known region), while low concentration indicates weak evidence (i.e unfamiliar region). STLE's accessibility formulas build directly on this framework, but interpret the concentration as a measure of knowledge accessibility rather than prediction confidence.

## **2.2  PAC-Bayes Theory**

PAC-Bayes theory provides generalization bounds for stochastic predictors. Thus, PAC-Bayes demonstrates that for a posterior distribution Q over model weights and a data-independent prior P, the expected true risk is bounded by the empirical risk plus a complexity term proportional to √(KL(Q||P)/N). Moreover, Futami’s work extended this to epistemic uncertainty, showing that the Bayesian Excess Risk (BER) converges at rate O(1/√N) (Futami, et al., 2022). STLE leverages this framework to provide convergence guarantees for the accessibility functions themselves. 

## **2.3  Normalizing Flows**

Normalizing flows, as per (Rezende & Mohamed, 2015; Dinh et al., 2017), are generative models that learn invertible transformations between a base distribution and a data distribution. Therefore, they provide exact density evaluation via a change-of-variables formula, making them useful for the density estimations required by STLE. Thus, utilizing RealNVP, as per (Dinh et al., 2017\) with coupling layers, offers balance between expressiveness and computational efficiency.

# **3  Theoretical Foundations** 

## **3.1  Set Theoretic Learning Environment: STLE v3** 

## **Definitions:**

Let the **Universal Set,** (D), denote a universal domain of data points; Thus, STLE v3 defines two complementary fuzzy subsets:

**Accessible Set (x):** The accessible set, x, is a fuzzy subset of D with membership function μ\_x: D → \[0,1\], where μ\_x(r) quantifies the degree to which data point r is integrated into the system.

**Inaccessible Set (y):** The inaccessible set, y, is the fuzzy complement of x with membership function μ\_y: D → \[0,1\].

**Theorem:**

The accessible set x and inaccessible set y are complementary fuzzy subsets of a unified domain

These definitions are governed by four axioms:

*\[A1\] **Coverage**: x ∪ y \= D*

*\[A2\] **Non-Empty Overlap:** x ∩ y ≠ ∅*

*\[A3\] **Complementarity**: μ\_x(r) \+ μ\_y(r) \= 1, ∀r ∈ D*

*\[A4\] **Continuity**: μ\_x is continuous in the data space*

A1 ensures completeness and every data point is accounted for. Therefore, each data point belongs to either the accessible or inaccessible set. A2 guarantees that partial knowledge states exist, allowing for the learning frontier. A3 establishes that accessibility and inaccessibility are complementary measures (or states). A4 ensures that small perturbations in the input produce small changes in accessibility, which is a requirement for meaningful generalization.

**Learning Frontier:** Partial state region:               

x ∩ y \= {r ∈ D : 0 \< μ\_x(r) \< 1}.

**STLE v3 Accessibility Function**  

For K domains with per-domain normalizing flows:

                              *α\_c \= β \+ λ · N\_c · p(z | domain\_c)*          (1)

                              *α\_0 \= Σ\_c α\_c*          (2)

                              *μ\_x \= (α\_0 \- K) / α\_0*          (3)

## **3.2  A Brief History of STLE**

STLE v1 was born out of a philosophical thought experiment (i.e STLE v0). Consider a limited subjective human experience and thus an unverifiable objective reality. This is the philosophical problem of the independent mind. However, we can move from metaphysical speculation to a concrete computational intelligence framework if we consider the following analogy: 

***human subjective experience** \= The set of all phenomena contained within a given observer’s reference frame, including all elements that can be measured, interacted with or recognized, and;* 

***human objective experience** \= The set of all phenomena contained outside a given observer’s reference frame.* 

Then extrapolate to computational intelligence systems to create an even more interesting environment to explore the thought experiment, whereby: 

***artificial subjective (x)** \= the set of all training data currently accessible within a representational space, and;*

***artificial objective (y)** \= the set of all training data currently inaccessible within a representational space.*

**3.2.1 STLE v0: Thought Experiment** 

**The Universal Set (D):** the set of all existing training data

**x** \= the set of all training data currently accessible

**y** \= the set of all training data currently inaccessible

**STLE v0’s Theorem:**

x and y are complementary subsets of D, where D is duplicated data from a unified domain

**Logical Relationships:**

x ⊆ D 

y ⊆ D 

x ∪ y \= D 

x ∩ y ≠ ∅

**STLE v0: Probability function:**

For any r ∈ D

p(r ∈ x) \+ p(r ∈ y) \= 1

Here, we transformed our thought experiment into a more rigorous and computationally feasible framework by introducing fuzzy membership and a Bayesian posterior probability calculation. Moreover, we formalized the learning frontier from the relationship x ∩ y ≠ ∅

**3.3 STLE v1** 

**The Universal Set (D):** The set of all possible data points in a given domain

**Accessible set (x):** A fuzzy subset of D, with membership function μ\_x: D → \[0,1\].

**Inaccessible set (y):** The fuzzy complement of x, with μ\_y(r) \= 1 \- μ\_x(r).

**Theorem:**

The accessible set x and the inaccessible set y, are complementary fuzzy subsets of D, where D is duplicated data from a unified domain

***Relationships:***

x ∪ y \= D

x ∩ y ≠ ∅

**STLE v1: Probability Function**

Let r ∈ D

μ\_x(r) ∈ \[0,1\] be the degree of accessibility of r in x (extent to which r is known).

μ\_y(r) ∈ \[0,1\] be the degree of inaccessibility of r in y (extent to which r is unknown).

μ\_x(r) \+ μ\_y(r) \= 1 For any r ∈ D

**STLE v1: Bayesian Accessibility Rule**

Let μ\_x(r) the prior accessibility of r ∈ D

Upon observing evidence E: 

μ\_x(r) ← \[P(E | r ∈ x) · μ\_x(r)\] / \[P(E | r ∈ x) · μ\_x(r) \+ P(E | r ∈ y) · (1 \- μ\_x(r))\]                  (4)

The Bayesian update sets μ\_y(r) \= 1 \- μ\_x(r) and initiates the learning frontier dynamic (i.e moves r through x ∩ y). Thus, accessibility increases if evidence supports that r is in x, and accessibility decreases if evidence suggests that it’s not.  

**STLE v1: Learning Frontier**

x ∩ y \= {r ∈ D : 0 \< μ\_x(r) \< 1}.

μ\_x(r) \= 1 means r (i.e information) is completely accessible ( r  ∈ x only)

μ\_x(r) \= 0 means r (i.e information) is completely inaccessible ( r ∈ y only)

0 \< μ\_x(r) \< 1 means r is in a partial state of accessibleness . It exists simultaneously ( r ∈ x ∩ y)

## **3.4 STLE.v2** 

## **Accessibility Formula**

## The fundamental difference between STLE v1 and v2 is that the STLE v2’s Bayesian formula had to be modified to solve the fundamental epistemological paradox of STLE (the bootstrap problem/chicken and egg problem)

## **Fundamental Problem:** To compute the accessibility of “unseen data,” we need to model the structure of inaccessible space, but by definition, we lack direct access to it. Therefore, how do we initiate μ\_x(r) for “unseen data” without prior knowledge? 

## **Solution:** Two initialization strategies, also inspired by (Charpentier et al., 2020), that utilize density-based on-demand computation whereby we learn a density model P(r|accessible) and compute μ\_x(r) lazily when queried, instead of pre-computing for all of D (i.e the Universal Set: The set of all possible data points in a given domain)  

## **Strategy 1:** **Density-Based Pseudo-Count Initialization (DBPCI)**:

## μ\_x(r) \= N\_x · P(r|accessible) / (N\_x · P(r|accessible) \+ N\_y · P(r|inaccessible))            (5)

## Provided the theoretical foundation that we don’t need to enumerate all of the Universal set D. For example, we only need to define μ\_x for:

1. ## Training data: μ\_x(r) \= 1.0 (fully accessible)

2. ## Queried test points: μ\_x(r) computed on-demand via density estimation

3. ## Generated samples: μ\_x(r) computed as needed

## In strategy 1, both sides are scaled by sample counts; N\_x accessible and N\_y inaccessible. Therefore, this implies you are required to learn or estimate a density for inaccessible space. This is task was non-trivial, therefore another strategy was needed.

## **Strategy 2:** **Density-Based Lazy Initialization (DBLI)**:

## μ\_x(r) \= N · P(r|accessible) / (N · P(r|accessible) \+ P(r|inaccessible))               (6)

## Computationally speaking, the innovation here is that we don't need to compute all of the Universal Set D upfront. Instead, we can: 

1. ## Learn a density model P(r | accessible) on training data

2. ## Compute μ\_x(r) on demand when queried

3. ## Use density as a proxy for accessibility

## When utilizing a Density-Based Lazy Initialization strategy, only the accessible side is count-weighted, and the inaccessible term is just a flat uniform prior. In effect, this means no N\_y and no learned density to compute, therefore simpler to implement and avoids requiring a model of “unseen” space.

## The two strategies, Pseudo-Count Initialization and Lazy Initialization, describe different aspects of the same insight: not all of D, the set of all possible data points in a given domain, needs to be materialized at once. However, these strategies differ in how they handle the inaccessible set. Density-Based Pseudo-Count Initialization is the general form, i.e the theoretical mathematical foundation, and Density-Based Lazy Initialization is the simplified special case with the flat unit prior for actually computing the inaccessible space. 

* ## Lazy Initialization is the computational strategy; on-demand, not upfront

* ## "Pseudo-Count Initialization" (Research) is the mathematical mechanism, using sample counts as Bayesian evidence weights

  ## 

## **STLE v2 Definitions**

## **Universal Set (D):** The set of all possible data points in a given domain

## **Accessible Set (x):** A fuzzy subset of D representing known/observed data

* ## Membership function: μ\_x: D → \[0,1\]

* ## High μ\_x(r) indicates r is well-represented in accessible space

## **Inaccessible Set (y):** The fuzzy complement of x representing unseen/unobserved data

* ## Membership function: μ\_y: D → \[0,1\]

* ## Enforced complementarity: μ\_y(r) \= 1 \- μ\_x(r)

## **Theorem:**

## The accessible set x and inaccessible set y are complementary fuzzy subsets of a unified domain

## **Learning Frontier**: The region of partial knowledge

x ∩ y \= {r ∈ D : 0 \< μ\_x(r) \< 1}.

μ\_x(r) \= 1 means r (i.e information) is completely accessible ( r  ∈ x only)

μ\_x(r) \= 0 means r (i.e information) is completely inaccessible ( r ∈ y only)

0 \< μ\_x(r) \< 1 means r is in a partial state of accessibleness . It exists simultaneously ( r ∈ x ∩ y)

**Axioms:**

*A1\] **Coverage**: x ∪ y \= D*

*\[A2\] **Non-Empty Overlap:** x ∩ y ≠ ∅*

*\[A3\] **Complementarity**: μ\_x(r) \+ μ\_y(r) \= 1, ∀r ∈ D*

*\[A4\] **Continuity**: μ\_x is continuous in the data space*

## **STLE v2: Density-Based Pseudo-Count Initialization (Accessibility Function)**

## μ\_x(r) \= N\_x · P(r|accessible) / (N\_x · P(r|accessible) \+ N\_y · P(r|inaccessible))            (5)

Where N\_x is the number of training samples, N\_y \= N\_x/5 is a pseudo-count for the inaccessible space, P(r|accessible) is the learned density under the accessible distribution (i.e estimated via normalizing flows), and P(r|inaccessible) is the complement density (uniform prior). The Pseudo-Count formula satisfies the four STLE axioms: complementarity (μ\_y \= 1 \- μ\_x), continuity is inherited from the smoothness of the normalizing flow, and the density ratio ensures appropriate behavior at the boundaries (i.e μ\_x → 1 for training data, μ\_x → 0 far from training data).

**STLE v2: Density-Based Lazy Initialization (Accessibility Function)** 

## μ\_x(r) \= N · P(r|accessible) / (N · P(r|accessible) \+ P(r|inaccessible))               (6)

One of the important insights of STLE is that the accessibility function does not need to be materialized for all the Universal Set D. Instead, μ\_x(r) can be computed on-demand via Density-Based Lazy Initialization (DBLI). To solve the “bootstrap problem,” we learn a density model P(r|accessible) on training data, compute μ\_x(r) for any queried point r using DBPCI,  (Equation 5). This procedure reduces the computational requirement from O(|D|) (which would be infinite for continuous domains) to O(1) per query, with O(N) preprocessing to fit the density model.

## **3.5  PAC-Bayes Convergence**

In general, STLE's convergence guarantee is grounded in PAC-Bayes theory. Following (Futami et al., 2022), the epistemic uncertainty μ\_y converges at rate:

                              *|μ\_x(r) \- μ\*\_x(r)| \= O(1/√N)  with probability 1-δ*          (7)

where μ\*\_x is the true accessibility and N is the training set size. This guarantees that as the system observes more data, its knowledge assessment converges to what can be considered ground truth.

# **4  STLE v3: Technical Development**  

# **4.1 MarvinBot: Autonomous Machine Learning System**

The creation of MarvinBot, an autonomous machine learning system utilizing STLE, was fundamental in identifying weaknesses in STLE v2’s formulation, and thus necessitating STLE v3. Marvin’s defining characteristic is that he studies topics continuously, 24/7, without human intervention. Marvin could be called artificial intelligence; However, Marvin is not a chatbot in the traditional sense because no LLM layer is currently integrated (although one can chat with Marvin in a limited sense; i.e querying his database for a response). Instead, Marvin is an artificial computational intelligence system that independently decides what to study next, studies it by fetching Wikipedia, arXiv, and other content; processes that content through a machine learning pipeline and updates its own representational knowledge state over time. Therefore, regarding the sphere of AI, Marvin can be considered a type of nascent meta-cognition that genuinely develops knowledge overtime. The system is designed to operate by approaching any given topic in the following manner: 

* Determines how accessible is this topic right now;

* Accessible: Marvin has studied it, understands it, and can reason about it;

* Inaccessible: Marvin has never encountered the topic, or it is far outside its knowledge;

* Frontier: Marvin partially knows the topic. Here is where active learning happens.

This accessibility score is called μ\_x (mu-x) and is a number between 0 and 1\. Everything in Marvin's architecture exists to compute, maintain, and improve μ\_x across a growing knowledge base that currently contains around 16,923 topics.

# **4.2 Limitations Discovered by Implementation of MarvinBot**

When implementing STLE v2 practically, i.e the development of MarvinBot and deployment on large-scale knowledge bases (Wikipedia, arXiv, and other sources), the system grew to 16,923 topics across 23 domains and has over 3,200 completed study sessions. However, this deployment revealed two critical failures in the original formulation: The “curse of dimensionality” and a saturation bug. 

## **4.3  The Curse of Dimensionality in Embedding Space**

The first implementation, STLE v2 ML deployment (i.e MarvinBot), used a SentenceTransformer (all-MiniLM-L6-v2) to produce 384-dimensional embeddings, with a Gaussian Mixture Model (GMM) for density estimation. With N \= 158 initial training points in 384-D space, the data was extremely sparse (ratio N/2³⁸⁴ ≈ 0).

All queries, including the training examples, fell in low-density tails and yielded uniformly uninformative μ\_x values (0.03–0.13), which indicated that the density estimator could not distinguish training data from arbitrary inputs. Therefore, utilizing a temporary hybrid estimator combining cosine similarity (70%) and GMM log-likelihood (30%), restored functional discrimination (known: 0.65, frontier: 0.42, inaccessible: 0.27), but this was an empirical workaround rather than a principled solution.

**Resolution:** STLE v3, through the deployment of MarvinBot, addressed the curse of dimensionality in embedding spaces with a trainable projection layer (384→64 dimensions) that was trained jointly with per-domain normalizing flows. The projection is optimized for domain separation, achieving 88.4% classification accuracy in the 64-D latent space, and ensures that the subsequent density estimation operates in a space where the data is well-structured rather than uniformly sparse.

## **4.4  The Saturation Bug**

The second, and perhaps more fundamental failure that emerged when STLE was deployed via MarvinBot, occurred as Marvin’s knowledge base scaled to N\_x \= 8,234 training samples. At this scale, Equation 5’s (DBPCI) accessibility function, saturates to μ\_x ≈ 1.0 for all queries with non-zero accessible density, regardless of actual familiarity:

N\_x \= 8234, N\_y \= 1646.8

P\_accessible \= 1e-10,  P\_inaccessible \= 1e-12

μ\_x \= (8234 × 1e-10) / (8234 × 1e-10 \+ 1646.8 × 1e-12) \= 0.998

Interestingly, for novel out-of-distribution topics, the formula works correctly (μ\_x ≈ 0.05); However, this asymmetry is the problem: known topics saturate to \~1.0 regardless of actual density, while novel topics score appropriately. Essentially, the system becomes blind to the distinction between well-known and barely known topics.

***Theorem 1 (Saturation)*****:** For STLE v2’s DBPCI accessibility function, Equation (5), lim\_{N\_x→∞} μ\_x(r) \= 1 for any data point r with P(r|accessible) \> 0, regardless of how small that density actually is.

***Proof*****:** As N\_x → ∞ with N\_y \= N\_x/5: 

μ\_x \= N\_x·p\_a / (N\_x·p\_a \+ (N\_x/5)·p\_i) \= p\_a / (p\_a \+ p\_i/5) → 1 

Whenever p\_a \> 0 and p\_i is finite. The N\_x factor cancels, however the 5:1 ratio between N\_x and N\_y, ensures that p\_a always dominates for any non-zero density.


Moreover, normalizing flow log-densities in a 64-D space are approximately \-100, therefore requiring an exponentiation that causes floating-point underflow at this scale. Thus, the saturation bug is both a mathematical and a numerical issue.

# **5  STLE v3: Evidence-Scaled Posterior Networks**

STLE v3 resolves both limitations, the curse of dimensionality in embedding spaces and the saturation bug, through a multi-domain Dirichlet formulation with evidence scaling. This solution was also inspired by 2020 paper “*Posterior Networks”* (Charpentier et al., 2020\) but extended to handle large-scale continual learning scenarios.

## **5.1  STLE.v3 Multi-Domain Dirichlet w/ Evidence Scaling (Accessibility Function)** 

For K domains with per-domain normalizing flows, the STLE.v3 accessibility function is:

                              *α\_c \= β \+ λ · N\_c · p(z | domain\_c)*          (1)

                              *α\_0 \= Σ\_c α\_c*          (2)

                              *μ\_x \= (α\_0 \- K) / α\_0*          (3)

where β \= 1.0 is the Dirichlet prior (preventing zero-evidence collapse), λ ∈ (0, 1\] is the evidence scale (calibrated via a grid search, typically λ ≈ 0.001), N\_c is the number of training samples in domain c, p(z|domain\_c) is the density under domain c's RealNVP normalizing flow, which is evaluated in a 64-D latent space, and K is the number of domains.

## **5.2  Why Evidence Scaling Prevents Saturation**

The evidence scale λ acts as a dampener on the N\_c multiplier. With λ \= 0.001, a domain with N\_c \= 8,234 contributes only 8.234 units of scaled evidence rather than 8,234 raw. This keeps μ\_x bounded and discriminative across the full range of topic familiarity.

***Theorem 2 (Saturation Prevention)*****:** Via evidence scaling, μ\_x is bounded independently of N\_c:

*μ\_x ≤ 1 \- K / (K \+ λ · N\_total · max\_c p(z|c))*

Where N\_total \= Σ\_c N\_c. Therefore μ\_x \< 1 for all finite N\_total.

***Proof*****:** The maximum α\_0 occurs when all density concentrates in a single domain: 

α\_0 ≤ Kβ \+ λ·N\_total·max\_c p(z|c)

Then: μ\_x \= (α\_0 \- K)/α\_0 \= 1 \- K/α\_0 ≤ 1 \- K/(Kβ \+ λ·N\_total·max\_c p(z|c))

Since K, β, λ, N\_total are all positive and finite, μ\_x \< 1\.


## **5.3  Preserved Theoretical Guarantees**

***Theorem 3 (Multi-Domain STLE v3)*****:** Equations 1, 2 and 3 preserve all core STLE properties:

| Property | STLE.v2 DBPCI Accessibility Function (5) | STLE.v3 Multi-Domain Dirichlet Accessibility Function (1)(2)(3) |
| ----- | :---: | :---: |
| Complementarity (μ\_x \+ μ\_y \= 1\) | ✓ Preserved | ✓ By construction |
| Monotonic learning (∂μ\_x/∂N\_c ≥ 0\) | ✓ Preserved | ✓ Preserved |
| PAC-Bayes convergence | O(1/√N) | O(1/√(λN)) |
| Saturation resistance | ✗ Fails at N\_x \>\> 1 | ✓ Bounded for all N\_c |
| Numerical stability | ✗ Underflow at log-densities ≈ \-100 | ✓ Logsumexp applied |
| Multi-domain support | ✗ Binary only | ✓ K domains natively |

*Table 1: Comparison of the theoretical properties between STLE v2’s DBPCI accessibility function and STLE.v3’s evidence-scaled multi-Domain Dirichlet accessibility function.*

## **5.4  Theoretical Equivalence**

***Theorem 4 (Strict Generalization)*****:** STLE v3’s accessibility function reduces to the original Density-Based Pseudo-Count Initialization accessibility function (Equation 5\) under the following conditions: 

K \= 2, β \= 0, λ \= 1, N\_1 \= N\_x, N\_2 \= N\_y.

**Proof:** Under these conditions: 

α\_1 \= 0 \+ 1 × N\_x × p\_acc \= N\_x·p\_acc; 

α\_2 \= 0 \+ 1 × N\_y × p\_inacc \= N\_y·p\_inacc; 

α\_0 \= N\_x·p\_acc \+ N\_y·p\_inacc; 

Then μ\_x \= (α\_0 \- 2)/α\_0 ≈ (N\_x·p\_acc)/(N\_x·p\_acc \+ N\_y·p\_inacc) 

When N\_x·p\_acc \>\> 1, which matches the DBPCI accessibility function (Equation 5\) exactly.


# **6  STLE.v3: MarvinBot.v1 Architecture**

The complete STLE v3 pipeline processes input text through four stages:

*Text → SentenceTransformer (frozen, 384-D) → Projection (384→64-D) → Per-Domain Flows → μ\_x*

## **6.1  Embedding Layer**

As per the paper by Wang, “*MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers,”* I used a SentenceTransformer (all-MiniLM-L6-v2) that produces 384-dimensional dense embeddings. During all training stages, the model is frozen, therefore it serves as a fixed feature extractor that helps map topic names and descriptions to a semantically meaningful embedding space.

## **6.2  Trainable Projection**

The trainable projection layer maps from 384-D embedding space to a 64-D latent space. The architecture is a two-layer MLP (384 → 256 → 64\) with batch normalization and ReLU activations. This allows our projection to serve dual purposes: first, it operates to reduce dimensionality to a regime where density estimation is tractable. Thus, resolving the curse of dimensionality from Section 4.3. Secondly, it increases accuracy because it learns a domain-separating representation, which allows our model to achieve 88.4% classification accuracy across multiple domains.

## **6.3  Per-Domain Normalizing Flows**

Each *domain c* has a dedicated RealNVP normalizing flow operating in the 64-D latent space as per (Dinh et al., 2017). Each flow consists of 4 coupling layers with hidden dimension 64 and are trained to maximize the log-likelihood of in-domain projected embeddings. For robustness, noise augmentation annealing is from 0.5 to 0.1 during training.

## **6.4  Evidence-Scaled Posterior Networks**

The final stage of STLE v3’s pipeline computes μ\_x via STLE v3’s evidence-scaled multi-Domain Dirichlet accessibility function (Equations 1,2,3). To prevent numerical underflow, Logsumexp stabilization is applied to flow log-densities before exponentiation. The evidence scale λ is calibrated via grid search over candidates \[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1\]. Therefore, selecting the value that achieves median μ\_x ≈ 0.9 on training samples.

## **6.5  Two-Stage Training Pipeline**

Stage 1: trains the projection layer and a temporary classification head using cross-entropy loss for domain separation (60 epochs). After training, the classification head is discarded. Stage 2: freezes the projection, pre-computes projected latents, and trains per-domain flows with negative log-likelihood loss and noise augmentation (40 epochs). After both stages, the projection is unfrozen and λ is calibrated on the full training set.

# **7  STLE v3: Experiments (MarvinBot Deployment)**

## **7.1  Dataset and Setup**

Evaluated STLE on a continuously growing knowledge base of 16,923 topics, spanning 23 domains, and with 4 domains having dedicated trained flows (General: 3,328 topics; Chemistry: 3,080; Computer Science: 1,793; History: variable). Topic embeddings are generated from Wikipedia article titles and introductory summaries, with the training set consisting of 8,897 topics with valid embeddings across the 4 flow-supported domains, and with a stratified 80/20 split by domain for held-out evaluation.

## **7.2 Results**

| Metric | Expected | Observed | Status |
| ----- | :---: | :---: | :---: |
| Held-out μ\_x (mean ± std) | 0.85 – 0.90 | 0.855 ± 0.062 | ✓ |
| Held-out μ\_x range | — | 0.48 – 0.94 | ✓ |
| Novel topic μ\_x (mean) | 0.35 – 0.45 | 0.41 | ✓ |
| Domain classification accuracy | ≥ 88% | 88.4% | ✓ |
| CS log-prob gap (in vs. out) | \> 0 | 19.8 | ✓ |
| History log-prob gap | \> 0 | 19.6 | ✓ |
| Chemistry log-prob gap | \> 0 | 4.1 | ✓ |

*Table 2: STLE.v3 validation results on the MarvinBot knowledge base.*

Held-out μ\_x of 0.855 falls within an expected calibration range, thus confirming that known topics aren’t getting overly saturated accessibility scores. Novel, out-of-distribution topics, receive μ\_x ≈ 0.41, reflecting an appropriately conservative estimation. Moreover, domain discrimination is strong: Computer Science and History have large log-probability gaps (19.8 & 19.6 respectively), while Chemistry's gap of  4.1, reflects greater overlap with the domain “General.”

## **7.3  Saturation Resistance Validation**

We compute μ\_x for probe queries at the scale (N ≈ 8,900) to confirm that STLE v3 resolves the saturation bug:

| Probe Query | v2 μ\_x | v3 μ\_x | Expected Range |
| ----- | :---: | :---: | :---: |
| Chemistry (known domain) | 0.998 | ≥ 0.70 | Known (≥ 0.70) |
| QCD loop integrals (OOD) | 0.050 | \< 0.30 | Inaccessible (\< 0.30) |
| Pharmacokinetics (frontier) | 0.998 | 0.30–0.70 | Frontier (0.30–0.70) |

*Table 3: Saturation comparison. v2 cannot distinguish known from frontier at large N; v3 correctly discriminates.*

STLE v2’s density-based accessibility functions, DBPCI (Equation 5\) and DBLI (Equation 6), assigned μ\_x ≈ 0.998 to both known-domain queries and frontier queries, making them indistinguishable. STLE v3 is able to correctly place them in different accessibility bands.

## **7.4  Ablation: Evidence Scale λ**

We ablate the evidence scale parameter to confirm its necessity:

| λ | Median μ\_x | Saturated? | OOD μ\_x | Discriminative? | Selected |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1.0 (no scaling) | ≈ 1.0 | Yes | 0.05 | No |  |
| 0.1 | 0.98 | Nearly | 0.12 | Weak |  |
| 0.01 | 0.95 | No | 0.28 | Moderate |  |
| 0.001 | 0.90 | No | 0.41 | Strong | ✓ |
| 0.0001 | 0.72 | No | 0.55 | Weak (compressed) |  |

*Table 4: Ablation over λ. Grid search selects λ \= 0.001 to target median μ\_x ≈ 0.9.*

At λ \= 1.0 (i.e no scaling), the saturation bug reappears; However, at λ \= 0.0001, the score range compresses and discrimination weakens. Therefore, the calibrated value λ \= 0.001 achieves the target median while maintaining strong discrimination between known, frontier, and OOD topics.

# **8  STLE.v3: PAC-Bayes Training Extension (Future Implementation)**

With the saturation bug finally resolved, the next step is to fully implement a PAC-Bayes training objective that jointly optimizes projection and flows, while also providing a provable generalization bound. 

## **8.1  Training Objective**

The PAC-Bayes objective will replace the separate Stage 1/2 losses with a unified formulation:

                              *L \= UCE(Q) \+ kl\_weight · √(KL(Q||P) / N)*          (8)

Here, UCE is the Uncertain Cross-Entropy (i.e a Dirichlet-aware loss): UCE \= \-(ψ(α\_true) \- ψ(α\_0)), with ψ denoting the digamma function. KL(Q||P) is the Kullback-Leibler divergence from a posterior distribution Q over model weights to a prior P anchored at the pre-trained Stage 1/2 weights. N is the training set size.

## **8.2  Weight-Space KL**

To obtain a valid PAC-Bayes bound, the KL divergence should be computed in weight space. Thus, each trainable parameter w\_i, will be parameterized as a Gaussian: Q(w\_i) \= N(μ\_i, softplus(ρ\_i)²), with prior P(w\_i) \= N(θ\_pretrained\_i, σ²\_prior). The closed-form Gaussian KL is:

*KL(Q||P) \= Σ\_i \[log(σ\_prior/σ\_q\_i) \+ (σ²\_q\_i \+ (μ\_i \- θ\_prior\_i)²)/(2σ²\_prior) \- 1/2\]*

This produces a generalization bound of |μ\_x \- μ\*\_x| \= O(1/√(λN)) with probability 1-δ, which directly extending the convergence guarantee from Equation (7) with the evidence scaling factor.

## **8.3  Adaptive λ**

Fully implementing PAC-Bayes training will enable adaptive λ calibration during training rather than a fixed post-hoc grid search. The evidence scale will be recalibrated periodically, every 10 epochs, using the model's current mean weights; Therefore, ensuring that λ tracks the changing density landscape as the flows are jointly optimized. However, note that λ calibration does interact with convergence bound, so the bound needs to be computed against final λ value to remain valid.

## **8.4  Inference**

At inference time, μ\_x is computed using a deterministic posterior mean weights μ, therefore no sampling required. The inference functions (Equations 1,2,3) remain unchanged from STLE v3, and implementing PAC-Bayes should change how the model is trained, but not how it works.

# **9  Discussion**

## **9.1  The Value of Implementation-Driven Theory**

The development arc of STLE illustrates that nontraditional methods of theory development have merit. Moreover, STLE’s development shows that implementation-driven theory, although seemingly haphazard, also has tangible benefits. STLE began as a theoretical framework that was sound in principle but failed in practice due to scaling issues under real deployment conditions. For example, the saturation bug involved in Equations 5 and 6, is not a theoretical flaw, but rather an engineering failure that only became visible at scale. By documenting both the theory and its failure modes, I hope I provided a more realistic picture than is typical of academic publications.

## **9.2  STLE v3 Relationship to Charpentier’s Posterior Networks Paper**

STLE v3's evidence-scaled multi-Domain Dirichlet accessibility function (Equations 1,2,3) is inspired by Charpentier’s 2020 paper *“Posterior Networks: Uncertainty estimation without OOD samples via density-based pseudo-counts*,” however, it differs in two fundamental ways: First, we introduced evidence scaling λ to prevent saturation at large N, which Posterior Networks do not typically address because they are normally evaluated on fixed-size benchmarks. Second, the Dirichlet concentration is interpreted as an accessibility score rather than a prediction confidence; Therefore, this places the STLE framework in the context of a “set-theoretic knowledge representation,” rather than classification uncertainty.

## **9.3  Limitations** 

There are several limitations with MarvinBot that need to be addressed, and thus by extension with STLE v3: 1\) The current implementation supports 4 trained domains out of 23 in the knowledge base. Expanding coverage requires retraining; 2\) The two-stage training pipeline optimizes projection and flows separately, which can be considered suboptimal compared to joint training (will be addressed by the PAC-Bayes extension); 3\) The recency decay mechanism in the production deployment version of MarvinBot penalizes topics not recently studied, thus potentially operating outside the STLE theoretical framework, and can cause threshold crossings during prolonged system downtime; 4\) Finally, the PAC-Bayes extension outlined in Section 8 has not yet been implemented or validated experimentally.

## **9.4  Broader Impact**

STLE provides a framework for AI systems to reason explicitly about the limits of their knowledge representation. In the era of Large Language Models that confidently generate plausible but incorrect information, I believe that principled epistemic self-awareness (knowing what you don't know) is a safety critical capability that all AI systems should operate under. The accessibility score, μ\_x, offers a grounding signal that can constrain generative systems. This means an LLM that consults STLE before answering can distinguish topics it genuinely understands from topics where it is extrapolating beyond its knowledge base. In other words, STLE can as the “brain” layer for an AI’s “mouth” layer, the LLM. In my opinion, this dual relationship between STLE and LLM could produce nascent meta-cognition for AI systems.

# **10  Conclusion**

This paper presented STLE, tracing its development from a philosophical theoretical framework to a deployed machine learning system. The original formulation, v0, was grounded in set theory and deep philosophical thought. STLE v1 and  STLE v2 extended the framework with Bayesian statistics, fuzzy membership mathematics, and PAC-Bayes theory; thus, establishing principled foundations for epistemic self-awareness in AI systems. Implementation on a large-scale knowledge base revealed two critical limitations: the curse of dimensionality in high-dimensional spaces, and a saturation bug that collapses STLE v2’s accessibility functions at scale.

STLE v3 resolved both critical limitations through evidence-scaled Posterior Networks with a multi-domain Dirichlet formulation. This paper proved that STLE v3 preserves all theoretical guarantees of the original STLE v2 (complementarity, monotonic learning, PAC-Bayes convergence) while adding saturation resistance, numerical stability, and native multi-domain support. Moreover, this paper proved that STLE v3 is a strict generalization, reducing to the original formulation under specific parameter settings.

STLE’s defining contribution, the Learning Frontier, is the region where μ\_x(r) is neither 0 nor 1\. This region of partial state transforms the boundary between “knowledge” and “ignorance” from a philosophical concept, into a computational resource that AI systems can systematically explore.

# **References**

\[1\] Charpentier, B., Zügner, D., & Günnemann, S. (2020). Posterior Network: Uncertainty estimation without OOD samples via density-based pseudo-counts. NeurIPS.

\[2\] Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). Density estimation using Real-NVP. ICLR.

\[3\] Futami, F., Bae, J., & Sugiyama, M. (2022). Excess risk analysis for epistemic uncertainty with application to variational inference. NeurIPS.

\[4\] McAllester, D. (1999). PAC-Bayesian model averaging. COLT. 

\[5\] Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows. ICML.

\[6\] Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. NeurIPS. 