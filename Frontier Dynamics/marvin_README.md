# MarvinBot

### An Autonomous AI Learning System Powered by STLE v3

**🟢 [See Marvin Live →](https://just-inquire.replit.app)**

*Watch Marvin study in real time. Observe μ_x scores across 16,923+ topics, chat with him, and see the learning frontier in action.*

---

## What Is MarvinBot?

MarvinBot is an autonomous AI learning system. Its defining characteristic is that it studies topics continuously, 24/7, without human intervention.

Marvin is not a chatbot in the traditional sense. He is an AI that independently decides what to study next, fetches information from sources like Wikipedia, arXiv, and Internet Archive, processes that content through a machine learning pipeline, and updates his own knowledge representation over time.

Everything in Marvin's architecture exists to compute, maintain, and improve μ_x — the STLE v3 accessibility score — across a growing knowledge base.

---

## How Marvin Uses STLE v3

Marvin approaches every topic by asking one question: **how accessible is this topic right now?**

| μ_x Range | Knowledge State | Marvin's Response |
|-----------|----------------|-------------------|
| ≥ 0.70 | **Known** | "I've studied this. I understand it." |
| 0.30 – 0.70 | **Frontier** | "I partially know this. Worth studying." |
| < 0.30 | **Unknown** | "I've never encountered this. It's outside my knowledge." |

The μ_x score drives every decision Marvin makes: which topics to study next, when to revisit stale knowledge, how to balance exploration and exploitation, and when to consolidate through sleep cycles.

---

## What Marvin Validates

MarvinBot is the production proof that STLE v3 works at scale, under continuous autonomous operation. The deployment has demonstrated:

| Metric | Value |
|--------|-------|
| Topics in knowledge base | 16,923+ |
| Completed study sessions | 3,200+ |
| Trained STLE domains | 4 (General, Chemistry, Computer Science, History) |
| Total domains in database | 23 |
| Held-out μ_x (mean) | 0.855 |
| Novel/OOD μ_x (mean) | 0.41 |
| Domain classification accuracy | 88.4% |
| Study interval | ~30 seconds |
| Saturation at scale | None (v3 bounded) |

---

## What You'll See on the Dashboard

**[→ Open Live Dashboard](https://just-inquire.replit.app)**

When you visit Marvin's dashboard, you can observe:

- **Currently Studying** — The topic Marvin is actively learning about right now
- **Knowledge Classification** — Live counts of Known, Frontier, and Unknown topics
- **Recent Activity** — A feed of what Marvin has studied recently
- **Knowledge Base** — Browse topic cards showing μ_x confidence scores, evidence counts, and timestamps
- **Chat with Marvin** — Ask him about any topic and he'll tell you what he knows, what he's still learning, and where his gaps are

---

## What MarvinBot Is Not

MarvinBot is **not** an LLM wrapper, a chatbot framework, or a RAG system. There is no large language model generating Marvin's knowledge — all intelligence is algorithmic. Marvin's chat responses come from querying his knowledge graph and STLE scores, not from prompting a language model.

LLM integration is planned as a future layer — STLE as the "brain" (epistemic grounding), LLM as the "mouth" (natural language generation). But the current system demonstrates that principled epistemic self-awareness can exist independently of language models.

---

## Further Reading

- **[STLE v3 Specification](../stle/v3/STLE_v3.md)** — The theoretical framework powering Marvin
- **[Architecture Overview](architecture.md)** — How STLE v3 fits into Marvin's stack
- **[Research Paper](../stle/v3/Set_Theoretic_Learning_Environment_Paper.md)** — Formal academic paper covering the full STLE v1→v3 arc

---

*"Marvin doesn't just store information — he knows what he knows, knows what he doesn't know, and systematically studies the boundary between the two."*
