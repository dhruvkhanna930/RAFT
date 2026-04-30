# Unicode-Based RAG Poisoning

Comparative analysis of invisible-character adversarial attacks across five RAG variants.

See `docs/research-plan.md` for the full research plan and `CLAUDE.md` (project root) for
coding conventions, metric definitions, and the evaluation matrix.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python experiments/01_baseline_replication.py --dataset nq --n_questions 100
```

## Structure

```
src/attacks/    — PoisonedRAG, RAG-Pull, and Hybrid attack implementations
src/rag/        — Vanilla, Self-RAG, CRAG, TrustRAG, RobustRAG wrappers
src/retrievers/ — Contriever and BGE dense retrievers
src/llms/       — Ollama, OpenAI, Anthropic, DeepSeek clients
src/defenses/   — Paraphrase, PPL filter, Unicode normalise, duplicate filter
src/metrics/    — ASR, retrieval P/R/F1, stealth, efficiency
experiments/    — Five numbered experiment scripts (01–05)
```
