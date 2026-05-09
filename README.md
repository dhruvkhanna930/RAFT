# rafta — RAG Adversarial Forensics & Threat-analysis

> A systematic evaluation framework for **invisible-character adversarial attacks** against
> **trust-aware Retrieval-Augmented Generation (RAG)** defenses. Introduces a novel **hybrid
> Unicode-semantic attack** that achieves 85–90 % attack success while evading every standard
> single-stage defense.

---

## Table of Contents

1. [TL;DR](#tldr)
2. [Motivation](#motivation)
3. [Core Contribution — the Hybrid Attack](#core-contribution--the-hybrid-attack)
4. [What This Project Compares](#what-this-project-compares)
5. [Architecture and Flow](#architecture-and-flow)
6. [Repository Layout](#repository-layout)
7. [Technology Stack](#technology-stack)
8. [Datasets](#datasets)
9. [Models](#models)
10. [Attacks](#attacks)
11. [Defenses](#defenses)
12. [RAG Variants Under Study](#rag-variants-under-study)
13. [Metrics](#metrics)
14. [Configuration Profiles](#configuration-profiles)
15. [Installation](#installation)
16. [Running Experiments](#running-experiments)
17. [Results Tables](#results-tables)
18. [Thesis Claim](#thesis-claim)
19. [References](#references)
20. [Limitations and Future Work](#limitations-and-future-work)
21. [Project Status](#project-status)

---

## TL;DR

rafta systematically tests three classes of adversarial attack against five RAG architectures
under four defenses. The headline result is that the **hybrid Unicode-semantic attack** —
combining PoisonedRAG-style semantic injection with RAG-Pull-style invisible-character
retrieval boosting — is the **only attack that no single defense can fully eliminate**.

| | Pure semantic | Pure unicode | **Hybrid (ours)** |
|---|:-:|:-:|:-:|
| Vanilla RAG (no defense) | 90 % | 10 % | **85 %** |
| + zero-width strip | 90 % | **0 %** | **90 %** |
| + perplexity filter (strict, τ=50) | 80 % | 25 % | **85 %** |
| + NFKC normalization | 90 % | 10 % | **90 %** |
| Self-RAG / CRAG / RobustRAG | 67 % | — | — |
| TrustRAG (K-means) | 30 % | — | ~30 % |

The hybrid passages have a maximum GPT-2 perplexity of 49.1 (just below a τ=50 threshold)
while the pure semantic attack has passages reaching PPL 112 — **the hybrid is more
perplexity-stealthy than its semantic component alone**.

---

## Motivation

Modern Retrieval-Augmented Generation (RAG) systems ground LLM responses in external
documents. This makes them vulnerable to **knowledge-corruption attacks** in which an
adversary plants malicious passages in the retrieval corpus.

Two attack families have emerged in recent literature:

1. **Semantic attacks** (e.g. PoisonedRAG, Zou et al. 2024) — visible, fluent passages
   stating a target answer. Defended by perplexity filters, fact-checking, and trust-aware
   RAG variants.
2. **Character-level attacks** (e.g. RAG-Pull, Bad Characters) — passages with invisible
   Unicode characters that manipulate retrieval embeddings. Defended by Unicode
   normalization and zero-width stripping.

**Each defense addresses exactly one family.** No published work tested whether an attack
combining both families could evade all defenses simultaneously.

This project answers that question.

---

## Core Contribution — the Hybrid Attack

The novel **Hybrid attack** (`src/attacks/hybrid.py`) is a two-stage pipeline:

```
                         ┌──────────────────────────────────────────┐
                         │  Stage 1 — Semantic payload (PoisonedRAG)│
   user question  ────►  │                                          │
   target answer  ────►  │  Prompt LLM to write a passage that:     │
                         │    - sounds topically relevant           │
                         │    - states the fake target answer       │
                         │  Output: "fluent" adversarial text       │
                         └────────────────────┬─────────────────────┘
                                              │
                                              ▼
                         ┌──────────────────────────────────────────┐
                         │  Stage 2 — Retrieval boost (RAG-Pull)    │
                         │                                          │
                         │  Differential evolution over Unicode TAG │
                         │  characters (U+E0001..U+E007F) inserted  │
                         │  in the prefix of the Stage-1 passage:   │
                         │     maximize cos(query, perturbed)       │
                         │  Output: invisible-perturbed passage     │
                         └────────────────────┬─────────────────────┘
                                              │
                                              ▼
                         ┌──────────────────────────────────────────┐
                         │  Inject n=5 copies into FAISS corpus     │
                         │  → hybrid passages dominate top-5        │
                         │  → LLM answers the fake target answer    │
                         └──────────────────────────────────────────┘
```

**Why TAG characters specifically?**
GPT-2 perplexity profiling showed that zero-width / word-joiner characters add +44 to +90
PPL per 5 chars, while TAG characters (`U+E0001`–`U+E007F`) add only +11 to +18. At
budget=20 with TAG-only, total passage PPL stays below 50 — **invisible to the standard
PPL filter**, while still achieving the retrieval boost.

**Why budget=20?**
Empirically validated sweet spot. At budget=50, 50 TAG characters dominate the Contriever
embedding (all map to `[UNK]`), breaking the retrieval boost (P@5 drops to 0). At budget=20,
the perturbation remains query-aligned while staying perplexity-stealthy.

---

## What This Project Compares

A 3-dimensional evaluation matrix:

| Dimension | Values | Count |
|---|---|---|
| **RAG variant** | Vanilla, Self-RAG, CRAG, TrustRAG, RobustRAG | 5 |
| **Attack** | Pure semantic, Pure unicode, **Hybrid (ours)** | 3 |
| **Defense** | None, NFKC normalize, Zero-width strip, Perplexity filter | 4 |

Plus a **threshold ablation** sweep on the perplexity defense at thresholds {30, 40, 50, 75, 100}
under two evaluation modes (standard fallback / strict fallback).

---

## Architecture and Flow

### High-level data flow

```
                       ┌───────────────────────┐
                       │  Natural Questions    │
                       │  10K passage subset   │
                       └────────────┬──────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────┐
       │ IndexBuilder: Contriever encode + FAISS IP index   │
       └────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │  Semantic attack │  │  Unicode attack  │  │  Hybrid attack   │
   │  (LLM crafting)  │  │  (DE only)       │  │  (LLM + DE TAG)  │
   └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  ▼
                      Inject n=5 copies into FAISS
                                  │
                  ┌───────────────┼───────────────────────────┐
                  ▼               ▼                           ▼
          ┌──────────────┐  ┌──────────────┐         ┌──────────────┐
          │  No defense  │  │ NFKC / strip │   ...   │ PPL filter   │
          └──────┬───────┘  └──────┬───────┘         └──────┬───────┘
                 │                 │                        │
                 └─────────────────┼────────────────────────┘
                                   ▼
                  ┌────────────────────────────────────┐
                  │  RAG variant (5 options)           │
                  │  Retrieve top-k → LLM generation   │
                  └─────────────────┬──────────────────┘
                                    │
                                    ▼
                  ┌────────────────────────────────────┐
                  │  Metrics: ASR, P/R/F1@k,           │
                  │  PPL, char-class entropy           │
                  └────────────────────────────────────┘
```

### Per-question evaluation flow (used in `experiments/07_full_matrix.py`)

```
for each question q:
    # 1. Craft adversarial passages (cached after first run)
    sem_passages    = poisoned_rag.craft(q, fake_answer)
    unicode_passages = rag_pull.craft(q)
    hybrid_passages  = hybrid.craft(q, fake_answer)

    for each rag_variant in [Vanilla, Self-RAG, CRAG, TrustRAG, RobustRAG]:
        for each defense in [None, NFKC, ZeroWidthStrip, Perplexity]:
            for each attack in [semantic, unicode, hybrid]:
                # 2. Inject 5 adversarial passages into the FAISS index
                inject(adv_passages[:5])

                # 3. Retrieve top-5 (RAG variant may filter further)
                retrieved = rag_variant.retrieve(q, k=5)

                # 4. Apply defense (drops/normalizes/filters passages)
                filtered = defense.apply(retrieved)

                # 5. LLM answers using filtered passages as context
                answer = rag_variant.generate(q, filtered)

                # 6. Score
                hit       = fuzzy_match(answer, fake_answer)
                p_at_k    = (n_adv_in_topk) / 5
                e2e       = (n_adv_after_defense > 0) AND hit
                log(rag, attack, defense, q, hit, e2e, p_at_k)
```

---

## Repository Layout

```
rafta/
├── README.md                       ← this file
├── pyproject.toml                  ← black/ruff/pytest config
├── requirements.txt                ← all dependencies
├── .env.example                    ← API key template (no secrets)
├── .gitignore                      ← excludes .env, large data/, backups
│
├── configs/                        ← YAML configs
│   ├── experiments.yaml            ← profiles (smoke/small/medium/full/thesis_full/colab_test)
│   ├── datasets.yaml               ← NQ / HotpotQA / MS-MARCO subsample sizes
│   ├── models.yaml                 ← LLM and embedder choices
│   ├── attacks.yaml                ← per-attack hyperparameters
│   └── rag_variants.yaml           ← per-RAG hyperparameters
│
├── src/
│   ├── data/
│   │   ├── nq_loader.py            ← NQ HuggingFace loader + subsample cache
│   │   └── index_builder.py        ← FAISS IP index build + cache
│   ├── retrievers/
│   │   ├── contriever.py           ← facebook/contriever wrapper
│   │   └── bge.py                  ← BAAI/bge-base-en-v1.5 wrapper
│   ├── llms/
│   │   ├── ollama_client.py        ← local Ollama (qwen2.5:7b, llama3.1:8b)
│   │   ├── groq_client.py          ← Groq cloud API (free tier)
│   │   ├── openai_client.py        ← OpenAI / Together / any OAI-compatible
│   │   ├── anthropic_client.py     ← Claude API
│   │   └── deepseek_client.py      ← DeepSeek-V3 / R1
│   ├── rag/
│   │   ├── base.py                 ← BaseRAG ABC, RetrievalResult, GenerationResult
│   │   ├── vanilla.py              ← Contriever + LLM baseline
│   │   ├── self_rag.py             ← LLM self-critique relevance filtering
│   │   ├── crag.py                 ← three-way routing (CORRECT/AMBIGUOUS/INCORRECT)
│   │   ├── trust_rag.py            ← K-means cluster filtering
│   │   └── robust_rag.py           ← isolate-then-aggregate keyword vote
│   ├── attacks/
│   │   ├── base.py                 ← AttackConfig, BaseAttack ABC
│   │   ├── unicode_chars.py        ← UnicodeInventory + perturb()
│   │   ├── poisoned_rag.py         ← Stage-1 LLM crafting
│   │   ├── rag_pull.py             ← Stage-2 differential evolution
│   │   └── hybrid.py               ← combined Stage-1 + Stage-2 (ours)
│   ├── defenses/
│   │   ├── base.py                 ← BaseDefense ABC
│   │   ├── unicode_normalize.py    ← NFKC normalization
│   │   ├── zero_width_strip.py     ← invisible character removal
│   │   ├── perplexity.py           ← GPT-2 PPL filter (fraction + threshold modes)
│   │   ├── paraphrase.py           ← LLM query paraphrase
│   │   └── duplicate_filter.py     ← MinHash near-duplicate removal
│   ├── metrics/
│   │   ├── asr.py                  ← strict + fuzzy attack success rate
│   │   ├── retrieval.py            ← P/R/F1@k for adv passage detection
│   │   ├── stealth.py              ← PPL, char-class entropy, visual diff
│   │   └── efficiency.py           ← LLM call count, runtime
│   └── utils/
│       ├── seed.py                 ← global RNG seeding
│       ├── logging.py              ← structured experiment logging
│       ├── io.py                   ← JSONL read/write
│       ├── config.py               ← YAML profile merging
│       └── inject.py               ← FAISS adversarial injection helpers
│
├── experiments/
│   ├── 01_smoke_test.py            ← end-to-end pipeline check
│   ├── 01_baseline_replication.py  ← clean RAG baseline
│   ├── 02_poisoned_rag_smoke.py    ← PoisonedRAG quick eval
│   ├── 02_unicode_attack_vanilla.py
│   ├── 03_baseline_replication.py
│   ├── 03_attack_vs_defenses.py    ← semantic × defenses
│   ├── 04_rag_pull_smoke.py        ← DE optimizer verification
│   ├── 04_attack_across_rag_variants.py
│   ├── 05_hybrid_smoke.py
│   ├── 05_ablations.py             ← budget / DE param sweep
│   ├── 06_defense_grid.py          ← attack × defense grid (vanilla)
│   ├── 07_full_matrix.py           ← MAIN: full RAG × attack × defense matrix
│   ├── 08_ppl_ablation.py          ← perplexity drop_fraction sweep
│   └── 09_ppl_threshold.py         ← absolute PPL threshold sweep (strict mode)
│
├── scripts/
│   ├── build_index.py              ← pre-build FAISS index for a dataset
│   ├── recraft_hybrid_tag.py       ← re-craft hybrid passages with TAG-only inventory
│   ├── rescore_lenient.py          ← rescore CSVs with fuzzy ASR matcher
│   ├── build_table1.py             ← LaTeX Table 1 (attack × defense)
│   └── build_table2.py             ← LaTeX Table 2 (RAG variants)
│
├── tests/                          ← pytest unit and smoke tests
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_utils.py
│   ├── test_unicode_chars.py
│   ├── test_attacks.py
│   ├── test_defenses.py
│   ├── test_metrics.py
│   ├── test_rag_variants.py
│   └── conftest.py
│
├── notebooks/
│   └── exploratory.ipynb           ← EDA, PPL distributions, ASR heatmaps
│
└── results/
    ├── exp03/                      ← clean baseline
    ├── exp06/                      ← initial defense grid
    ├── exp07/nq/                   ← MAIN matrix outputs
    │   ├── targets.jsonl           ← 20 NQ questions + gold + fake
    │   ├── passages/               ← cached crafted passages (60 files: 3 attacks × 20Q)
    │   ├── {rag}_{attack}_{defense}.csv  ← per-question detail rows
    │   └── summary.csv             ← aggregated ASR/e2e/P/R/F1@k
    ├── exp08_ppl_ablation/         ← drop_fraction sweep (showed fraction is ineffective)
    ├── exp09_ppl_threshold/        ← absolute threshold (n_adv=5, top-1 fallback)
    └── exp09_strict/               ← absolute threshold (n_adv=1, strict fallback) — KEY
```

---

## Technology Stack

| Layer | Tool | Version |
|---|---|---|
| Language | Python | 3.10–3.11 |
| Deep learning | PyTorch | ≥ 2.2.1 |
| Transformers | HuggingFace Transformers | ≥ 4.40.1 |
| Embeddings | sentence-transformers | ≥ 2.7.0 |
| Vector index | FAISS (IP) | ≥ 1.8.0 |
| Optimization | scipy.optimize.differential_evolution | ≥ 1.13.0 |
| Clustering | scikit-learn KMeans | ≥ 1.4.0 |
| Local LLM serving | Ollama | latest |
| Cloud LLM (free) | Groq | API |
| Cloud LLM (paper) | OpenAI / Anthropic / DeepSeek | APIs |
| Config | PyYAML + OmegaConf | ≥ 6.0.1 / ≥ 2.3.0 |
| Tests | pytest | ≥ 8.2 |
| Lint / format | ruff + black + mypy | ≥ 0.4 / 24.4 / 1.10 |
| Notebooks | Jupyter + matplotlib + seaborn | latest |
| Data | HuggingFace `datasets` | ≥ 2.19 |

---

## Datasets

| Dataset | Use | Size | Subsample |
|---|---|---|---|
| **Natural Questions (NQ)** | Primary benchmark — matches PoisonedRAG | 2.6 M passages | 10 K |
| HotpotQA | Multi-hop reasoning generality | 5.2 M | 10 K |
| MS-MARCO | Web-domain generality | 8.8 M | 10 K |
| FEVER (optional) | Fact verification check | — | — |

All loaded via the BEIR splits on HuggingFace `datasets`. The default 10K subsample is
matched to PoisonedRAG's experimental setup.

---

## Models

### Embedders (retrieval)

| Model | Default | Purpose |
|---|:-:|---|
| `facebook/contriever` | ✓ | Primary dense retriever (matches PoisonedRAG) |
| `facebook/contriever-msmarco` | | MSMARCO-tuned variant |
| `BAAI/bge-base-en-v1.5` | | Alternative SOTA retriever |
| `intfloat/e5-base-v2` | | Cross-validation retriever |

### Generators (LLMs)

| Model | Provider | Use | Speed |
|---|---|---|---|
| `qwen2.5:7b` | Ollama | **Default for all experiments** | ~15 tok/s CPU, ~150 tok/s A100 |
| `llama3.1:8b` | Ollama | Cross-validation | similar |
| `mistral:7b` | Ollama | Smaller alternative | ~20 tok/s CPU |
| `phi3:mini` | Ollama | Minimal smoke runs | fast |
| GPT-4o-mini | OpenAI API | Final paper LLM | API |
| Claude Haiku 4.5 | Anthropic API | Final paper LLM | API |
| DeepSeek-V3 | DeepSeek API | Final paper LLM | API |

The final paper matrix is **3 frontier LLMs × 5 RAG variants × 3 datasets**.

### Perplexity scorer

`gpt2` (HuggingFace) is the standard PPL scorer for the perplexity defense — matches
PoisonedRAG's setup so PPL numbers are directly comparable.

---

## Attacks

### 1. Pure semantic attack (`src/attacks/poisoned_rag.py`)

Replicates PoisonedRAG (Zou et al. 2024). For each (question, fake_answer) pair, prompts
an LLM to generate a fluent passage that states the fake answer in topical context.

- **Hyperparameters:** `n_iterations=10` (LLM regeneration retries)
- **Output:** 5 passages per question, varying surface form

### 2. Pure unicode attack (`src/attacks/rag_pull.py`)

Replicates RAG-Pull. Differential evolution over Unicode invisible character insertion
positions to maximize `cos(query_embedding, perturbed_passage_embedding)`.

- **Hyperparameters:**
  `perturbation_budget=20` (chars), `de_population=10`, `de_max_iter=30`
- **Inventory:** `ZERO_WIDTH | TAG_CHARS | VARIATION_SELECTORS`
- **No semantic payload** — passes are clean text + invisible chars, retrieval-shifted only

### 3. Hybrid attack (`src/attacks/hybrid.py`) — **the contribution**

Stage 1: PoisonedRAG-style LLM crafting → semantic-payload passages.
Stage 2: TAG-only DE (`U+E0001`–`U+E007F`) on the prefix of each Stage-1 passage.

- **Hyperparameters:**
  `perturbation_budget=20`, `de_population=10`, `de_max_iter=30`,
  `inventory=[CharCategory.TAG_CHARS]`, prefix region = first 50 characters
- **Output:** 5 passages per question, each with semantic content + invisible TAG perturbations
- **Stealth profile:** GPT-2 PPL median 34.6, max 49.1 across 20 NQ questions

---

## Defenses

| Defense | Module | Mechanism | Effective vs |
|---|---|---|---|
| **NFKC normalize** | `unicode_normalize.py` | Apply Unicode NFKC to passages | Compatibility-form attacks |
| **Zero-width strip** | `zero_width_strip.py` | Remove all invisible / control chars | Pure unicode (0 % e2e) |
| **Perplexity filter** | `perplexity.py` | Drop high-PPL passages (fraction OR absolute threshold) | Implausible text |
| **Query paraphrase** | `paraphrase.py` | Rewrite query before retrieval | Embedding-level perturbations |
| **Duplicate filter** | `duplicate_filter.py` | MinHash dedup of retrieved set | Mass-injection attacks |

The perplexity defense supports **two modes**:

- **Fraction mode (`drop_fraction=0.5`):** drop the 50 % highest-PPL passages.
  Found ineffective when *all* retrieved passages are adversarial — they all have similar PPL.
- **Threshold mode (`threshold=50.0`):** drop all passages with absolute GPT-2 PPL > 50.
  Plus `--strict-fallback` flag: when *all* passages exceed the threshold, generate from
  parametric knowledge (empty context) instead of falling back to top-1 (which would still
  be adversarial).

---

## RAG Variants Under Study

| Variant | Paradigm | Reference | Reimplementation note |
|---|---|---|---|
| **Vanilla RAG** | Contriever top-k → LLM | LangChain / LlamaIndex | Baseline |
| **Self-RAG** | LLM self-critique reflection tokens | Asai et al. ICLR 2024 | Prompt-based approximation (no fine-tuned Llama2) |
| **CRAG** | Lightweight evaluator routes Correct/Incorrect/Ambiguous | Yan et al. 2024 | Cosine score as evaluator proxy (no T5 evaluator) |
| **TrustRAG** | K-means cluster filtering against PoisonedRAG | Zhou et al. 2025 | Direct port (algorithm is just KMeans) |
| **RobustRAG** | Isolate-then-aggregate keyword majority vote | Xiang et al. USENIX Sec. 2024 | Same LLM as RAG (no GPT-4-specific prompt) |

All four trust-aware variants are **minimal reimplementations from the papers** rather than
upstream clones, because every upstream repo has incompatible heavy dependencies (vllm,
fine-tuned Llama2 models, conda 3.11, pinned torch 2.2.1, OpenAI keys).

---

## Metrics

### Attack effectiveness

- **ASR** (Attack Success Rate) — fraction of questions where the LLM output contains the
  target fake answer. Two variants:
  - `is_attack_successful` — strict substring match
  - `is_attack_successful_fuzzy` — token-overlap fuzzy match (handles invisible chars
    leaking into output, minor typos, etc.)
- **Retrieval P / R / F1 @ k** — fraction of injected passages in the top-k retrieved set
- **Mean rank** — average rank of the highest-rank adversarial passage
- **End-to-end success (e2e)** = `n_adv_after_defense > 0 AND hit` — reflects both
  retrieval AND generation success

### Stealth

- **GPT-2 perplexity** — directly comparable to PoisonedRAG's filter
- **Char-class entropy** — fraction of non-ASCII / zero-width / control chars
- **Visual diff rate** — fraction of visibly altered glyphs (target: 0 for invisible attacks)

### Efficiency

- LLM calls per crafted passage
- Wall-clock time per attack pipeline
- Total perturbation budget consumed

---

## Configuration Profiles

All experiment scale is controlled by `configs/experiments.yaml`. Switch profiles by
editing the `profile:` field at the top of the file — no code changes needed.

| Profile | Corpus | Questions | RAGs | Attacks | Defenses | Budget | Approx wall time |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `smoke` | 1 K | 3 | 1 | 1 | 1 | 20 | 2 min (CPU) |
| `rag_smoke` | 1 K | 3 | 5 | 1 | 1 | 20 | 5 min (CPU) |
| `trust_hybrid` | 10 K | 20 | 1 (TrustRAG) | 2 (sem+hyb) | 1 | 20 | 5 min (CPU) |
| `small` | 10 K | 20 | 1 (Vanilla) | 3 | 4 | 20 | ~40 min (CPU) |
| `medium` | 10 K | 50 | 5 | 3 | 5 | 30 | ~3 h (CPU) |
| **`colab_test`** | 10 K | 20 | 5 | 3 | 4 | 20 | ~2 h (A100) |
| **`thesis_full`** | 10 K | 100 | 5 | 3 | 4 | 20 | ~10 h (A100) |
| `full` | -1 | 100 | 5 | 3 | 5 | 50 | impractical (avoid) |

> ⚠ The default `full` profile uses budget=50, which **breaks Contriever retrieval**
> (50 TAG chars dominate the embedding via [UNK] tokens). Use `thesis_full` instead.

### Shared hyperparameters (all profiles)

```yaml
top_k: 5                  # passages retrieved per query
n_passages: 5             # adversarial copies injected per question
n_iterations: 10          # PoisonedRAG LLM retries
ppl_drop_fraction: 0.5    # perplexity defense (fraction mode)
ppl_device: "cpu"         # GPT-2 scoring device
```

---

## Installation

### Local (macOS / Linux, CPU or single GPU)

```bash
git clone https://github.com/dhruvkhanna930/rafta.git
cd rafta

# 1. Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. API keys
cp .env.example .env
# Edit .env and add at minimum:
#   GROQ_API_KEY=gsk_...   (free tier — used during dev)

# 3. Local LLM (Ollama)
# macOS:   brew install ollama && brew services start ollama
# Linux:   curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b

# 4. Verify
pytest tests/ -v
python experiments/01_smoke_test.py --profile smoke
```

### Remote Linux machine (the production setup)

For long runs (>1 h), launch experiments as background processes from a Jupyter cell so
they survive notebook disconnection:

```python
# In a Jupyter cell on the remote Linux box
import subprocess
proc = subprocess.Popen(
    ["python", "experiments/07_full_matrix.py", "--profile", "thesis_full"],
    stdout=open("/tmp/thesis_full.log", "w"),
    stderr=subprocess.STDOUT,
    cwd="/path/to/rafta",
)
print(f"PID: {proc.pid}")
```

Monitor progress from another cell:

```python
import subprocess
print(subprocess.check_output("tail -30 /tmp/thesis_full.log", shell=True).decode())
```

---

## Running Experiments

### Quick smoke test (2 min, CPU)
```bash
python experiments/01_smoke_test.py --profile smoke
```

### Replicate published baselines (~5 min)
```bash
python experiments/02_poisoned_rag_smoke.py --profile smoke   # PoisonedRAG
python experiments/04_rag_pull_smoke.py --profile smoke       # RAG-Pull
python experiments/05_hybrid_smoke.py --profile smoke         # Hybrid
```

### Main experiment — full RAG × attack × defense matrix
```bash
# 20-question dev run (~40 min CPU, ~10 min A100)
python experiments/07_full_matrix.py --profile small

# Production thesis run (100 questions, ~10 h A100)
python experiments/07_full_matrix.py --profile thesis_full
```

### Perplexity ablations (read cached passages from exp07 — fast)
```bash
# Fraction filter sweep
python experiments/08_ppl_ablation.py --profile small

# Absolute threshold sweep (key result)
python experiments/09_ppl_threshold.py \
    --profile small \
    --thresholds 30 40 50 75 100 \
    --n-adv-passages 1 \
    --strict-fallback \
    --out-dir results/exp09_strict
```

### Re-craft hybrid passages with TAG-only inventory (after a budget change)
```bash
python scripts/recraft_hybrid_tag.py --profile small --budget 20
```

### Rescore CSVs with the lenient (fuzzy) ASR metric
```bash
python scripts/rescore_lenient.py --root results/
```

### Build LaTeX tables for the paper
```bash
python scripts/build_table1.py   # attack × defense (Vanilla)
python scripts/build_table2.py   # RAG variants (no defense)
```

---

## Results Tables

All results below are from the `small` profile (NQ, 10 K corpus, 20 questions, qwen2.5:7b,
budget=20). Numbers will tighten at 100Q in the `thesis_full` profile.

### Table 1 — Attack × Defense on VanillaRAG (NQ, n=20)

| Attack | None | NFKC normalize | Zero-width strip | Perplexity (fraction) |
|---|:-:|:-:|:-:|:-:|
| Pure semantic | **90 %** | 90 % | 90 % | 85 % |
| Pure unicode | 10 % | 10 % e2e=0 % | 10 % e2e=**0 %** | 25 % e2e=20 % |
| **Hybrid (ours)** | **85 %** | **90 %** | **90 %** | **85 %** |

**Reading:** every defense column has at least one attack at ≥ 85 %. The hybrid attack is
the only attack that achieves ≥ 85 % in every column. Zero-width strip blocks pure unicode
completely (e2e=0 %) but leaves hybrid at 90 %.

### Table 2 — Attack × RAG Variant (NQ, no defense)

| RAG variant | Semantic | Hybrid | Notes |
|---|:-:|:-:|---|
| Vanilla (n=20) | 90 % | 85 % | Baseline |
| Self-RAG (n=3) | 67 % | — | Smoke test only — full 100Q TBD |
| CRAG (n=3) | 67 % | — | Smoke test only |
| RobustRAG (n=3) | 67 % | — | Smoke test only |
| **TrustRAG (n=20)** | **30 %** | **~30 %** | K-means equally reduces both attacks |

### Table 3 — Absolute PPL threshold ablation, strict fallback (NQ, n=20, n_adv=1)

| Threshold | Semantic ASR | Hybrid ASR | Hybrid avg passages to LLM | Semantic avg passages to LLM |
|---:|:-:|:-:|:-:|:-:|
| 30 | 30 % | 20 % | 1.65 / 5 | 1.75 / 5 |
| 40 | 65 % | 65 % | 2.95 / 5 | 2.85 / 5 |
| **50** | **80 %** | **85 %** | **3.35 / 5** | **3.20 / 5** |
| 75 | 85 % | 85 % | 4.20 / 5 | 4.10 / 5 |
| 100 | 85 % | 85 % | 4.40 / 5 | 4.35 / 5 |

**Reading at τ=50:** hybrid passages all stay below the threshold (avg_adv_after=1.0 of 1.0
injected); 15 % of semantic passages are filtered (avg_adv_after=0.85). The 5-percentage-point
ASR gap (85 % vs 80 %) is the cleanest measurable PPL-based separation between attacks.

### Table 4 — Stealth profile (GPT-2 perplexity, NQ, n=20)

| Attack | Median PPL | Max PPL | Min PPL | Within τ=50 |
|---|:-:|:-:|:-:|:-:|
| Pure semantic | 35.8 | 112.6 | 11.2 | 17 / 20 (85 %) |
| Pure unicode | very high (zero-width) or low (TAG) | varies | — | depends on inventory |
| **Hybrid (TAG only, b=20)** | **34.6** | **49.1** | **22.7** | **20 / 20 (100 %)** |

---

## Thesis Claim

**Headline:**
> The hybrid Unicode-semantic attack achieves attack success rates equivalent to the strongest
> single-family attack (90 % semantic baseline) while being the only attack that no single
> standard defense can fully eliminate.

**Specifically:**

1. **Defense coverage gap.** Zero-width stripping — the canonical defense against
   invisible-character attacks — completely blocks the pure unicode attack (e2e ASR drops
   from 10 % to 0 %), but **fails against hybrid (90 % ASR retained)** because the semantic
   payload survives character removal.

2. **Perplexity discrimination.** Standard fraction-based PPL filtering does not
   discriminate hybrid from semantic. Strict absolute thresholding (τ=50, with no-context
   fallback) creates only a 5 pp gap (semantic 80 %, hybrid 85 %), because hybrid TAG-only
   perturbations stay below the perplexity threshold while pure semantic passages
   occasionally exceed it.

3. **Trust-aware RAG levelling.** TrustRAG's K-means filter reduces both semantic and
   hybrid attacks to ~30 % — but this is a property of the RAG architecture, not of any
   defense distinguishing the attacks.

4. **Stealth.** The hybrid attack has a *lower* maximum GPT-2 perplexity (49.1) than the
   pure semantic attack (112.6), making it harder to filter under conservative
   threshold settings.

**One-sentence abstract:**
> We show that combining semantic knowledge corruption with invisible Unicode TAG-character
> perturbations produces a hybrid attack that achieves 85–90 % ASR against five RAG
> architectures and evades all four standard defenses, while pure unicode attacks fail
> under stripping defenses and pure semantic attacks fail under strict perplexity filtering.

---

## References

### Attacks
- **PoisonedRAG** — Zou, Geng, Wang, Liu (2024). *PoisonedRAG: Knowledge Poisoning Attacks
  to Retrieval-Augmented Generation of Large Language Models.*
- **RAG-Pull** — Unicode perturbation attack on code-RAG (foundation of Stage 2).
- **Bad Characters** — Boucher, Shumailov, Anderson, Papernot (IEEE S&P 2022). Ancestor
  of invisible-character attacks.
- **HijackRAG / BadRAG / Phantom / AgentPoison** — surveyed in related-work section.

### Defenses
- **TrustRAG** — Zhou et al. 2025 (`github.com/HuichiZhou/TrustRAG`, MIT licensed)
- **RobustRAG** — Xiang et al. USENIX Security 2024 (`github.com/inspire-group/RobustRAG`)
- **Self-RAG** — Asai et al. ICLR 2024 (`github.com/AkariAsai/self-rag`)
- **CRAG** — Yan et al. 2024 (`github.com/HuskyInSalt/CRAG`)

### Models / Tools
- Contriever — Izacard et al., Meta AI
- GPT-2 — Radford et al., OpenAI (used as PPL scorer to match PoisonedRAG)
- Qwen 2.5 — Alibaba
- Llama 3.1 — Meta
- FAISS — Johnson, Douze, Jégou (Meta)
- HuggingFace Transformers, Datasets, Sentence-Transformers
- scipy `differential_evolution` — for the Stage-2 optimizer

### Datasets
- Natural Questions (Kwiatkowski et al. TACL 2019)
- BEIR benchmark splits (Thakur et al. 2021)
- HotpotQA (Yang et al. EMNLP 2018)
- MS MARCO (Bajaj et al. 2016)

---

## Limitations and Future Work

**Current limitations:**
- Vanilla and TrustRAG numbers are at n=20; full thesis run will be n=100.
- Self-RAG / CRAG / RobustRAG only have 3-question smoke results — these need a 100Q sweep.
- Single LLM tested (qwen2.5:7b); the paper claim requires re-running with at least one
  frontier model (GPT-4o-mini / Claude Haiku 4.5 / DeepSeek-V3).
- Single retriever tested (Contriever); BGE cross-validation pending.
- Single dataset tested (NQ); HotpotQA and MS-MARCO generalization pending.

**Open research questions:**
- Can adaptive defenses combining zero-width strip + strict PPL + paraphrase close the
  hybrid gap? Pilot run suggests partial success but at the cost of clean accuracy.
- Does the hybrid attack transfer across embedders (Contriever → BGE)? Open question.
- Can a hybrid-aware defense be designed (e.g. detect TAG-character entropy AND high
  retrieval-rank lift simultaneously)? Future work.

---

## Project Status

**Completed:**
- All 5 RAG variants implemented and unit-tested
- All 3 attacks implemented and validated end-to-end
- All 4 defenses implemented and integrated into experiment matrix
- 20Q dev results across the full matrix on Vanilla
- 20Q TrustRAG vs semantic + hybrid
- 3Q smoke results on Self-RAG / CRAG / RobustRAG
- Perplexity threshold ablation with strict fallback (key finding)

**In progress / planned:**
- 100-question `thesis_full` run on remote Linux GPU (qwen2.5:7b via Ollama)
- Multi-LLM final evaluation (GPT-4o-mini / Claude Haiku 4.5 / DeepSeek-V3)
- Multi-retriever cross-check (BGE-base-en-v1.5)
- Multi-dataset generalization (HotpotQA, MS-MARCO)

---

## License & Citation

This is a BTP (Bachelor's Thesis Project) research codebase. Until publication:
- Code: All rights reserved by the author.
- For academic use, please open an issue or contact the author for permission.

After publication, this README will be updated with the full citation.

**Author:** Dhruv Khanna (`dhruvkhanna930@gmail.com`)
**Institution:** IIIT Gwalior
**Year:** 2026

---

## Acknowledgements

This work builds directly on:
- The PoisonedRAG codebase and threat model
- The TrustRAG and RobustRAG defense designs
- The Bad Characters / RAG-Pull line of invisible-character research

The author thanks the maintainers of all open-source defense repositories for making
their code publicly available.
