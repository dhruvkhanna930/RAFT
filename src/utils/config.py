"""Typed configuration loader for the unicode-rag-poison project.

Reads the four YAML files under ``configs/`` and parses them into a tree of
typed dataclasses rooted at :class:`ProjectCfg`.

Usage::

    from src.utils.config import load_project_config
    cfg = load_project_config("configs")

    # Scale control — the only place that knows about corpus size:
    print(cfg.datasets["nq"].corpus_size)   # -1 (full) or a positive cap
    print(cfg.datasets["nq"].n_questions)   # -1 (all)  or a positive cap

    # Convenience properties:
    print(cfg.models.default_embedder.hf_id)   # "facebook/contriever"
    print(cfg.models.default_ollama.id)         # "llama3.1:8b"

Design rules
------------
- ``corpus_size`` and ``n_questions`` live **only** in :class:`DatasetCfg`.
  No other dataclass carries scale information.
- All fields have explicit defaults so callers can build partial configs in
  tests without touching the filesystem.
- Parsers use ``.get()`` everywhere; missing YAML keys fall back to the
  dataclass defaults — no KeyError surprises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.io import load_yaml


# ── Dataset ───────────────────────────────────────────────────────────────────


@dataclass
class DatasetCfg:
    """Per-dataset loading configuration.

    Args:
        name: Dataset key used throughout the codebase (e.g. ``"nq"``).
        hf_id: HuggingFace ``datasets`` identifier passed to
            ``load_dataset(hf_id, ...)``.
        beir_id: BEIR subset identifier for retrieval benchmarks.
        total_passages: Informational only — approximate full corpus size.
            Never used to slice data; only for logging/documentation.
        corpus_size: How many passages to load.
            ``-1`` means load the entire corpus (full run).
            Any positive integer caps loading at that value (dev run).
        n_questions: How many questions to evaluate.
            ``-1`` means evaluate all questions in the split (full run).
            Any positive integer caps the question count (dev run).
        question_field: Name of the question field in the raw dataset.
        answer_field: Name of the answer field in the raw dataset.
        hf_config: Optional HuggingFace dataset config name
            (e.g. ``"distractor"`` for HotpotQA).
        splits: Dataset splits to load.
    """

    name: str
    hf_id: str
    beir_id: str
    total_passages: int = 0
    corpus_size: int = -1
    n_questions: int = -1
    question_field: str = "question"
    answer_field: str = "answer"
    hf_config: str | None = None
    splits: list[str] = field(default_factory=lambda: ["train", "validation"])

    @property
    def is_full_corpus(self) -> bool:
        """True when ``corpus_size == -1`` (no cap applied)."""
        return self.corpus_size == -1

    @property
    def is_full_eval(self) -> bool:
        """True when ``n_questions == -1`` (all questions evaluated)."""
        return self.n_questions == -1

    def effective_corpus_size(self) -> int | None:
        """Return the passage cap, or ``None`` if loading the full corpus."""
        return None if self.is_full_corpus else self.corpus_size

    def effective_n_questions(self) -> int | None:
        """Return the question cap, or ``None`` if evaluating all questions."""
        return None if self.is_full_eval else self.n_questions


# ── Models ────────────────────────────────────────────────────────────────────


@dataclass
class OllamaModelCfg:
    """Single Ollama model entry.

    Args:
        key: YAML key (e.g. ``"llama3_8b"``).
        id: Ollama model tag (e.g. ``"llama3.1:8b"``).
        context_length: Maximum context window in tokens.
        default: Whether this model is the default dev model.
        note: Optional human-readable note.
    """

    key: str
    id: str
    context_length: int
    default: bool = False
    note: str = ""


@dataclass
class EmbedderCfg:
    """Single sentence-transformers embedder entry.

    Args:
        key: YAML key (e.g. ``"contriever"``).
        hf_id: HuggingFace model ID.
        dim: Embedding dimension.
        default: Whether this embedder is the project default.
    """

    key: str
    hf_id: str
    dim: int
    default: bool = False


@dataclass
class ApiModelCfg:
    """External API LLM configuration.

    Args:
        model: Model ID string passed to the API.
        api_key_env: Name of the environment variable holding the API key.
        base_url: Optional base URL override (used for DeepSeek).
    """

    model: str
    api_key_env: str
    base_url: str = ""


@dataclass
class ModelsCfg:
    """All model configuration in one place.

    Args:
        ollama_base_url: Ollama server URL.
        ollama_models: Keyed dict of available Ollama models.
        embedders: Keyed dict of available sentence-transformers embedders.
        openai: OpenAI API config (``None`` if not configured).
        anthropic: Anthropic API config (``None`` if not configured).
        deepseek: DeepSeek API config (``None`` if not configured).
        ppl_scorer_model: HuggingFace model used for perplexity scoring.
        ppl_scorer_device: Device for the perplexity scorer.
    """

    ollama_base_url: str = "http://localhost:11434"
    ollama_models: dict[str, OllamaModelCfg] = field(default_factory=dict)
    embedders: dict[str, EmbedderCfg] = field(default_factory=dict)
    openai: ApiModelCfg | None = None
    anthropic: ApiModelCfg | None = None
    deepseek: ApiModelCfg | None = None
    ppl_scorer_model: str = "gpt2"
    ppl_scorer_device: str = "cpu"

    @property
    def default_ollama(self) -> OllamaModelCfg:
        """Return the model marked ``default: true``, or the first model."""
        for m in self.ollama_models.values():
            if m.default:
                return m
        return next(iter(self.ollama_models.values()))

    @property
    def default_embedder(self) -> EmbedderCfg:
        """Return the embedder marked ``default: true``, or the first one."""
        for e in self.embedders.values():
            if e.default:
                return e
        return next(iter(self.embedders.values()))


# ── Attacks ───────────────────────────────────────────────────────────────────


@dataclass
class PoisonedRagCfg:
    """PoisonedRAG black-box optimisation settings.

    Args:
        enabled: Whether to include this attack in experiment runs.
        num_adv_passages: Passages crafted per target question.
        num_iterations: Optimisation loop iterations per passage.
        retrieval_top_k: Retrieval condition cutoff (passage must rank ≤ k).
        target_answer_position: Where the target answer should appear
            in the optimised passage.
    """

    enabled: bool = True
    num_adv_passages: int = 5
    num_iterations: int = 50
    retrieval_top_k: int = 5
    target_answer_position: str = "first_sentence"


@dataclass
class RagPullCfg:
    """RAG-Pull Unicode perturbation settings.

    Args:
        enabled: Whether to include this attack in experiment runs.
        perturbation_budget: Maximum invisible characters to insert.
        char_categories: List of ``CharCategory`` string names to use.
        optimizer: Optimiser type (``"differential_evolution"``).
        de_population: DE population size.
        de_max_iter: Maximum DE generations.
        de_mutation: DE mutation factor F.
        de_crossover: DE crossover probability CR.
        insertion_strategy: Where to insert characters in the passage text.
    """

    enabled: bool = True
    perturbation_budget: int = 50
    char_categories: list[str] = field(default_factory=list)
    optimizer: str = "differential_evolution"
    de_population: int = 20
    de_max_iter: int = 100
    de_mutation: float = 0.8
    de_crossover: float = 0.9
    insertion_strategy: str = "whitespace"


@dataclass
class HybridCfg:
    """Hybrid attack settings (semantic injection + Unicode trigger).

    Args:
        enabled: Whether to include this attack in experiment runs.
        num_adv_passages: Stage-1 passage count.
        num_iterations: Stage-1 optimisation iterations.
        perturbation_budget: Stage-2 invisible character budget.
        char_categories: Stage-2 character categories.
        optimizer: Stage-2 optimiser type.
        de_population: Stage-2 DE population size.
        de_max_iter: Stage-2 maximum DE generations.
        trigger_location: Where in the passage to concentrate Unicode chars.
    """

    enabled: bool = True
    num_adv_passages: int = 5
    num_iterations: int = 50
    perturbation_budget: int = 30
    char_categories: list[str] = field(default_factory=list)
    optimizer: str = "differential_evolution"
    de_population: int = 20
    de_max_iter: int = 100
    trigger_location: str = "prefix"


@dataclass
class AttacksCfg:
    """Top-level attacks configuration.

    Args:
        injection_budget: Adversarial passages injected per question.
        poisoned_rag: PoisonedRAG hyper-parameters.
        rag_pull: RAG-Pull hyper-parameters.
        hybrid: Hybrid attack hyper-parameters.
    """

    injection_budget: int = 5
    poisoned_rag: PoisonedRagCfg = field(default_factory=PoisonedRagCfg)
    rag_pull: RagPullCfg = field(default_factory=RagPullCfg)
    hybrid: HybridCfg = field(default_factory=HybridCfg)


# ── RAG variants ──────────────────────────────────────────────────────────────


@dataclass
class RagVariantCfg:
    """Configuration for a single RAG variant.

    Args:
        name: Variant key (e.g. ``"vanilla"``, ``"trust_rag"``).
        enabled: Whether to include this variant in experiment runs.
        retriever: Embedder key from ``models.yaml`` (``"contriever"``).
        top_k: Default retrieval cutoff for this variant.
        extra: Any variant-specific keys forwarded verbatim from the YAML
            (e.g. ``kmeans_clusters``, ``use_vllm``).
    """

    name: str
    enabled: bool = True
    retriever: str = "contriever"
    top_k: int = 5
    extra: dict[str, Any] = field(default_factory=dict)


# ── Top-level ─────────────────────────────────────────────────────────────────


@dataclass
class ProjectCfg:
    """Full project configuration tree.

    Args:
        datasets: Keyed dict of :class:`DatasetCfg` (one per dataset).
        models: All model/embedder configuration.
        attacks: All attack hyper-parameters.
        rag_variants: Keyed dict of :class:`RagVariantCfg`.
        data_raw: Path to raw downloaded data (relative to repo root).
        data_processed: Path to processed corpora.
        data_poisoned: Path to poisoned corpora.
    """

    datasets: dict[str, DatasetCfg] = field(default_factory=dict)
    models: ModelsCfg = field(default_factory=ModelsCfg)
    attacks: AttacksCfg = field(default_factory=AttacksCfg)
    rag_variants: dict[str, RagVariantCfg] = field(default_factory=dict)
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    data_poisoned: str = "data/poisoned"

    @property
    def enabled_rag_variants(self) -> dict[str, RagVariantCfg]:
        """Return only the variants that have ``enabled: true``."""
        return {k: v for k, v in self.rag_variants.items() if v.enabled}

    @property
    def enabled_attacks(self) -> list[str]:
        """Return names of attacks that have ``enabled: true``."""
        names = []
        if self.attacks.poisoned_rag.enabled:
            names.append("poisoned_rag")
        if self.attacks.rag_pull.enabled:
            names.append("rag_pull")
        if self.attacks.hybrid.enabled:
            names.append("hybrid")
        return names


# ── Parsers ───────────────────────────────────────────────────────────────────

_RAG_VARIANT_NAMES = ("vanilla", "self_rag", "crag", "trust_rag", "robust_rag")
_RAG_BASE_KEYS = frozenset({"enabled", "retriever", "top_k", "upstream_repo"})


def _parse_datasets(raw: dict[str, Any]) -> tuple[dict[str, DatasetCfg], dict[str, str]]:
    datasets: dict[str, DatasetCfg] = {}
    for name, d in raw.get("datasets", {}).items():
        datasets[name] = DatasetCfg(
            name=name,
            hf_id=d["hf_id"],
            beir_id=d["beir_id"],
            total_passages=int(d.get("total_passages", 0)),
            corpus_size=int(d.get("corpus_size", -1)),
            n_questions=int(d.get("n_questions", -1)),
            question_field=d.get("question_field", "question"),
            answer_field=d.get("answer_field", "answer"),
            hf_config=d.get("hf_config"),
            splits=list(d.get("splits", ["train", "validation"])),
        )
    paths: dict[str, str] = raw.get("paths", {})
    return datasets, paths


def _parse_models(raw: dict[str, Any]) -> ModelsCfg:
    ollama_raw = raw.get("ollama", {})
    ollama_models: dict[str, OllamaModelCfg] = {}
    for key, m in ollama_raw.get("models", {}).items():
        ollama_models[key] = OllamaModelCfg(
            key=key,
            id=m["id"],
            context_length=int(m["context_length"]),
            default=bool(m.get("default", False)),
            note=str(m.get("note", "")),
        )

    embedders: dict[str, EmbedderCfg] = {}
    for key, e in raw.get("embedders", {}).items():
        embedders[key] = EmbedderCfg(
            key=key,
            hf_id=e["hf_id"],
            dim=int(e["dim"]),
            default=bool(e.get("default", False)),
        )

    openai = anthropic = deepseek = None
    if "openai" in raw:
        o = raw["openai"]
        openai = ApiModelCfg(model=o["model"], api_key_env=o["api_key_env"])
    if "anthropic" in raw:
        a = raw["anthropic"]
        anthropic = ApiModelCfg(model=a["model"], api_key_env=a["api_key_env"])
    if "deepseek" in raw:
        ds = raw["deepseek"]
        deepseek = ApiModelCfg(
            model=ds["model"],
            api_key_env=ds["api_key_env"],
            base_url=ds.get("base_url", ""),
        )

    ppl = raw.get("perplexity_scorer", {})
    return ModelsCfg(
        ollama_base_url=ollama_raw.get("base_url", "http://localhost:11434"),
        ollama_models=ollama_models,
        embedders=embedders,
        openai=openai,
        anthropic=anthropic,
        deepseek=deepseek,
        ppl_scorer_model=ppl.get("model", "gpt2"),
        ppl_scorer_device=ppl.get("device", "cpu"),
    )


def _parse_attacks(raw: dict[str, Any]) -> AttacksCfg:
    pr = raw.get("poisoned_rag", {})
    rp = raw.get("rag_pull", {})
    hy = raw.get("hybrid", {})
    return AttacksCfg(
        injection_budget=int(raw.get("injection_budget", 5)),
        poisoned_rag=PoisonedRagCfg(
            enabled=bool(pr.get("enabled", True)),
            num_adv_passages=int(pr.get("num_adv_passages", 5)),
            num_iterations=int(pr.get("num_iterations", 50)),
            retrieval_top_k=int(pr.get("retrieval_top_k", 5)),
            target_answer_position=str(pr.get("target_answer_position", "first_sentence")),
        ),
        rag_pull=RagPullCfg(
            enabled=bool(rp.get("enabled", True)),
            perturbation_budget=int(rp.get("perturbation_budget", 50)),
            char_categories=list(rp.get("char_categories", [])),
            optimizer=str(rp.get("optimizer", "differential_evolution")),
            de_population=int(rp.get("de_population", 20)),
            de_max_iter=int(rp.get("de_max_iter", 100)),
            de_mutation=float(rp.get("de_mutation", 0.8)),
            de_crossover=float(rp.get("de_crossover", 0.9)),
            insertion_strategy=str(rp.get("insertion_strategy", "whitespace")),
        ),
        hybrid=HybridCfg(
            enabled=bool(hy.get("enabled", True)),
            num_adv_passages=int(hy.get("num_adv_passages", 5)),
            num_iterations=int(hy.get("num_iterations", 50)),
            perturbation_budget=int(hy.get("perturbation_budget", 30)),
            char_categories=list(hy.get("char_categories", [])),
            optimizer=str(hy.get("optimizer", "differential_evolution")),
            de_population=int(hy.get("de_population", 20)),
            de_max_iter=int(hy.get("de_max_iter", 100)),
            trigger_location=str(hy.get("trigger_location", "prefix")),
        ),
    )


def _parse_rag_variants(raw: dict[str, Any]) -> dict[str, RagVariantCfg]:
    defaults = raw.get("defaults", {})
    result: dict[str, RagVariantCfg] = {}
    for name in _RAG_VARIANT_NAMES:
        if name not in raw:
            continue
        d = raw[name]
        extra = {k: v for k, v in d.items() if k not in _RAG_BASE_KEYS}
        result[name] = RagVariantCfg(
            name=name,
            enabled=bool(d.get("enabled", True)),
            retriever=str(d.get("retriever", defaults.get("retriever", "contriever"))),
            top_k=int(d.get("top_k", defaults.get("top_k", 5))),
            extra=extra,
        )
    return result


# ── Public entry point ────────────────────────────────────────────────────────


def load_project_config(config_dir: str | Path = "configs") -> ProjectCfg:
    """Load all four YAML files from *config_dir* into a :class:`ProjectCfg`.

    Args:
        config_dir: Directory containing ``datasets.yaml``, ``models.yaml``,
            ``attacks.yaml``, and ``rag_variants.yaml``.  Accepts both
            absolute and relative paths.

    Returns:
        Fully parsed and typed :class:`ProjectCfg` instance.

    Raises:
        FileNotFoundError: If any of the four YAML files is missing.
    """
    d = Path(config_dir)
    raw_ds = load_yaml(d / "datasets.yaml")
    raw_mo = load_yaml(d / "models.yaml")
    raw_at = load_yaml(d / "attacks.yaml")
    raw_rv = load_yaml(d / "rag_variants.yaml")

    datasets, paths = _parse_datasets(raw_ds)
    return ProjectCfg(
        datasets=datasets,
        models=_parse_models(raw_mo),
        attacks=_parse_attacks(raw_at),
        rag_variants=_parse_rag_variants(raw_rv),
        data_raw=paths.get("raw", "data/raw"),
        data_processed=paths.get("processed", "data/processed"),
        data_poisoned=paths.get("poisoned", "data/poisoned"),
    )
