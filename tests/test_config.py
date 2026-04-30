"""Tests for the config system and all three abstract base classes.

Two categories:
1. Config tests — load the real YAML files and verify typed dataclasses.
2. ABC tests — verify that (a) abstract classes cannot be instantiated
   directly, and (b) stub concrete subclasses raise NotImplementedError
   when their abstract methods are called.

No ML models are required; all tests run instantly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.attacks.base import AttackBase, AttackConfig, AdversarialPassage
from src.defenses.base import DefenseBase
from src.rag.base import GenerationResult, RagBase, RetrievalResult
from src.utils.config import (
    AttacksCfg,
    DatasetCfg,
    HybridCfg,
    ModelsCfg,
    ProjectCfg,
    PoisonedRagCfg,
    RagPullCfg,
    RagVariantCfg,
    load_project_config,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def _load() -> ProjectCfg:
    """Load the project config from the real YAML files."""
    return load_project_config(CONFIGS_DIR)


# ── Minimal concrete stubs (no logic — just satisfy the ABC) ──────────────────

class _StubRAG(RagBase):
    def retrieve(self, query: str, k: int) -> RetrievalResult:
        raise NotImplementedError

    def generate(self, query: str, retrieved: RetrievalResult) -> GenerationResult:
        raise NotImplementedError


class _StubAttack(AttackBase):
    def craft_malicious_passages(
        self, target_question: str, target_answer: str, n: int
    ) -> list[str]:
        raise NotImplementedError

    def inject(self, corpus: list[str], adversarial_passages: list[str]) -> list[str]:
        raise NotImplementedError


class _StubDefense(DefenseBase):
    def apply(self, text_or_passages: str | list[str]) -> str | list[str]:
        raise NotImplementedError


# ════════════════════════════════════════════════════════════════════════════
# 1. Abstract base classes
# ════════════════════════════════════════════════════════════════════════════


class TestRagBase:
    def test_cannot_instantiate_directly(self) -> None:
        """RagBase must not be instantiable — it has abstract methods."""
        with pytest.raises(TypeError):
            RagBase(retriever=None, llm=None)  # type: ignore[abstract]

    def test_retrieve_raises_not_implemented(self) -> None:
        rag = _StubRAG(retriever=None, llm=None)
        with pytest.raises(NotImplementedError):
            rag.retrieve("What is the capital of France?", k=5)

    def test_generate_raises_not_implemented(self) -> None:
        rag = _StubRAG(retriever=None, llm=None)
        result = RetrievalResult(passages=["p1"], scores=[0.9])
        with pytest.raises(NotImplementedError):
            rag.generate("question", result)

    def test_answer_chains_retrieve_then_generate(self) -> None:
        """answer() must call retrieve() first — so it propagates its error."""
        rag = _StubRAG(retriever=None, llm=None)
        with pytest.raises(NotImplementedError):
            rag.answer("question")

    def test_answer_uses_top_k_default(self) -> None:
        """answer() without explicit k uses self.top_k."""
        calls: list[int] = []

        class _Recorder(_StubRAG):
            def retrieve(self, query: str, k: int) -> RetrievalResult:  # type: ignore[override]
                calls.append(k)
                raise NotImplementedError

        rag = _Recorder(retriever=None, llm=None, top_k=7)
        with pytest.raises(NotImplementedError):
            rag.answer("q")
        assert calls == [7]

    def test_answer_respects_explicit_k(self) -> None:
        calls: list[int] = []

        class _Recorder(_StubRAG):
            def retrieve(self, query: str, k: int) -> RetrievalResult:  # type: ignore[override]
                calls.append(k)
                raise NotImplementedError

        rag = _Recorder(retriever=None, llm=None, top_k=5)
        with pytest.raises(NotImplementedError):
            rag.answer("q", k=3)
        assert calls == [3]

    def test_load_corpus_delegates_to_retriever(self) -> None:
        """load_corpus() must call retriever.build_index()."""
        calls: list[list[str]] = []

        class _FakeRetriever:
            def build_index(self, corpus: list[str]) -> None:
                calls.append(corpus)

        rag = _StubRAG(retriever=_FakeRetriever(), llm=None)
        rag.load_corpus(["p1", "p2"])
        assert calls == [["p1", "p2"]]

    def test_repr(self) -> None:
        rag = _StubRAG(retriever=None, llm=None, top_k=10)
        assert "top_k=10" in repr(rag)


class TestAttackBase:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            AttackBase(config=AttackConfig())  # type: ignore[abstract]

    def test_craft_raises_not_implemented(self) -> None:
        attack = _StubAttack(config=AttackConfig())
        with pytest.raises(NotImplementedError):
            attack.craft_malicious_passages("q", "a", n=5)

    def test_inject_raises_not_implemented(self) -> None:
        attack = _StubAttack(config=AttackConfig())
        with pytest.raises(NotImplementedError):
            attack.inject(["passage1"], ["adv1"])

    def test_repr(self) -> None:
        cfg = AttackConfig(injection_budget=3)
        attack = _StubAttack(config=cfg)
        assert "budget=3" in repr(attack)

    def test_attack_config_stored(self) -> None:
        cfg = AttackConfig(injection_budget=10)
        attack = _StubAttack(config=cfg)
        assert attack.config.injection_budget == 10


class TestDefenseBase:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            DefenseBase()  # type: ignore[abstract]

    def test_apply_raises_not_implemented(self) -> None:
        defense = _StubDefense()
        with pytest.raises(NotImplementedError):
            defense.apply(["passage1", "passage2"])

    def test_apply_single_string_raises_not_implemented(self) -> None:
        defense = _StubDefense()
        with pytest.raises(NotImplementedError):
            defense.apply("single passage")

    def test_callable_alias(self) -> None:
        """__call__ must delegate to apply."""
        defense = _StubDefense()
        with pytest.raises(NotImplementedError):
            defense("passage")


# ════════════════════════════════════════════════════════════════════════════
# 2. Config loading
# ════════════════════════════════════════════════════════════════════════════


class TestLoadProjectConfig:
    def test_returns_project_cfg(self) -> None:
        cfg = _load()
        assert isinstance(cfg, ProjectCfg)

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_project_config(tmp_path / "nonexistent")

    def test_datasets_loaded(self) -> None:
        cfg = _load()
        assert len(cfg.datasets) >= 3          # nq, hotpotqa, msmarco at minimum

    def test_dataset_keys_present(self) -> None:
        cfg = _load()
        assert "nq" in cfg.datasets
        assert "hotpotqa" in cfg.datasets
        assert "msmarco" in cfg.datasets


class TestDatasetCfg:
    def test_corpus_size_is_minus_one_by_default(self) -> None:
        """The demo-vs-full sentinel must default to -1 in every dataset."""
        cfg = _load()
        for name, ds in cfg.datasets.items():
            assert ds.corpus_size == -1, (
                f"{name}.corpus_size should be -1 (full) by default, got {ds.corpus_size}"
            )

    def test_n_questions_is_minus_one_by_default(self) -> None:
        cfg = _load()
        for name, ds in cfg.datasets.items():
            assert ds.n_questions == -1, (
                f"{name}.n_questions should be -1 (all) by default, got {ds.n_questions}"
            )

    def test_is_full_corpus_property(self) -> None:
        ds = DatasetCfg(name="x", hf_id="x", beir_id="x", corpus_size=-1)
        assert ds.is_full_corpus is True

    def test_is_not_full_corpus_when_capped(self) -> None:
        ds = DatasetCfg(name="x", hf_id="x", beir_id="x", corpus_size=10_000)
        assert ds.is_full_corpus is False

    def test_effective_corpus_size_none_when_full(self) -> None:
        ds = DatasetCfg(name="x", hf_id="x", beir_id="x", corpus_size=-1)
        assert ds.effective_corpus_size() is None

    def test_effective_corpus_size_returns_cap(self) -> None:
        ds = DatasetCfg(name="x", hf_id="x", beir_id="x", corpus_size=5000)
        assert ds.effective_corpus_size() == 5000

    def test_nq_hf_id(self) -> None:
        cfg = _load()
        assert cfg.datasets["nq"].hf_id == "nq_open"

    def test_nq_question_field(self) -> None:
        cfg = _load()
        assert cfg.datasets["nq"].question_field == "question"

    def test_hotpotqa_has_hf_config(self) -> None:
        cfg = _load()
        assert cfg.datasets["hotpotqa"].hf_config == "distractor"


class TestModelsCfg:
    def test_models_cfg_type(self) -> None:
        cfg = _load()
        assert isinstance(cfg.models, ModelsCfg)

    def test_ollama_base_url(self) -> None:
        cfg = _load()
        assert cfg.models.ollama_base_url == "http://localhost:11434"

    def test_ollama_models_nonempty(self) -> None:
        cfg = _load()
        assert len(cfg.models.ollama_models) >= 1

    def test_default_ollama_exists(self) -> None:
        cfg = _load()
        model = cfg.models.default_ollama
        assert model.id == "llama3.1:8b"

    def test_embedders_nonempty(self) -> None:
        cfg = _load()
        assert len(cfg.models.embedders) >= 1

    def test_default_embedder_is_contriever(self) -> None:
        cfg = _load()
        assert cfg.models.default_embedder.hf_id == "facebook/contriever"

    def test_default_embedder_dim(self) -> None:
        cfg = _load()
        assert cfg.models.default_embedder.dim == 768

    def test_openai_cfg_present(self) -> None:
        cfg = _load()
        assert cfg.models.openai is not None
        assert cfg.models.openai.model == "gpt-4o-mini"

    def test_anthropic_cfg_present(self) -> None:
        cfg = _load()
        assert cfg.models.anthropic is not None

    def test_deepseek_cfg_present(self) -> None:
        cfg = _load()
        assert cfg.models.deepseek is not None
        assert cfg.models.deepseek.base_url != ""

    def test_ppl_scorer_model(self) -> None:
        cfg = _load()
        assert cfg.models.ppl_scorer_model == "gpt2"


class TestAttacksCfg:
    def test_attacks_cfg_type(self) -> None:
        cfg = _load()
        assert isinstance(cfg.attacks, AttacksCfg)

    def test_injection_budget(self) -> None:
        cfg = _load()
        assert cfg.attacks.injection_budget == 5

    def test_poisoned_rag_enabled(self) -> None:
        cfg = _load()
        assert cfg.attacks.poisoned_rag.enabled is True

    def test_poisoned_rag_iterations(self) -> None:
        cfg = _load()
        assert cfg.attacks.poisoned_rag.num_iterations == 50

    def test_rag_pull_enabled(self) -> None:
        cfg = _load()
        assert cfg.attacks.rag_pull.enabled is True

    def test_rag_pull_has_char_categories(self) -> None:
        cfg = _load()
        assert len(cfg.attacks.rag_pull.char_categories) > 0

    def test_rag_pull_de_mutation_type(self) -> None:
        cfg = _load()
        assert isinstance(cfg.attacks.rag_pull.de_mutation, float)

    def test_hybrid_trigger_location(self) -> None:
        cfg = _load()
        assert cfg.attacks.hybrid.trigger_location == "prefix"

    def test_enabled_attacks_list(self) -> None:
        cfg = _load()
        enabled = cfg.enabled_attacks
        assert "poisoned_rag" in enabled
        assert "rag_pull" in enabled
        assert "hybrid" in enabled


class TestRagVariantsCfg:
    def test_five_variants_present(self) -> None:
        cfg = _load()
        assert set(cfg.rag_variants.keys()) == {
            "vanilla", "self_rag", "crag", "trust_rag", "robust_rag"
        }

    def test_all_variants_enabled(self) -> None:
        cfg = _load()
        for name, variant in cfg.rag_variants.items():
            assert variant.enabled, f"{name} should be enabled"

    def test_all_variants_use_contriever(self) -> None:
        cfg = _load()
        for name, variant in cfg.rag_variants.items():
            assert variant.retriever == "contriever", f"{name} retriever mismatch"

    def test_top_k_is_five(self) -> None:
        cfg = _load()
        for name, variant in cfg.rag_variants.items():
            assert variant.top_k == 5, f"{name}.top_k should be 5"

    def test_enabled_rag_variants_property(self) -> None:
        cfg = _load()
        assert len(cfg.enabled_rag_variants) == 5

    def test_trust_rag_extra_keys(self) -> None:
        cfg = _load()
        assert "kmeans_clusters" in cfg.rag_variants["trust_rag"].extra
        assert "trust_threshold" in cfg.rag_variants["trust_rag"].extra

    def test_self_rag_extra_hf_model(self) -> None:
        cfg = _load()
        assert "hf_model_id" in cfg.rag_variants["self_rag"].extra

    def test_robust_rag_extra_aggregation(self) -> None:
        cfg = _load()
        assert cfg.rag_variants["robust_rag"].extra["aggregation"] == "majority_vote"


class TestProjectCfgPaths:
    def test_data_raw_path(self) -> None:
        cfg = _load()
        assert cfg.data_raw == "data/raw"

    def test_data_processed_path(self) -> None:
        cfg = _load()
        assert cfg.data_processed == "data/processed"

    def test_data_poisoned_path(self) -> None:
        cfg = _load()
        assert cfg.data_poisoned == "data/poisoned"


class TestDatasetCfgDefaults:
    """Verify DatasetCfg can be constructed with only required fields
    and that all defaults are sane — without touching the filesystem."""

    def test_corpus_size_defaults_to_minus_one(self) -> None:
        ds = DatasetCfg(name="test", hf_id="test/id", beir_id="test")
        assert ds.corpus_size == -1

    def test_n_questions_defaults_to_minus_one(self) -> None:
        ds = DatasetCfg(name="test", hf_id="test/id", beir_id="test")
        assert ds.n_questions == -1

    def test_hf_config_defaults_none(self) -> None:
        ds = DatasetCfg(name="test", hf_id="test/id", beir_id="test")
        assert ds.hf_config is None

    def test_overriding_corpus_size_for_dev(self) -> None:
        """Simulate what an experiment script does to run a dev subset."""
        ds = DatasetCfg(
            name="nq", hf_id="nq_open", beir_id="nq",
            corpus_size=10_000,
            n_questions=100,
        )
        assert not ds.is_full_corpus
        assert not ds.is_full_eval
        assert ds.effective_corpus_size() == 10_000
        assert ds.effective_n_questions() == 100
