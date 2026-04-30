"""Tests for src/utils/ — seed, logging, and I/O helpers.

No ML models required; all tests run instantly.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import pytest

from src.utils.io import ensure_dir, load_json, load_yaml, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed


# ── set_seed ──────────────────────────────────────────────────────────────────

class TestSetSeed:
    def test_python_random_reproducible(self) -> None:
        set_seed(42)
        a = [random.random() for _ in range(5)]
        set_seed(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_numpy_random_reproducible(self) -> None:
        import numpy as np

        set_seed(0)
        a = np.random.rand(5).tolist()
        set_seed(0)
        b = np.random.rand(5).tolist()
        assert a == pytest.approx(b)

    def test_different_seeds_differ(self) -> None:
        set_seed(1)
        a = random.random()
        set_seed(2)
        b = random.random()
        assert a != b

    def test_set_seed_no_crash_without_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """set_seed should not raise even if torch is not importable."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "torch":
                raise ImportError("mocked torch absence")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        set_seed(42)  # should not raise


# ── get_logger ────────────────────────────────────────────────────────────────

class TestGetLogger:
    def test_returns_logger(self) -> None:
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)

    def test_name_set(self) -> None:
        logger = get_logger("my.module")
        assert logger.name == "my.module"

    def test_default_level_info(self) -> None:
        logger = get_logger("info_test")
        assert logger.level == logging.INFO

    def test_custom_level(self) -> None:
        logger = get_logger("debug_test", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_no_duplicate_handlers(self) -> None:
        # Calling twice should not add a second handler
        name = "dedup_test_logger"
        l1 = get_logger(name)
        l2 = get_logger(name)
        assert len(l2.handlers) == 1


# ── I/O helpers ───────────────────────────────────────────────────────────────

class TestIOHelpers:
    def test_save_and_load_json(self, tmp_path: Path) -> None:
        data = {"key": "value", "num": 42, "list": [1, 2, 3]}
        path = tmp_path / "data.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_save_json_pretty_printed(self, tmp_path: Path) -> None:
        path = tmp_path / "pretty.json"
        save_json({"a": 1}, path, indent=4)
        raw = path.read_text()
        assert "\n" in raw  # pretty-printed has newlines

    def test_load_json_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nonexistent.json")

    def test_load_yaml_basic(self, tmp_path: Path) -> None:
        yaml_content = "name: test\nvalue: 42\nlist:\n  - a\n  - b\n"
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)
        data = load_yaml(path)
        assert data["name"] == "test"
        assert data["value"] == 42
        assert data["list"] == ["a", "b"]

    def test_load_yaml_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_yaml(tmp_path / "missing.yaml")

    def test_ensure_dir_creates_directory(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "a" / "b" / "c"
        result = ensure_dir(new_dir)
        assert result.is_dir()

    def test_ensure_dir_returns_path(self, tmp_path: Path) -> None:
        result = ensure_dir(tmp_path / "new")
        assert isinstance(result, Path)

    def test_ensure_dir_idempotent(self, tmp_path: Path) -> None:
        d = tmp_path / "exists"
        ensure_dir(d)
        ensure_dir(d)  # should not raise
        assert d.is_dir()

    def test_json_unicode_preserved(self, tmp_path: Path) -> None:
        # Invisible chars must survive a round-trip
        text = "hello\u200bworld"
        path = tmp_path / "unicode.json"
        save_json({"text": text}, path)
        loaded = load_json(path)
        assert loaded["text"] == text
