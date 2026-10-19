# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Detect stale runtime overlays and checkpoint identity before loading weights."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "tools/agentvue/check_deepseek_runtime.py"
spec = importlib.util.spec_from_file_location("runtime_audit", PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_runtime_audit_detects_stale_file_and_accepts_line_endings(tmp_path):
    source, installed = tmp_path / "source.py", tmp_path / "installed"
    installed.mkdir()
    source.write_bytes(b"value = 1\r\n")
    (installed / "source.py").write_bytes(b"value = 1\n")
    expected = {"source.py": audit.file_digest(source)}
    report = audit.audit_installation(expected, installed_root=installed)
    audit.validate_reports({"one": report})
    (installed / "source.py").write_bytes(b"value = 0\n")
    report = audit.audit_installation(expected, installed_root=installed)
    with pytest.raises(ValueError, match="source.py"):
        audit.validate_reports({"one": report})


def test_cluster_audit_rejects_tokenizer_mismatch(tmp_path):
    expected = {}
    report = audit.audit_installation(expected, installed_root=tmp_path)
    first = report | {"checkpoint": {"tokenizer.json": "one"}}
    second = report | {"checkpoint": {"tokenizer.json": "two"}}
    with pytest.raises(ValueError, match="checkpoint differ"):
        audit.validate_reports({"one": first, "two": second})


def test_runtime_install_restores_only_manifest_files(tmp_path, monkeypatch):
    checkout, installed = tmp_path / "checkout", tmp_path / "installed"
    (checkout / "vllm").mkdir(parents=True)
    installed.mkdir()
    (checkout / "vllm/source.py").write_text("value = 1\n")
    (installed / "source.py").write_text("value = 0\n")
    (installed / "native.so").write_bytes(b"untouched")
    monkeypatch.setattr(
        audit,
        "source_manifest",
        lambda root: {"source.py": audit.file_digest(checkout / "vllm/source.py")},
    )
    report = audit.install_runtime(checkout, installed_root=installed)
    assert report["mismatches"] == []
    assert (installed / "source.py").read_text() == "value = 1\n"
    assert (installed / "native.so").read_bytes() == b"untouched"
