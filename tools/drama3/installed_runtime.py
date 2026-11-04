# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch and inspect regular Python package installations without loading models."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

_BASE_PP_BLOB = "976caad4cbe430ec91873268ee482e5b8dc9a264"
_PP_PATH = Path("v1/worker/gpu/pp_utils.py")


def package_dir(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise SystemExit(f"{name} is not installed in the selected Python environment")
    distribution = importlib.metadata.distribution(name.replace("_", "-"))
    installed = Path(distribution.locate_file(f"{name}/__init__.py")).resolve()
    if not installed.is_file() or Path(spec.origin).resolve() != installed:
        raise SystemExit(
            f"{name} imports from {spec.origin} instead of its regular installation. "
            "Remove source-directory overrides from PYTHONPATH and use a regular "
            "package installation in the selected environment."
        )
    return installed.parent


def blob_id(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode() + b"\0" + data, usedforsecurity=False
    ).hexdigest()


def core_files(repo: Path) -> tuple[Path, bytes, bytes]:
    source = (repo / "vllm" / _PP_PATH).read_bytes()
    target = package_dir("vllm") / _PP_PATH
    if not target.is_file():
        raise SystemExit(f"Installed vLLM lacks the supported V2 PP helper: {target}")
    current = target.read_bytes()
    if current != source and blob_id(current) != _BASE_PP_BLOB:
        raise SystemExit(
            f"Refusing to replace an unfamiliar PP helper: {target}. "
            "Install the matching fork build or review its differences first."
        )
    return target, current, source


def check_companion(repo: Path) -> None:
    installed = package_dir("vllm_omni")
    checkout = repo / ".cache/drama3/vllm-omni-pp"
    patch = repo / "tools/drama3/vllm-omni-pp.patch"
    for line in patch.read_text(encoding="utf-8").splitlines():
        if not line.startswith("+++ b/vllm_omni/"):
            continue
        relative = PurePosixPath(line.removeprefix("+++ b/vllm_omni/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"Invalid companion patch path: {relative}")
        expected = checkout / "vllm_omni" / relative
        actual = installed / relative
        if not actual.is_file() or actual.read_bytes() != expected.read_bytes():
            raise SystemExit(
                f"Installed companion differs at {actual}; run setup again "
                "to install the patched package."
            )


def write_constraints(path: Path) -> None:
    constraints = []
    for name in ("vllm", "torch", "torchaudio", "torchvision", "triton"):
        try:
            constraints.append(f"{name}=={importlib.metadata.version(name)}\n")
        except importlib.metadata.PackageNotFoundError:
            if name in ("vllm", "torch"):
                raise SystemExit(f"Missing required runtime package: {name}") from None
    constraints.append("transformers==5.14.1\n")
    path.write_text("".join(constraints), encoding="utf-8")


def apply_core(repo: Path, target: Path, current: bytes, source: bytes) -> None:
    if current == source:
        print(f"Installed core already patched: {target}")
        return
    backup = repo / ".cache/drama3/core-backups" / blob_id(current) / "pp_utils.py"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists() and backup.read_bytes() != current:
        raise SystemExit(f"Existing core backup differs: {backup}")
    backup.write_bytes(current)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(source)
        shutil.copymode(target, temporary)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Patched installed core: {target}")
    print(f"Original core backup: {backup}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "apply", "check"))
    parser.add_argument("repo", type=Path)
    parser.add_argument("--constraints", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    target, current, source = core_files(repo)
    if args.action == "prepare":
        if args.constraints is None:
            parser.error("prepare requires --constraints")
        write_constraints(args.constraints)
        return
    check_companion(repo)
    if args.action == "apply":
        apply_core(repo, target, current, source)
    elif current != source:
        raise SystemExit("The installed core is missing its PP patch; run setup again")


if __name__ == "__main__":
    main()
