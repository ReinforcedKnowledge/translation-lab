import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

EXPECTED_VLLM_VERSION = "0.22.1"
EXPECTED_XGRAMMAR_VERSION = "0.2.1"
EXPECTED_ORIGINAL_SHA256 = "e78109e76ea84a4464f5206e5ee4f26679367394ce6f31ef5304ce87fa785f21"
BACKEND_PATH = Path("vllm/v1/structured_output/backend_xgrammar.py")
ORIGINAL_BLOCK = """\
        else:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer,
                vocab_size=self.vocab_size,
            )
"""
REPAIRED_BLOCK = """\
        else:
            generation_config = (
                self.vllm_config.model_config.try_get_generation_config()
            )
            stop_token_ids = generation_config.get("eos_token_id")
            if stop_token_ids is None:
                stop_token_ids = self.tokenizer.eos_token_id
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer,
                vocab_size=self.vocab_size,
                stop_token_ids=stop_token_ids,
            )
"""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _distribution_version(site_packages: Path, name: str) -> str:
    matches = list(site_packages.glob(f"{name}-*.dist-info/METADATA"))
    if len(matches) != 1:
        raise ValueError(f"expected one {name} distribution in {site_packages}")
    for line in matches[0].read_text().splitlines():
        if line.startswith("Version: "):
            return line.removeprefix("Version: ")
    raise ValueError(f"{matches[0]} has no Version field")


def repair_source(source: str) -> str:
    if source.count(ORIGINAL_BLOCK) != 1:
        raise ValueError("expected one unpatched XGrammar tokenizer block")
    if REPAIRED_BLOCK in source:
        raise ValueError("source contains both original and repaired blocks")
    return source.replace(ORIGINAL_BLOCK, REPAIRED_BLOCK)


def patch_vllm_0221(site_packages: Path) -> dict[str, Any]:
    path = site_packages / BACKEND_PATH
    if not path.is_file():
        raise FileNotFoundError(path)
    original = path.read_bytes()
    backup = path.with_suffix(path.suffix + ".translation-lab.original")
    source = original.decode()
    if REPAIRED_BLOCK in source and ORIGINAL_BLOCK not in source:
        return {"status": "already_applied", "backend": str(path)}
    if _sha256(original) != EXPECTED_ORIGINAL_SHA256:
        raise ValueError("refusing to patch an unrecognized vLLM backend")
    if backup.exists():
        raise FileExistsError(backup)
    repaired = repair_source(source).encode()
    shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_bytes(repaired)
    os.chmod(temporary, path.stat().st_mode)
    os.replace(temporary, path)
    return {
        "status": "applied",
        "backend": str(path),
        "backup": str(backup),
        "original_sha256": _sha256(original),
        "repaired_sha256": _sha256(repaired),
    }


def validate_stop_fix(site_packages: Path, model_path: Path) -> dict[str, Any]:
    vllm_version = _distribution_version(site_packages, "vllm")
    xgrammar_version = _distribution_version(site_packages, "xgrammar")
    if vllm_version != EXPECTED_VLLM_VERSION:
        raise ValueError(f"expected vLLM {EXPECTED_VLLM_VERSION}, found {vllm_version}")
    if xgrammar_version != EXPECTED_XGRAMMAR_VERSION:
        raise ValueError(f"expected XGrammar {EXPECTED_XGRAMMAR_VERSION}, found {xgrammar_version}")
    path = site_packages / BACKEND_PATH
    source = path.read_text()
    if REPAIRED_BLOCK not in source or ORIGINAL_BLOCK in source:
        raise ValueError("the XGrammar stop-token repair is not active")
    generation = json.loads((model_path / "generation_config.json").read_text())
    raw_ids = generation.get("eos_token_id")
    stop_ids = [raw_ids] if isinstance(raw_ids, int) else raw_ids
    if sorted(set(stop_ids or [])) != [1, 50, 106]:
        raise ValueError("Gemma 4 generation_config.json does not declare [1, 50, 106]")
    return {
        "valid": True,
        "vllm_version": vllm_version,
        "xgrammar_version": xgrammar_version,
        "backend": str(path),
        "backend_sha256": _sha256(path.read_bytes()),
        "declared_stop_token_ids": [1, 50, 106],
    }
