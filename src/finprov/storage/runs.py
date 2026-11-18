"""Arquivos tratados por execução, sem snapshots de respostas ou código."""

import hashlib
import json
from pathlib import Path
from uuid import UUID


def canonical_json(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_path(root, run_id: str) -> Path:
    try:
        if str(UUID(run_id)) != run_id:
            raise ValueError
    except ValueError:
        raise ValueError("Identificador de execução inválido.") from None
    return Path(root) / "runs" / run_id
