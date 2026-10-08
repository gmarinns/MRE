"""Contrato comum a todas as funções do motor."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import numpy as np
from scipy import stats

# Namespace fixo: o mesmo cálculo, com os mesmos insumos, gera o mesmo run_id.
_RUN_NAMESPACE = uuid.UUID("6f1b0c9a-6a1e-5f6b-9d2a-1f7c0b3e4d55")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _stable(value):
    return round(value, 10) if isinstance(value, float) else value


@dataclass
class RiskResult:
    """O número de risco e tudo o que foi usado para chegar nele."""

    value: float  # perda potencial em fração (0.032 = 3,2%)
    method: str
    confidence_level: float
    params: dict = field(default_factory=dict)
    input_hash: str = ""
    horizon_days: int = 1
    computed_at: str = field(default_factory=_now)

    @property
    def run_id(self) -> str:
        """Identificador determinístico: insumo + método + parâmetros."""
        payload = "|".join(
            [
                self.method,
                self.input_hash,
                f"{self.confidence_level:.6f}",
                f"{self.horizon_days}",
                repr(sorted((k, _stable(v)) for k, v in self.params.items())),
            ]
        )
        return str(uuid.uuid5(_RUN_NAMESPACE, payload))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["run_id"] = self.run_id
        return data


def sha256_of_array(arr: np.ndarray) -> str:
    """Hash dos insumos: identifica de forma única a série usada no cálculo."""
    arr = np.ascontiguousarray(np.round(np.asarray(arr, dtype=float), 10))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def z_score(confidence_level: float) -> float:
    """Quantil da normal padrão na cauda esquerda (negativo)."""
    return float(stats.norm.ppf(1.0 - confidence_level))
