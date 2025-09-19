from __future__ import annotations

"""Загрузчик артефактов моделей (например, ONNX)."""

import os
from pathlib import Path

try:  # pragma: no cover - опциональная зависимость
    import onnxruntime as ort
except Exception:  # noqa: BLE001
    ort = None  # type: ignore[assignment]


class ArtifactNotAvailable(RuntimeError):
    """Поднимается, если артефакт недоступен или не может быть прочитан."""


def load_model_artifact(relative_path: str):
    """Загрузить ONNX-модель из каталога артефактов.

    Каталог задаётся переменной окружения ``MODEL_ARTIFACT_ROOT`` (по умолчанию
    ``artifacts`` в корне проекта). Возвращает ``onnxruntime.InferenceSession``.
    """

    if ort is None:
        raise ArtifactNotAvailable("onnxruntime не установлен")
    root = Path(os.getenv("MODEL_ARTIFACT_ROOT", "artifacts"))
    path = Path(relative_path)
    full_path = path if path.is_absolute() else root / path
    if not full_path.exists():
        raise FileNotFoundError(full_path)
    try:
        return ort.InferenceSession(full_path.as_posix(), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        raise ArtifactNotAvailable(str(exc)) from exc

