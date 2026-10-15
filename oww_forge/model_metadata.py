"""Publish trained ONNX models with the identity supplied by their config."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile


def training_metadata(config: dict, version: str = "") -> dict[str, str]:
    if not isinstance(config, dict):
        raise ValueError("training config must be a mapping")
    phrases = config.get("target_phrase")
    if (
        not isinstance(phrases, list)
        or not phrases
        or any(not isinstance(p, str) or not p.strip() for p in phrases)
    ):
        raise ValueError("target_phrase must be a nonempty list of phrases")
    language = config.get("language", "en")
    if not isinstance(language, str) or not re.fullmatch(
        r"[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*", language
    ):
        raise ValueError("language must be a language code such as en or de-DE")
    result = {
        "wake_word": phrases[0].strip(),
        "target_phrases": json.dumps([p.strip() for p in phrases], ensure_ascii=False),
        "language": language.replace("_", "-"),
        "trained_by": "oww_forge",
        "training_date": datetime.now(timezone.utc).isoformat(),
    }
    if version:
        result["oww_forge_version"] = version
    threshold = config.get("recommended_threshold")
    if threshold is not None:
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError(
                "recommended_threshold must be a finite number from 0 to 1"
            )
        result["recommended_threshold"] = str(threshold)
    return result


def publish_model(source: Path, destination: Path, config: dict) -> None:
    # Keep ONNX out of forge's other CLI operations and pure config tests.
    import onnx

    metadata = training_metadata(config, os.environ.get("FORGE_VERSION", ""))
    model = onnx.load(source)
    existing = {entry.key: entry.value for entry in model.metadata_props}
    existing.update(metadata)
    onnx.helper.set_model_props(model, existing)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".model-", suffix=".onnx", dir=destination.parent
    )
    os.close(fd)
    try:
        onnx.save(model, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
