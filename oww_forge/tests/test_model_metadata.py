"""Real ONNX publication and inference, separate from lightweight controller CI."""

import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "oww_forge"), str(ROOT / "controller")]
from model_metadata import publish_model, training_metadata
import em_oww_metadata


def test_export_preserves_inference_and_overrides_filename(tmp_path):
    source, destination = tmp_path / "source.onnx", tmp_path / "wrong_filename.onnx"
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["x"], ["y"])],
        "test",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9
    )
    onnx.helper.set_model_props(model, {"unrelated": "preserved"})
    onnx.save(model, source)
    config = {
        "target_phrase": ["Hallo Clara", "Hallo Klara"],
        "language": "de",
        "recommended_threshold": 0.6,
    }
    publish_model(source, destination, config)
    onnx.checker.check_model(onnx.load(destination))
    for path in [source, destination]:
        result = ort.InferenceSession(str(path)).run(
            None, {"x": np.array([0.25], dtype=np.float32)}
        )
        np.testing.assert_array_equal(result[0], [0.25])
    values = {v.key: v.value for v in onnx.load(destination).metadata_props}
    assert values["unrelated"] == "preserved"
    assert json.loads(values["target_phrases"]) == config["target_phrase"]
    assert values["recommended_threshold"] == "0.6"
    assert values["trained_by"] == "oww_forge"
    assert "training_date" in values
    assert em_oww_metadata.resolve(str(destination)) == em_oww_metadata.WakeWord(
        "Hallo Clara", ("de",)
    )
    assert em_oww_metadata.resolve(str(source)).name == "source"
    config["target_phrase"] = ["Bonjour Clara"]
    config["language"] = "fr"
    publish_model(source, destination, config)
    assert em_oww_metadata.resolve(str(destination)) == em_oww_metadata.WakeWord(
        "Bonjour Clara", ("fr",)
    )


@pytest.mark.parametrize(
    "change",
    [
        {"target_phrase": []},
        {"target_phrase": [1]},
        {"target_phrase": "hello"},
        {"language": []},
        {"recommended_threshold": float("nan")},
        {"recommended_threshold": True},
        {"recommended_threshold": 2},
    ],
)
def test_invalid_training_metadata_is_rejected(change):
    with pytest.raises(ValueError):
        training_metadata({"target_phrase": ["hello"], **change})


def test_absent_optional_values_are_not_invented():
    values = training_metadata({"target_phrase": ["hello"]})
    assert values["language"] == "en"
    assert "recommended_threshold" not in values
    assert "oww_forge_version" not in values


def test_invalid_stamp_preserves_existing_published_file(tmp_path):
    source = tmp_path / "source.onnx"
    destination = tmp_path / "published.onnx"
    destination.write_bytes(b"previous published model")
    with pytest.raises(ValueError):
        publish_model(source, destination, {"target_phrase": []})
    assert destination.read_bytes() == b"previous published model"
    assert not list(tmp_path.glob(".model-*"))


def test_build_publishes_training_config_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import forge

    name = "filename"
    words = tmp_path / "wakewords"
    directory = words / name
    directory.mkdir(parents=True)
    (directory / "config.yml").write_text(
        'target_phrase: ["Bonjour Clara"]\nlanguage: fr\n'
    )
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["x"], ["y"])],
        "test",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9
    )

    def train(*args, **kwargs):
        onnx.save(model, directory / f"{name}.onnx")

    monkeypatch.setattr(forge, "WAKEWORDS", words)
    monkeypatch.setattr(forge, "MODELS", tmp_path / "models")
    monkeypatch.setattr(forge, "missing_assets", lambda: [])
    monkeypatch.setattr(forge.subprocess, "run", train)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="fixture", cuda=SimpleNamespace(is_available=lambda: False)
        ),
    )
    forge.cmd_build(
        SimpleNamespace(
            name=name, from_step="train", only_step="train", overwrite=False
        )
    )
    assert em_oww_metadata.resolve(
        str(tmp_path / "models" / f"{name}.onnx")
    ) == em_oww_metadata.WakeWord("Bonjour Clara", ("fr",))
