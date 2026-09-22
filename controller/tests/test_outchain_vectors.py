"""
The device's output chain (device/internal/outchain) is a port of em_eq,
em_mbc and em_limiter, held to vectors generated from them. This checks the
committed vectors still ARE what the Python produces.

Without it the two halves can drift in silence: change the limiter here and
the Go test keeps passing against the old recording, while a device and the
controller now process the same audio differently. When this fails, carry
the change to the Go, then regenerate:

    cd controller && python3 ../device/internal/outchain/testdata/gen_vectors.py
"""

import gzip
import importlib.util
import json
import os

import numpy as np
import pytest

VECTORS = os.path.join(os.path.dirname(__file__), "..", "..", "device",
                       "internal", "outchain", "testdata")


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "gen_vectors", os.path.join(VECTORS, "gen_vectors.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load_generator()

with open(os.path.join(VECTORS, "vectors.json")) as _f:
    MANIFEST = {c["name"]: c for c in json.load(_f)}


def _read(name, suffix):
    with gzip.open(os.path.join(VECTORS, f"{name}.{suffix}.s16.gz")) as f:
        return np.frombuffer(f.read(), dtype="<i2")


def test_every_case_is_committed():
    assert sorted(MANIFEST) == sorted(c["name"] for c in gen.CASES)


@pytest.mark.parametrize("case", gen.CASES, ids=lambda c: c["name"])
def test_committed_vectors_match_the_python_chain(case):
    x, y, stats = gen.render(case)
    assert np.array_equal(_read(case["name"], "in"), x)
    assert np.array_equal(_read(case["name"], "out"), y), \
        "the Python chain no longer produces the committed output"
    assert MANIFEST[case["name"]] == gen.manifest_entry(case, stats)
