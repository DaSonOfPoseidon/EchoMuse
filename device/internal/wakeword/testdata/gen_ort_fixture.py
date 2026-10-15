#!/usr/bin/env python3
"""
gen_ort_fixture.py — regenerate testdata/ort_fixture.bin

Companion to gen_fixture.py, answering a different question. That fixture
REPLAYS model outputs to prove the Go buffering matches openWakeWord's,
without needing ONNX Runtime. This one carries its own INPUT AUDIO so the
`ort` package can be run end to end — real ONNX Runtime, real models — and
checked against what Python produced from the same samples. Together they
cover the two halves: buffering, and inference.

    python3 gen_ort_fixture.py testdata/ort_fixture.bin [chunks]

Two deliberate choices:

  * The audio is SYNTHETIC (a deterministic LCG, int16 scale), not a
    recording. A real utterance would mean committing recognisable speech
    from someone's home to a public repository, and it buys nothing: this
    test asserts numerical agreement between two inference engines, for
    which noise is as discriminating as speech. Detection quality is not
    what is being measured here.

  * The audio is STORED IN THE FIXTURE rather than regenerated in Go from a
    matching LCG. Two implementations of "the same" generator diverging is
    not a hypothetical: an earlier ARM/x86 comparison appeared to show the
    two platforms disagreeing numerically, and the real cause was
    `unsigned long` being 32-bit on armv7a, making a `>>33` undefined. A
    fixture that carries its input cannot have that bug.

A note on the probe block at the end. Run over synthetic noise the classifier
returns 0.0000 for every chunk, so comparing scores through the streaming path
would pass just as happily for a classifier stuck at zero — it asserts almost
nothing. The probe therefore feeds the embedding and classifier models one
deterministic wide-range tensor each, chosen to land in the range each model
actually sees (post `/10+2` mel values; embeddings), and records the outputs.
That gives the classifier non-degenerate coverage independent of the audio.

Format (little-endian):
    magic "OWWORT02", uint32 n_samples, int16[n_samples] audio,
    uint32 n_records, then per record:
      uint32 mel_input_samples   -- length of the audio slice fed to melspec
      uint32 mel_frames          -- rows in the melspec model's raw output
      float32[mel_frames*32]     -- that raw output (pre /10+2 transform)
      float32[96]                -- the embedding model's output
      float32                    -- the classifier's score
    then the probe block:
      float32[76*32] embedding input, float32[96]    its output
      float32[16*96] classifier input, float32[1]    its output
"""

import numpy as np, struct, sys, os

OUT = sys.argv[1]
CHUNKS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
CH = 1280

import openwakeword
from openwakeword.utils import AudioFeatures
import onnxruntime as ort

RES = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
CLS = os.environ.get("EM_CLASSIFIER", os.path.join(RES, "hey_mycroft_v0.1.onnx"))

# Deterministic input, int16 scale. Speech peaks around -10dBFS on this
# hardware; +/-8000 is the same order, so the melspectrogram sees plausible
# magnitudes rather than a degenerate near-silent or clipped signal.
rng = np.random.RandomState(20260730)
audio = rng.randint(-8000, 8001, size=CHUNKS * CH).astype(np.int16)

# AudioFeatures seeds its feature ring from random audio; fix the global seed
# so this file is byte-reproducible. (The Go side starts clean instead, which
# is why the classifier comparison below only starts once the ring holds real
# audio -- see the test.)
np.random.seed(0)
af = AudioFeatures(ncpu=1)
cls = ort.InferenceSession(CLS, providers=["CPUExecutionProvider"])
cls_in = cls.get_inputs()[0].name

mel_out = []
orig_mel = af.melspec_model_predict


def spy_mel(x):
    o = orig_mel(x)
    mel_out.append((x.shape[1], np.squeeze(o[0]).astype(np.float32)))
    return o


af.melspec_model_predict = spy_mel

emb_out = []
orig_emb = af.embedding_model_predict


def spy_emb(x):
    o = orig_emb(x)
    emb_out.append(o.reshape(-1, 96))
    return o


af.embedding_model_predict = spy_emb

recs = []
for i in range(0, len(audio), CH):
    before_mel, before_emb = len(mel_out), len(emb_out)
    af._streaming_features(audio[i : i + CH])
    feats = af.get_features(16)
    score = float(cls.run(None, {cls_in: feats.astype(np.float32)})[0].squeeze())
    new_mel, new_emb = mel_out[before_mel:], emb_out[before_emb:]
    assert len(new_mel) == 1 and len(new_emb) == 1, (len(new_mel), len(new_emb))
    recs.append(
        dict(
            mel_in_len=new_mel[0][0],
            mel=new_mel[0][1],
            emb=new_emb[0].reshape(-1),
            score=score,
        )
    )

# Probe tensors: deterministic, wide-range, and in the distribution each model
# expects -- mel values after the /10+2 transform sit around 0..4, embeddings
# span roughly -4..4.
prng = np.random.RandomState(777)
emb_probe_in = prng.uniform(0.0, 4.0, size=76 * 32).astype(np.float32)
emb_probe_out = np.asarray(
    af.embedding_model_predict(emb_probe_in.reshape(1, 76, 32, 1)), dtype=np.float32
).reshape(-1)
cls_probe_in = prng.uniform(-4.0, 4.0, size=16 * 96).astype(np.float32)
cls_probe_out = np.asarray(
    cls.run(None, {cls_in: cls_probe_in.reshape(1, 16, 96)})[0], dtype=np.float32
).reshape(-1)
assert emb_probe_out.size == 96, emb_probe_out.shape
assert cls_probe_out.size == 1, cls_probe_out.shape

with open(OUT, "wb") as f:
    f.write(b"OWWORT02")
    f.write(struct.pack("<I", audio.size))
    f.write(audio.astype("<i2").tobytes())
    f.write(struct.pack("<I", len(recs)))
    for r in recs:
        f.write(struct.pack("<I", r["mel_in_len"]))
        f.write(struct.pack("<I", r["mel"].shape[0]))
        f.write(r["mel"].astype("<f4").tobytes())
        f.write(r["emb"].astype("<f4").tobytes())
        f.write(struct.pack("<f", r["score"]))
    f.write(emb_probe_in.astype("<f4").tobytes())
    f.write(emb_probe_out.astype("<f4").tobytes())
    f.write(cls_probe_in.astype("<f4").tobytes())
    f.write(cls_probe_out.astype("<f4").tobytes())

print("probe: classifier score on wide-range input =", float(cls_probe_out[0]))
print(f"wrote {len(recs)} records ({audio.size} samples) -> {OUT}")
print("onnxruntime:", ort.__version__)
print("classifier:", CLS)
print("mel input lens:", sorted(set(r["mel_in_len"] for r in recs)))
print("mel frames/chunk:", sorted(set(r["mel"].shape[0] for r in recs)))
print("scores:", " ".join(f"{r['score']:.4f}" for r in recs[:8]), "...")
