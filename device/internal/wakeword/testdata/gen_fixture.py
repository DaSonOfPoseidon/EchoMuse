#!/usr/bin/env python3
"""
gen_fixture.py — regenerate testdata/stream_fixture.bin

The fixture is a golden capture of openWakeWord's OWN streaming buffer
behaviour, used by stream_test.go to prove the Go implementation reproduces
it tensor for tensor without needing ONNX Runtime in CI.

Run inside the controller container (it has openwakeword + onnxruntime and
the model files):

    docker cp gen_fixture.py echomuse-controller:/tmp/
    docker exec echomuse-controller python /tmp/gen_fixture.py \
        /app/data/recordings/<some>.wav /tmp/stream_fixture.bin
    docker cp echomuse-controller:/tmp/stream_fixture.bin \
        device/internal/wakeword/testdata/

Any 16kHz mono 16-bit WAV of at least ~2.5s works; the utterance recordings
written by saveUtterances are the obvious source. Note those contain no wake
word (the preroll discard removes it), so the captured scores are ~0 — that
is fine, because the assertions are on the FNV hashes of the exact tensors
fed to each model, not on the scores.

Format (little-endian):
    magic "OWWFIX01", uint32 n_records, then per record:
      uint32 mel_input_samples   -- length of the audio slice fed to melspec
      uint32 mel_frames          -- rows in the melspec model's raw output
      float32[mel_frames*32]     -- that raw output (pre /10+2 transform)
      uint64 window_fnv          -- FNV-1a of the 76x32 embedding input
      float32[96]                -- the embedding model's output
      uint64 feat_fnv            -- FNV-1a of the 16x96 classifier input
      float32                    -- the classifier's score
"""

import numpy as np, struct, wave, sys

sys.path.insert(0, "/usr/local/lib/python3.12/site-packages")
from openwakeword.utils import AudioFeatures
import onnxruntime as ort

WAV = sys.argv[1]
OUT = sys.argv[2]
D = "/usr/local/lib/python3.12/site-packages/openwakeword/resources/models"
with wave.open(WAV, "rb") as w:
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
print("audio", audio.shape, "=", audio.size / 16000, "s")

# AudioFeatures seeds its feature ring by running four seconds of RANDOM
# audio through the embedding model, so without a fixed seed this fixture is
# not byte-reproducible. The Go test only asserts the classifier input from
# chunk 16 onward (by which point the ring holds only real audio), but a
# golden file that changes on every regeneration is a bad artifact.
np.random.seed(0)
af = AudioFeatures(ncpu=1)
cls = ort.InferenceSession(
    f"{D}/hey_mycroft_v0.1.onnx", providers=["CPUExecutionProvider"]
)
cls_in = cls.get_inputs()[0].name


def fnv(a):
    h = 14695981039346656037
    for b in np.ascontiguousarray(a, dtype=np.float32).tobytes():
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


# monkeypatch to capture what actually gets fed to the embedding model
captured = []
orig = af.embedding_model_predict


def spy(x):
    out = orig(x)
    captured.append((fnv(x), out.reshape(-1, 96)))
    return out


af.embedding_model_predict = spy

# also capture melspec model outputs (what the Go mock will replay)
mel_out = []
orig_mel = af.melspec_model_predict


def spy_mel(x):
    o = orig_mel(x)
    mel_out.append((x.shape[1], np.squeeze(o[0]).astype(np.float32)))
    return o


af.melspec_model_predict = spy_mel

recs = []
CH = 1280
for i in range(0, len(audio) // CH * CH, CH):
    chunk = audio[i : i + CH]
    before_mel = len(mel_out)
    before_emb = len(captured)
    af._streaming_features(chunk)
    feats = af.get_features(16)
    score = float(cls.run(None, {cls_in: feats.astype(np.float32)})[0].squeeze())
    new_mel = mel_out[before_mel:]
    new_emb = captured[before_emb:]
    assert len(new_mel) == 1 and len(new_emb) == 1, (len(new_mel), len(new_emb))
    recs.append(
        dict(
            mel_in_len=new_mel[0][0],
            mel=new_mel[0][1],
            win_fnv=new_emb[0][0],
            emb=new_emb[0][1].reshape(-1),
            feat_fnv=fnv(feats),
            score=score,
        )
    )

with open(OUT, "wb") as f:
    f.write(b"OWWFIX01")
    f.write(struct.pack("<I", len(recs)))
    for r in recs:
        f.write(struct.pack("<I", r["mel_in_len"]))
        f.write(struct.pack("<I", r["mel"].shape[0]))
        f.write(r["mel"].astype("<f4").tobytes())
        f.write(struct.pack("<Q", r["win_fnv"]))
        f.write(r["emb"].astype("<f4").tobytes())
        f.write(struct.pack("<Q", r["feat_fnv"]))
        f.write(struct.pack("<f", r["score"]))
print(f"wrote {len(recs)} records -> {OUT}")
print("mel input lens:", sorted(set(r["mel_in_len"] for r in recs)))
print("mel frames/chunk:", sorted(set(r["mel"].shape[0] for r in recs)))
print("scores:", " ".join(f"{r['score']:.4f}" for r in recs[:12]), "...")
