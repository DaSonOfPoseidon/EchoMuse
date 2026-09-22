"""
em_listen.py — private listening: sessions, per-Echo state, capture time
=========================================================================

The controller half of docs/listening.md. An Echo listening privately scores
its own wake word and sends nothing until it fires; a wake opens a numbered
SESSION whose audio arrives on /data as 0x07 frames tagged with that number,
and the controller closes it at end of speech. Everything here is pure, so it
is tested without aiohttp, openwakeword or a socket (tests/test_listen.py).

Three pieces:

- `resolve` — what an Echo is actually doing, from what was configured, what
  the firmware can do and what the Echo last reported. The dashboard's privacy
  statement is built from this and nothing else.
- `SessionRouter` — which session audio may reach a turn. Control and data are
  separate sockets, so a session's first frames can beat its `oww_wake`, and a
  late frame of a closed session can arrive after the controller moved on.
  Untagged audio was routed by a flag, and a frame on the wrong side of a flag
  flip became the start of the next person's command.
- `heard_at` / `arbitration_slack` — capture time in the controller's clock,
  so arbitration picks the Echo that heard first rather than the one whose
  message arrived first.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# The controller's feature, announced on the ack. The device listens privately
# only when it sees this: an older controller acts on a device wake only when
# wake-stream frames are arriving, so a device that went quiet on it is deaf.
FEATURE = "listen_session"

# The device's capability: it CAN listen privately. Whether it IS comes from
# its listen_state report.
CAPABILITY = "oww_local_only"

FRAME_TYPE = 0x07
_HEADER = struct.Struct(">BIH")    # type, session, seq

# How long frames for a session nobody has announced are held, waiting for its
# oww_wake. Covers the control plane lagging the data plane by a retransmit or
# two (#139 measured RTO at 500-800ms) with room to spare; beyond it the wake
# is not coming, and the device's own ack timeout (3s) has closed the session.
PENDING_MAX_S = 3.0
# At most this many unannounced sessions held at once. More means something is
# wrong, and the oldest is dropped rather than growing without bound.
PENDING_MAX_SESSIONS = 4
# Closed ids remembered, so their stragglers are dropped rather than held.
CLOSED_MEMORY = 32

# Arbitration never holds a claim open longer than this past its window,
# whatever the RTT estimate says — a claim window measured in many seconds
# would swallow genuinely separate wakes in other rooms.
MAX_ARB_SLACK_S = 3.0


# ── State ────────────────────────────────────────────────────────────────────

STATE_LOCAL      = "local"
STATE_CONTROLLER = "controller"
STATE_DIAGNOSTIC = "diagnostic"
STATE_LEGACY     = "legacy"
STATE_DEGRADED   = "degraded"
STATE_UNKNOWN    = "unknown"


@dataclass(frozen=True)
class ListenView:
    """What an Echo is doing with its microphone, for the dashboard and logs.

    `streams` is True/False, or None when it is not yet known — which the
    dashboard must render as unknown, never as private.
    """
    state:   str
    streams: bool | None
    reason:  str = ""

    @property
    def private(self) -> bool:
        """Session-driven: no wake stream, turns start from device wakes."""
        return self.state in (STATE_LOCAL, STATE_DEGRADED)


def resolve(configured: str, capabilities, reported: str | None,
            reported_reason: str = "") -> ListenView:
    """
    The state in force for one Echo.

    `configured` is the stored owwOnDevice ("on"/"off"/"shadow", already
    normalised). `reported` is the device's last listen_state, None if it has
    not sent one on this connection.
    """
    caps = set(capabilities or ())
    if configured == "off":
        return ListenView(STATE_CONTROLLER, True)
    if configured == "shadow":
        return ListenView(STATE_DIAGNOSTIC, True,
                          "diagnostic: both detectors score a continuous stream")
    # configured == "on"
    if CAPABILITY not in caps:
        # Firmware from before private listening streams in every mode. It is
        # shown streaming, because it is, with the fix named.
        return ListenView(STATE_LEGACY, True,
                          "update the firmware for private listening")
    if reported == "local":
        return ListenView(STATE_LOCAL, False)
    if reported == "degraded":
        return ListenView(STATE_DEGRADED, False,
                          reported_reason or "cannot run its wake word — button only")
    if reported == "stream":
        # The device has not switched yet (a config push in flight). It is
        # streaming now, and saying otherwise would be the one lie this
        # module exists to prevent.
        return ListenView(STATE_UNKNOWN, True, "switching to private listening")
    return ListenView(STATE_UNKNOWN, None, "waiting for the Echo to report")


def fleet_summary(views) -> dict:
    """Counts for the dashboard's fleet line ("1 of 3 Echoes streams…")."""
    views = list(views)
    return {
        "total":     len(views),
        "streaming": sum(1 for v in views if v.streams is True),
        "private":   sum(1 for v in views if v.streams is False),
        "unknown":   sum(1 for v in views if v.streams is None),
        "degraded":  sum(1 for v in views if v.state == STATE_DEGRADED),
    }


# ── Frames ───────────────────────────────────────────────────────────────────

def parse_frame(data: bytes) -> tuple[int, int, bytes] | None:
    """(session, seq, pcm) from a 0x07 frame, or None if it is not one."""
    if len(data) < _HEADER.size or data[0] != FRAME_TYPE:
        return None
    _, session, seq = _HEADER.unpack_from(data)
    if session == 0:
        return None
    return session, seq, bytes(data[_HEADER.size:])


def build_frame(session: int, seq: int, pcm: bytes) -> bytes:
    """The inverse of parse_frame; for tests and tools."""
    return _HEADER.pack(FRAME_TYPE, session, seq & 0xFFFF) + pcm


# ── Routing ──────────────────────────────────────────────────────────────────

@dataclass
class _Pending:
    first_seen: float
    frames: list = field(default_factory=list)


class SessionRouter:
    """
    Decides which session audio reaches a turn, for one Echo.

    - A frame for the ACTIVE session is delivered.
    - A frame for a session this router has closed is dropped.
    - A frame for a session not yet announced is held (bounded) until its
      wake arrives, then delivered in order ahead of anything newer.

    One active session at a time, because an Echo holds at most one open.
    """

    def __init__(self) -> None:
        self.active: int | None = None
        self._pending: dict[int, _Pending] = {}
        self._closed: list[int] = []
        self.dropped = 0          # frames discarded: closed or expired

    def frame(self, session: int, pcm: bytes, now: float) -> list[bytes]:
        """Route one frame; returns what to deliver now (0 or 1 chunks)."""
        self._expire(now)
        if session == self.active:
            return [pcm]
        if session in self._closed:
            self.dropped += 1
            return []
        p = self._pending.get(session)
        if p is None:
            if len(self._pending) >= PENDING_MAX_SESSIONS:
                oldest = min(self._pending, key=lambda s: self._pending[s].first_seen)
                self.dropped += len(self._pending.pop(oldest).frames)
            p = self._pending[session] = _Pending(first_seen=now)
        p.frames.append(pcm)
        return []

    def open(self, session: int, now: float) -> list[bytes]:
        """Make `session` active; returns its held frames, oldest first.

        Any previously active session is closed: the Echo has moved on, so
        its stragglers belong to nobody.
        """
        self._expire(now)
        if self.active is not None and self.active != session:
            self._remember_closed(self.active)
        self.active = session
        p = self._pending.pop(session, None)
        return p.frames if p else []

    def close(self, session: int | None = None) -> int | None:
        """Close `session` (or the active one). Returns what was closed."""
        target = self.active if session is None else session
        if target is None:
            return None
        if target == self.active:
            self.active = None
        p = self._pending.pop(target, None)
        if p:
            self.dropped += len(p.frames)
        self._remember_closed(target)
        return target

    def is_closed(self, session: int) -> bool:
        """True if `session` has ended, here or on the Echo."""
        return session in self._closed

    def reset(self) -> None:
        """A new connection: nothing from the old one is routable."""
        self.__init__()

    def _remember_closed(self, session: int) -> None:
        if session in self._closed:
            return
        self._closed.append(session)
        if len(self._closed) > CLOSED_MEMORY:
            del self._closed[0]

    def _expire(self, now: float) -> None:
        for s in [s for s, p in self._pending.items()
                  if now - p.first_seen > PENDING_MAX_S]:
            self.dropped += len(self._pending.pop(s).frames)
            self._remember_closed(s)


# ── Capture time ─────────────────────────────────────────────────────────────

def heard_at(arrived: float, age_ms, srtt_ms) -> float:
    """
    When the wake's audio was captured, in the controller's clock.

    `age_ms` is how long before sending the device captured it (0 for a wake
    the controller scored itself); half the smoothed RTT estimates the one-way
    delay. Both absent reads as "heard on arrival", which is exactly what the
    arbiter assumed before this existed — degrade to old behaviour, never to a
    guess.
    """
    age = max(0.0, float(age_ms or 0)) / 1000.0
    one_way = max(0.0, float(srtt_ms or 0)) / 2000.0
    return arrived - age - one_way


class Frame(bytes):
    """
    Stream mic audio stamped with when it arrived, in the loop's clock.

    A controller-scored wake is heard when its frame ARRIVED, not when
    inference on it finished: the frame can wait behind a backlog in
    mic_queue (up to 5s) and the executor. Timing the claim at the crossing
    put that wait into every arbitration against an Echo that reports its
    own capture age, and the wait grows with the fleet. A bytes subclass so
    every other reader of the queues is unchanged.
    """

    def __new__(cls, data: bytes, arrived: float):
        f = super().__new__(cls, data)
        f.arrived = arrived
        return f


def arrival(payload, default: float) -> float:
    """When `payload` arrived; `default` for audio that was never stamped."""
    return getattr(payload, "arrived", default)


def arbitration_slack(rto_ms_values) -> float:
    """
    How long past its window a claim is held, so a claim delayed in flight
    still finds it and cedes rather than reading as a new utterance.

    The worst retransmission timeout across the fleet (TCP's srtt + 4·rttvar)
    bounds how late a message can plausibly arrive; capped at
    MAX_ARB_SLACK_S.
    """
    worst = max((float(v) for v in rto_ms_values if v), default=0.0)
    return min(MAX_ARB_SLACK_S, worst / 1000.0)


class RttEstimator:
    """TCP's smoothed RTT and variance (RFC 6298), per Echo, in ms."""

    def __init__(self) -> None:
        self.srtt: float | None = None
        self.rttvar: float = 0.0

    def add(self, rtt_ms: float) -> None:
        r = float(rtt_ms)
        if self.srtt is None:
            self.srtt, self.rttvar = r, r / 2
            return
        self.rttvar = 0.75 * self.rttvar + 0.25 * abs(self.srtt - r)
        self.srtt = 0.875 * self.srtt + 0.125 * r

    @property
    def rto(self) -> float | None:
        if self.srtt is None:
            return None
        return self.srtt + 4 * self.rttvar


# ── Wakes ────────────────────────────────────────────────────────────────────

def parse_wake(msg: dict, arrived: float) -> dict | None:
    """
    A session-bearing oww_wake, validated. None if it cannot be acted on.

    `age_ms` is how long before sending the device captured the wake word's
    last frame; `floor` is its room noise floor (RMS 0..1), which the
    controller can no longer measure from a stream it does not receive.
    """
    try:
        session = int(msg["session"])
        score = float(msg["score"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 < session <= 0xFFFFFFFF:
        return None
    def _num(key, default=None):
        try:
            v = msg.get(key)
            return default if v is None else float(v)
        except (TypeError, ValueError):
            return default
    age = _num("ageMs", 0.0)
    return {
        "session":   session,
        "score":     score,
        "threshold": _num("threshold"),
        "age_ms":    int(max(0.0, age)),
        "floor":     _num("floor"),
        "barge":     bool(msg.get("barge")),
        "arrived":   arrived,
    }
