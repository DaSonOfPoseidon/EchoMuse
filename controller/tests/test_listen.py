"""em_listen — private listening (docs/listening.md)."""

import em_listen as L


# ── resolve ──────────────────────────────────────────────────────────────────

NEW = {"mic", "oww_shadow", "oww_trigger", L.CAPABILITY}
OLD = {"mic", "oww_shadow", "oww_trigger"}


def test_controller_mode_streams():
    v = L.resolve("off", NEW, "local")
    assert v.state == L.STATE_CONTROLLER and v.streams is True


def test_shadow_is_a_streaming_diagnostic():
    v = L.resolve("shadow", NEW, None)
    assert v.state == L.STATE_DIAGNOSTIC and v.streams is True


def test_private_only_on_the_devices_word():
    """Configuration alone never makes an Echo private: only its report."""
    assert L.resolve("on", NEW, "local").streams is False
    assert L.resolve("on", NEW, None).streams is None
    assert L.resolve("on", NEW, "stream").streams is True


def test_degraded_does_not_stream_and_says_why():
    v = L.resolve("on", NEW, "degraded", "model missing")
    assert v.state == L.STATE_DEGRADED and v.streams is False
    assert "model missing" in v.reason
    assert v.private


def test_old_firmware_is_shown_streaming():
    v = L.resolve("on", OLD, "local")   # a stray report must not override
    assert v.state == L.STATE_LEGACY and v.streams is True
    assert "firmware" in v.reason
    assert not v.private


def test_fleet_summary():
    views = [L.resolve("on", NEW, "local"), L.resolve("off", NEW, None),
             L.resolve("on", NEW, None), L.resolve("on", NEW, "degraded")]
    s = L.fleet_summary(views)
    assert s == {"total": 4, "streaming": 1, "private": 2, "unknown": 1,
                 "degraded": 1}


# ── frames ───────────────────────────────────────────────────────────────────

def test_frame_round_trip():
    pcm = bytes(range(10))
    f = L.build_frame(0xFFFFFFFF, 65537, pcm)
    assert L.parse_frame(f) == (0xFFFFFFFF, 1, pcm)


def test_parse_rejects_other_frames_and_session_zero():
    assert L.parse_frame(b"\x01\x00\x00" + b"\x00" * 10) is None
    assert L.parse_frame(L.build_frame(0, 0, b"xx")) is None
    assert L.parse_frame(b"\x07\x00") is None


def test_frame_layout_matches_the_firmware():
    """[0x07][session u32 BE][seq u16 BE][PCM] — data.go sendListenFrame."""
    f = L.build_frame(1, 2, b"P")
    assert f == b"\x07\x00\x00\x00\x01\x00\x02P"


# ── routing ──────────────────────────────────────────────────────────────────

def test_audio_that_beats_its_wake_is_held_then_delivered_in_order():
    r = L.SessionRouter()
    assert r.frame(1, b"a", 0.0) == []
    assert r.frame(1, b"b", 0.1) == []
    assert r.open(1, 0.2) == [b"a", b"b"]
    assert r.frame(1, b"c", 0.3) == [b"c"]


def test_stragglers_of_a_closed_session_never_reach_the_next():
    r = L.SessionRouter()
    r.open(1, 0.0)
    r.close(1)
    assert r.frame(1, b"late", 0.1) == []
    r.open(2, 0.2)
    assert r.frame(1, b"later", 0.3) == []
    assert r.frame(2, b"x", 0.3) == [b"x"]


def test_opening_a_new_session_closes_the_old():
    r = L.SessionRouter()
    r.open(1, 0.0)
    r.open(2, 0.1)
    assert r.frame(1, b"old", 0.2) == []


def test_unannounced_audio_expires():
    r = L.SessionRouter()
    r.frame(9, b"a", 0.0)
    r.frame(8, b"b", L.PENDING_MAX_S + 0.5)    # triggers expiry of 9
    assert r.open(9, L.PENDING_MAX_S + 0.6) == []
    assert r.dropped >= 1


def test_pending_sessions_are_bounded():
    r = L.SessionRouter()
    for s in range(1, L.PENDING_MAX_SESSIONS + 3):
        r.frame(s, b"x", 0.01 * s)
    assert len(r._pending) == L.PENDING_MAX_SESSIONS
    assert r.open(1, 0.1) == []     # the oldest went


def test_close_with_no_active_session_is_harmless():
    r = L.SessionRouter()
    assert r.close() is None


# ── capture time ─────────────────────────────────────────────────────────────

def test_heard_at_subtracts_age_and_one_way_delay():
    assert L.heard_at(10.0, 500, 200) == 10.0 - 0.5 - 0.1


def test_heard_at_without_information_is_arrival():
    assert L.heard_at(10.0, None, None) == 10.0
    assert L.heard_at(10.0, -40, None) == 10.0   # a negative age is noise


def test_slack_is_the_worst_rto_capped():
    assert L.arbitration_slack([100, 900, None]) == 0.9
    assert L.arbitration_slack([60000]) == L.MAX_ARB_SLACK_S
    assert L.arbitration_slack([]) == 0.0


def test_rtt_estimator_follows_rfc6298():
    e = L.RttEstimator()
    assert e.rto is None
    e.add(100)
    assert e.srtt == 100 and e.rto == 100 + 4 * 50
    e.add(200)
    assert e.srtt == 0.875 * 100 + 0.125 * 200


def test_parse_wake():
    ev = L.parse_wake({"session": 3, "score": 0.8, "threshold": 0.5,
                       "ageMs": 120, "floor": 0.002, "barge": True}, 5.0)
    assert ev == {"session": 3, "score": 0.8, "threshold": 0.5, "age_ms": 120,
                  "floor": 0.002, "barge": True, "arrived": 5.0}


def test_parse_wake_refuses_what_it_cannot_act_on():
    assert L.parse_wake({"score": 0.8}, 0) is None
    assert L.parse_wake({"session": 0, "score": 0.8}, 0) is None
    assert L.parse_wake({"session": 1 << 32, "score": 0.8}, 0) is None
    assert L.parse_wake({"session": 2, "score": "x"}, 0) is None


def test_parse_wake_tolerates_missing_optionals():
    ev = L.parse_wake({"session": 2, "score": 0.9}, 1.0)
    assert ev["floor"] is None and ev["threshold"] is None and ev["age_ms"] == 0


def test_frame_is_bytes_to_every_other_reader():
    f = L.Frame(b"\x01\x02\x03\x04", 12.5)
    assert f == b"\x01\x02\x03\x04" and isinstance(f, bytes)
    assert not isinstance(f, str)          # queue sentinels are str
    buf = bytearray(); buf.extend(f)
    assert bytes(buf[:2]) == b"\x01\x02"
    assert L.arrival(f, 99.0) == 12.5


def test_unstamped_audio_arrives_now():
    assert L.arrival(b"\x00\x00", 99.0) == 99.0
    assert L.arrival("listen_mode", 99.0) == 99.0


def test_a_session_the_echo_closed_is_known_closed():
    r = L.SessionRouter()
    assert not r.is_closed(8)
    r.close(8)                     # listen_end before its wake was acted on
    assert r.is_closed(8)
    assert r.frame(8, b"\0\0", 0.0) == []
