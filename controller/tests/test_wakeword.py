"""
The wake word switch (#286) and how it stays out of the mute button's way (#438).

The rules this file pins, in the order they matter:

- The button and the switch are INDEPENDENT. A press in either direction
  leaves the switch exactly where HA put it — including the re-report the
  device sends on every reconnect, which is not a press at all.
- Turning the wake word off stops the wake stream; turning it back on
  restarts it — unless the device is muted, where it would refuse mic_start
  anyway and restarts its own stream on unmute.
- Because it restarts that stream itself, an unmute while the wake word is
  off leaves a stream up that the listener only discards. That one case, and
  only that one, makes the button reach in here.
- Setting the state it is already in is a no-op, so HA re-asserting a switch
  does not bounce the mic stream.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import em_wakeword


# ── The switch from HA ───────────────────────────────────────────────────────

def test_turning_the_wake_word_off_stops_the_stream():
    t = em_wakeword.on_switch(want=False, enabled=True, hard=False)
    assert t.enabled is False
    assert t.stop_stream is True
    assert t.start_stream is False
    assert t.changed is True


def test_turning_the_wake_word_on_restarts_the_stream():
    t = em_wakeword.on_switch(want=True, enabled=False, hard=False)
    assert t.enabled is True
    assert t.start_stream is True
    assert t.stop_stream is False
    assert t.changed is True


def test_turning_it_on_while_muted_does_not_start_the_stream():
    """The device refuses every mic_start while muted; do not send one we
    know is refused. It restarts the stream itself on unmute."""
    t = em_wakeword.on_switch(want=True, enabled=False, hard=True)
    assert t.enabled is True
    assert t.start_stream is False
    assert t.changed is True


def test_turning_it_off_while_muted_does_not_stop_the_stream():
    """The stream is already down — the device stopped it on mute. The state
    is still recorded, so HA reads back the switch it set."""
    t = em_wakeword.on_switch(want=False, enabled=True, hard=True)
    assert t.enabled is False
    assert t.stop_stream is False
    assert t.changed is True


def test_setting_the_state_already_held_is_a_no_op():
    for want in (True, False):
        t = em_wakeword.on_switch(want=want, enabled=want, hard=False)
        assert t.enabled is want
        assert t.stop_stream is False
        assert t.start_stream is False
        assert t.changed is False


# ── The button does not move the switch ──────────────────────────────────────

def test_the_button_never_moves_the_switch():
    """
    `on_hard_mute` answers one question — whether to re-assert a stopped
    stream — and returns a bare bool rather than a Transition for exactly
    that reason: there is no field here that could carry a new switch state,
    so no press can turn the wake word on or off. That is HA's state, and
    only HA moves it.
    """
    for before, now in ((False, True), (True, False), (False, False), (True, True)):
        for enabled in (True, False):
            out = em_wakeword.on_hard_mute(
                hard_before=before, hard_now=now, enabled=enabled,
            )
            assert out is True or out is False
            assert not isinstance(out, em_wakeword.Transition)


def test_pressing_mute_leaves_the_stream_alone():
    """The device stops its own stream on mute; a controller mic_stop racing
    that is how a stream ends up stuck."""
    for enabled in (True, False):
        assert em_wakeword.on_hard_mute(
            hard_before=False, hard_now=True, enabled=enabled,
        ) is False


def test_unmuting_with_the_wake_word_off_stops_the_stream_again():
    """The device restarts its own wake stream on unmute (cmd/server.go).
    With the wake word off those frames are only discarded, so the controller
    takes the stream back down — which re-asserts the switch rather than
    changing it."""
    assert em_wakeword.on_hard_mute(
        hard_before=True, hard_now=False, enabled=False,
    ) is True


def test_unmuting_with_the_wake_word_on_leaves_the_stream_up():
    assert em_wakeword.on_hard_mute(
        hard_before=True, hard_now=False, enabled=True,
    ) is False


def test_a_reconnect_re_report_changes_nothing():
    """The device sends mute_state on every (re)connect. A Dot that dropped
    off Wi-Fi must not read as a button press in either direction."""
    for hard in (False, True):
        for enabled in (True, False):
            assert em_wakeword.on_hard_mute(
                hard_before=hard, hard_now=hard, enabled=enabled,
            ) is False


# ── Whether the wake word may act ────────────────────────────────────────────

def test_wake_is_allowed_only_when_listening_and_unmuted():
    assert em_wakeword.wake_allowed(hard=False, enabled=True) is True
    assert em_wakeword.wake_allowed(hard=True, enabled=True) is False
    assert em_wakeword.wake_allowed(hard=False, enabled=False) is False
    assert em_wakeword.wake_allowed(hard=True, enabled=False) is False
