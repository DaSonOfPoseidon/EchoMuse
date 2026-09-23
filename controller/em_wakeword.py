"""
em_wakeword.py — the wake word switch Home Assistant sets (#286)

Two things stop this device answering, and they are unrelated:

- The MUTE BUTTON stops the microphone. The firmware mutes all four codec
  pairs and refuses every `mic_start` while it holds. The controller only
  ever LEARNS it, from `mute_state`, and reports it to HA as a read-only
  sensor (#438) — there is no entity that could clear it.
- The WAKE WORD SWITCH stops the controller acting on a wake word, and
  takes the continuous wake stream down. Nothing more. It never touches the
  microphone path, so it cannot enable anything that is not already enabled
  by default — which is what makes exposing it to HA safe. Under private
  listening (#602) there is no continuous stream, and `mic_stop` ends
  neither a session nor the Echo's local listening, so the Echo keeps
  scoring; `_private_wake_turn` declines each of its wakes instead, with
  `listen_close(session, "wake_off")`.

They are INDEPENDENT, in both directions: pressing mute does not turn the
switch off, and pressing unmute does not turn it back on. The switch is HA's
state and only HA moves it, the same way the mute is the device's and only
the button moves it. That independence is why "soft mute" was the wrong name
for this — the microphone is still on, and an HA-initiated
`start_conversation` or `ask_question` still listens, exactly as it does
under the button mute.

What the button does force is the STREAM. The device stops its own wake
stream on mute and restarts it on unmute (`cmd/server.go`), with no
controller involvement, so an unmute while the switch is off brings up a
stream whose frames the wake listener would only discard. `on_hard_mute`
says to take it back down. That re-asserts the switch; it does not change
it, and it is the only thing the button decides here.

Like `em_button`, this is split out because the test suite does not import
`em_controller`, and this is policy that deserves coverage.
"""

from __future__ import annotations

from typing import NamedTuple


class Transition(NamedTuple):
    # Wake word state after this event — True means listening.
    enabled: bool
    # Send mic_stop — take the wake stream down.
    stop_stream: bool
    # Send mic_start — bring the wake stream back.
    start_stream: bool
    # The state changed, so HA needs a SwitchStateResponse.
    changed: bool


def on_switch(*, want: bool, enabled: bool, hard: bool) -> Transition:
    """
    HA set the switch to `want`, with the wake word currently `enabled` and
    the button mute currently `hard`.
    """
    if want == enabled:
        return Transition(enabled=enabled, stop_stream=False, start_stream=False, changed=False)
    if want:
        # The device refuses mic_start while muted, and restarts its own
        # stream on unmute — so under the hard mute there is nothing to send.
        return Transition(enabled=True, stop_stream=False, start_stream=not hard, changed=True)
    # Under the hard mute the device already stopped the stream itself.
    return Transition(enabled=False, stop_stream=not hard, start_stream=False, changed=True)


def on_hard_mute(*, hard_before: bool, hard_now: bool, enabled: bool) -> bool:
    """
    The device reported `mute_state` as `hard_now`; `hard_before` is what the
    controller remembered across connections (the device re-sends mute_state
    on every reconnect, and that is not a press).

    Returns whether to send mic_stop. The switch itself never moves here: the
    button and the switch are independent, so this answers one question only —
    the device restarts its own wake stream on unmute, and with the switch off
    those frames are discarded, so the stream goes back down.
    """
    return hard_before and not hard_now and not enabled


def wake_allowed(*, hard: bool, enabled: bool) -> bool:
    """Whether a wake-word crossing may start a turn."""
    return enabled and not hard
