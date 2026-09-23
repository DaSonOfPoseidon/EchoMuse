"""
The two entities Home Assistant sees for a device's ears (#438, #286).

Read as source, not imported: em_esphome pulls in zeroconf and aiohttp, which
this suite deliberately does without. What these pin:

- The binary sensor for the button mute is READ-ONLY. A writable entity here
  would be a remote unmute, the one thing the mute button exists to make
  impossible — so nothing in the controller may accept a command for it, and
  the only writable entity is the wake word switch.
- The button and the wake word switch are INDEPENDENT: `mute_state` must not
  move the switch, in either direction.
- The switch is NOT named "Wake word". HA's own ESPHome integration puts an
  entity by that name on this same device (its satellite wake-word picker),
  and two controls sharing a name on one device page is the trap this name
  was chosen to avoid.
- HA's picker cannot WRITE this switch — HA re-reads the satellite config
  only at setup and right after it writes, so a picker wired to the switch
  would drift the moment the switch moved. It still READS the truth.
- Both entities are gated on `mic`, like every entity, so a device that
  cannot listen does not grow a switch that does nothing.
- Keys 4 and 5 are taken and stay taken: HA keys its registry on them.
- Subscribing to states yields both, or HA shows "unknown" until the next
  device event — which for the button mute could be never.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ESPHOME = ROOT / "controller" / "em_esphome.py"
CONTROLLER = ROOT / "controller" / "em_controller.py"


def _block(src: str, start: str, end: str) -> str:
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j]


def test_entity_keys_are_appended_not_renumbered():
    src = ESPHOME.read_text()
    assert re.search(r"^MEDIA_PLAYER_KEY\s*=\s*1\b", src, re.M)
    assert re.search(r"^EVENT_KEY\s*=\s*2\b", src, re.M)
    assert re.search(r"^AMBIENT_LUX_KEY\s*=\s*3\b", src, re.M)
    assert re.search(r"^MIC_MUTED_KEY\s*=\s*4\b", src, re.M)
    assert re.search(r"^WAKE_WORD_KEY\s*=\s*5\b", src, re.M)


def test_both_entities_are_advertised_and_gated_on_mic():
    src = ESPHOME.read_text()
    entities = _block(src, "isinstance(msg, api_pb2.ListEntitiesRequest)",
                      "ListEntitiesDoneResponse()")
    sensor = _block(entities, "ListEntitiesBinarySensorResponse(", ")")
    switch = _block(entities, "ListEntitiesSwitchResponse(", ")")
    assert "key=MIC_MUTED_KEY" in sensor
    assert "key=WAKE_WORD_KEY" in switch
    # Gated: the yield sits under the mic check, not at the top level.
    gate = entities.index("if self._mic_capable:")
    assert gate < entities.index("ListEntitiesBinarySensorResponse(")
    assert gate < entities.index("ListEntitiesSwitchResponse(")


def test_the_switch_is_named_for_what_it_does_and_not_after_has_entity():
    """
    "Soft Mute" was the first name and the wrong one: the microphone stays
    on, and only the button turns it off. On means listening.

    "Wake word" is the wrong one too, for a different reason — that is the
    name HA's own ESPHome integration gives the satellite wake-word picker
    (translation_key "wake_word"), which it creates for every device with
    non-zero voice_assistant_feature_flags, i.e. ours. Two "Wake word"
    controls on one device page is worse than the name it replaced.
    """
    src = ESPHOME.read_text()
    entities = _block(src, "isinstance(msg, api_pb2.ListEntitiesRequest)",
                      "ListEntitiesDoneResponse()")
    switch = _block(entities, "ListEntitiesSwitchResponse(", ")")
    assert 'name="Wake word detection"' in switch
    assert 'object_id="wake_word_detection"' in switch
    assert "mute" not in switch.lower()


def test_entity_names_do_not_repeat_the_device_label():
    src = ESPHOME.read_text()
    entities = _block(src, "isinstance(msg, api_pb2.ListEntitiesRequest)",
                      "ListEntitiesDoneResponse()")
    for ctor in ("ListEntitiesBinarySensorResponse(", "ListEntitiesSwitchResponse("):
        assert "self.label" not in _block(entities, ctor, ")")


def test_the_button_mute_is_read_only():
    """
    The mute button is worth having because software cannot undo it. The only
    command the controller accepts is the wake word switch, and it must not
    act on any other key.
    """
    src = ESPHOME.read_text()
    handler = _block(src, "isinstance(msg, api_pb2.SwitchCommandRequest)",
                     "\n        if isinstance(msg, ")
    assert "WAKE_WORD_KEY" in handler
    assert "MIC_MUTED_KEY" not in handler
    assert "_apply_wake_word" in handler, "the switch is the one writable control"


def test_subscribing_to_states_reports_both():
    src = ESPHOME.read_text()
    sub = _block(src, "isinstance(msg, (api_pb2.SubscribeStatesRequest,",
                 "SubscribeVoiceAssistantRequest")
    assert "_mute_state_msg()" in sub
    assert "_wake_word_msg()" in sub


def test_the_switch_defaults_to_listening():
    """A device nobody has told otherwise listens, so absent state reads ON.
    The mute defaults the other way: absent state is not muted."""
    src = ESPHOME.read_text()
    init = _block(src, "class DeviceESPhomeServer", "def set_capabilities")
    assert re.search(r"^\s{8}self\.wake_word_enabled\s*:\s*bool\s*=\s*True", init, re.M)
    assert re.search(r"^\s{8}self\.muted\s*:\s*bool\s*=\s*False", init, re.M)
    assert "return False, True" in _block(src, "def get_mute_and_wake(", "\n\n\n")


def test_the_switch_survives_a_device_reconnect():
    """
    A Device is rebuilt per connection, so the switch must live on the server
    object that outlives it — the same place volume lives — and so must the
    hard state that `mute_state` is compared against, or every reconnect
    reads as a button press.
    """
    src = CONTROLLER.read_text()
    connect = _block(src, "device.wake_word_enabled = esphome.get_mute_and_wake(",
                     "\n        #")
    assert "get_mute_and_wake" in connect


def test_the_picker_reports_the_switch_but_cannot_set_it():
    """
    HA re-reads the satellite configuration in exactly two places — when the
    satellite entity is added, and straight after HA itself writes wake words
    (assist_satellite.py `_update_satellite_config`). There is no unsolicited
    push, so a picker wired to the switch would agree when HA drove it and
    drift the moment the switch did. It declines instead, and the warning
    names the switch.

    The READ side still follows the switch: reporting the model
    unconditionally would tell HA the device is listening when it is not, and
    reporting the truth makes the refusal self-correcting, since HA re-reads
    right after a rejected write.
    """
    src = ESPHOME.read_text()
    setcfg = _block(src, "isinstance(msg, api_pb2.VoiceAssistantSetConfiguration)",
                    "\n        if isinstance(msg, ")
    assert "_apply_wake_word" not in setcfg, "the picker must not write the switch"
    assert "Wake word detection switch" in setcfg, "the refusal must name the switch"
    cfg = _block(src, "isinstance(msg, api_pb2.VoiceAssistantConfigurationRequest)",
                 "\n        if isinstance(msg, ")
    assert "active_wake_words=[self.oww_model_id] if enabled else []" in cfg
    assert "wake_word_enabled" in cfg


def test_the_controller_reports_the_button_mute_to_ha():
    src = CONTROLLER.read_text()
    handler = _block(src, 'msg_type == "mute_state"', 'msg_type == "volume_state"')
    assert "esphome.update_mute_state(" in handler


def test_the_button_does_not_move_the_wake_word_switch():
    """
    The review's ask, pinned where it can regress: `mute_state` may READ the
    switch — it has to, to know whether the stream the device just restarted
    is wanted — but it must never assign it. The two are independent.
    """
    src = CONTROLLER.read_text()
    handler = _block(src, 'msg_type == "mute_state"', 'msg_type == "volume_state"')
    assert "em_wakeword.on_hard_mute(" in handler
    assert not re.search(r"\.wake_word_enabled\s*=", handler)
    assert "esphome.update_wake_word(" not in handler


def test_unmuting_with_the_wake_word_off_takes_the_stream_back_down():
    """The device restarts its own wake stream on unmute; with the wake word
    off the controller would only discard those frames."""
    src = CONTROLLER.read_text()
    handler = _block(src, 'msg_type == "mute_state"', 'msg_type == "volume_state"')
    branch = _block(handler, "em_wakeword.on_hard_mute(", "if device.muted and")
    assert "mic_stop()" in branch


def test_the_wake_listener_honours_the_switch():
    """Both the frame gate and the stall watchdog of the stream path: a
    watchdog that only knows the button mute would restart the stream the
    switch just stopped. The private path is its own test, below."""
    src = CONTROLLER.read_text()
    listener = _block(src, "async def _stream_listen(", "\nasync def ")
    assert listener.count("em_wakeword.wake_allowed(") >= 2


def test_the_wake_stream_does_not_come_up_with_the_switch_off():
    """
    Nine call sites restart the wake stream after something — a turn, an
    announcement, an alarm, a barge. Gating each one is nine places to
    forget, so `Device.mic_start` is the gate, the same way the device
    itself refuses `mic_start` under the button mute. `mic_start_turn` is
    deliberately not gated: an HA-initiated turn is HA's decision.
    """
    src = CONTROLLER.read_text()
    start = _block(src, "    async def mic_start(self):", "    async def mic_start_turn(self):")
    assert "wake_word_enabled" in start
    turn = _block(src, "    async def mic_start_turn(self):", "    async def mic_stop(self):")
    assert "wake_word_enabled" not in turn


def test_the_api_readout_does_not_default_the_switch():
    """
    `/api/devices` reports the switch for a device that is OFFLINE, because
    the switch lives on the ESPHome server and the server outlives the
    connection — so it must be read from there, not defaulted off `live`.

    Its neighbours in that block (speaking/listening/thinking) default to
    False for a disconnected device and that is correct: an offline device
    genuinely is not doing any of them. The switch is not live activity, and
    a default there would report "listening" for a device HA switched off —
    a readout disagreeing with the decision it describes, which is the rule
    written on em_esphome.get_status.
    """
    src = (ROOT / "controller" / "em_api.py").read_text()
    lines = [ln for ln in src.splitlines() if '"wake_word":' in ln]
    assert len(lines) == 1, lines
    assert "em_esphome.get_wake_word(" in lines[0], lines[0].strip()
    assert "if live else" not in lines[0], lines[0].strip()
    # And the accessor says "unknown" rather than picking a side.
    esp = _block(ESPHOME.read_text(), "def get_wake_word(", "\n\n\n")
    assert "Optional[bool]" in esp
    assert "return None" in esp


def test_the_switch_paints_nothing_on_the_ring():
    """
    #286 asked for no ring indicator and #66 is where one belongs — HA
    driving the idle ring can then show whatever the user wants. So the
    switch must not reach the LED paths at all, and `em_scenes` must not
    grow a colour for it that would later have to become configurable.
    """
    src = CONTROLLER.read_text()
    setter = _block(src, "async def _set_wake_word(", "        # Capabilities before")
    assert "await leds_" not in setter
    assert "send_led_anim" not in setter and "set_leds" not in setter
    scenes = (ROOT / "controller" / "em_scenes.py").read_text()
    assert "wake_word" not in scenes
    for fn in ("async def leds_off(", "async def _leds_turn_end("):
        assert "wake_word_enabled" not in _block(src, fn, "\n\n\n")


def test_a_private_wake_is_declined_with_the_wake_word_off():
    """
    Under private listening (#602, the default) the Echo scores the wake
    word itself and sends `oww_wake` with a session. The stream-path gates
    never see that, and the switch's mic_stop ends neither the session nor
    local listening — so without a check here, a wake with the switch off
    still starts a turn. It is closed with its own reason, so neither log
    blames the button.
    """
    src = CONTROLLER.read_text()
    turn = _block(src, "async def _private_wake_turn(", "\n\n\n")
    gate = turn.index("if not device.wake_word_enabled:")
    close = turn.index('await device.listen_close(session, "wake_off")', gate)
    assert close < turn.index("Wake word detected"), \
        "the wake must be declined before a turn is set up"
    assert turn.index('listen_close(session, "muted")') < gate, \
        "the button mute keeps its own reason and is checked first"
