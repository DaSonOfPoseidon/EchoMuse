import version


def test_env_var_wins(monkeypatch):
    monkeypatch.setenv("EM_CONTROLLER_VERSION", "v9.9.9")
    assert version._resolve() == "v9.9.9"


def test_git_describe_strips_controller_prefix(monkeypatch):
    monkeypatch.delenv("EM_CONTROLLER_VERSION", raising=False)
    v = version._resolve()
    # In a checkout this is the described tag; in a bare copy it's "dev".
    # Either way the controller- prefix must never leak into the display.
    assert not v.startswith("controller-")
    assert v  # never empty


def test_no_env_no_git_is_dev(monkeypatch):
    monkeypatch.delenv("EM_CONTROLLER_VERSION", raising=False)
    monkeypatch.setattr(
        version.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no git")),
    )
    assert version._resolve() == "dev"


# ─── Controller update comparison ────────────────────────────────────────────
#
# The controller can only tell you an update exists; applying it is a
# `docker compose pull` the user performs. That makes a WRONG answer here
# unusually expensive — there is no button to press that would reveal the
# mistake, so a bad comparison just quietly misinforms.

import pytest


@pytest.mark.parametrize(
    "text,expected",
    [
        ("v2.10.0", (2, 10, 0)),
        ("controller-v2.10.0", (2, 10, 0)),
        ("2.10.0", (2, 10, 0)),
        ("v2.10.0-3-gabc1234", (2, 10, 0)),
        ("v2.10.0-3-gabc1234-dirty", (2, 10, 0)),
        ("dev", None),
        ("", None),
        ("v", None),
    ],
)
def test_parse_handles_every_form_the_resolver_can_produce(text, expected):
    assert version.parse(text) == expected


def test_newer_tag_is_an_update():
    assert version.compare("v2.10.0", "v2.11.0") == {
        "status": "update",
        "available": True,
    }


def test_tags_sort_numerically_not_lexically():
    """
    v2.9.0 vs v2.10.0 is the case a string compare gets backwards, and the
    GitHub refs API returns tags in lexical order — so this is not a
    hypothetical, it is the ordering the caller is handed.
    """
    assert version.parse("v2.10.0") > version.parse("v2.9.0")
    assert version.compare("v2.9.0", "v2.10.0")["available"] is True
    assert version.compare("v2.10.0", "v2.9.0")["available"] is False


def test_running_the_newest_tag_is_not_an_update():
    assert version.compare("v2.11.0", "v2.11.0") == {
        "status": "current",
        "available": False,
    }


def test_a_build_ahead_of_the_tag_is_not_an_update():
    """
    A bare-metal checkout between tags describes as v2.11.0-3-gabc1234, whose
    base parses EQUAL to v2.11.0. It is ahead, not behind — nagging a
    developer to 'update' to the tag they are already past is the bug.
    """
    assert version.compare("v2.11.0-3-gabc1234", "v2.11.0")["available"] is False
    assert version.compare("v2.11.0-3-gabc1234-dirty", "v2.11.0")["available"] is False


def test_an_unknown_running_version_never_claims_to_be_up_to_date():
    """
    "dev" (no env var, no git) cannot be compared. Reporting "current" would
    be the answer that stops someone looking; unknown is the honest one.
    """
    result = version.compare("dev", "v2.11.0")
    assert result["status"] == "unknown"
    assert result["available"] is False
