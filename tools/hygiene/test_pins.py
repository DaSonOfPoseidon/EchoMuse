# A tool pinned in two places drifts, and then pre-commit passes a change
# that CI fails (#822). Every tool .pre-commit-config.yaml installs from PyPI
# must be pinned to the same version as the `pip install` that runs it in
# ci.yml.

import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
PIN = re.compile(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)")


def precommit_pins():
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    pins = {}
    for repo in config["repos"]:
        for hook in repo["hooks"]:
            for dep in hook.get("additional_dependencies", []):
                m = PIN.fullmatch(dep)
                if m:
                    pins[m[1].lower()] = m[2]
    return pins


def ci_pins():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    pins = {}
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            for line in str(step.get("run", "")).splitlines():
                if "pip install" in line:
                    for name, version in PIN.findall(line):
                        pins.setdefault(name.lower(), set()).add(version)
    return pins


def test_every_hook_tool_is_pinned_the_same_in_ci():
    hooks, ci = precommit_pins(), ci_pins()
    assert hooks, "no pinned hook dependencies found -- has the file moved?"
    for name, version in hooks.items():
        if name in ("numpy", "scipy"):
            # Not a lint tool: CI takes these from controller/requirements.txt.
            continue
        assert name in ci, f"{name} is pinned in .pre-commit-config.yaml and installed nowhere in ci.yml"
        assert ci[name] == {version}, f"{name}: pre-commit {version}, ci.yml {sorted(ci[name])}"


def test_numpy_matches_the_controller_pin():
    req = (ROOT / "controller/requirements.txt").read_text()
    m = re.search(r"^numpy==(\S+)", req, re.M)
    assert m and precommit_pins()["numpy"] == m[1]
