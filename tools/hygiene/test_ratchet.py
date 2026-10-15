# The ratchet decides whether CI is red, so its rules are tested from the
# edges: a rise, a fall, a new file, a deleted file, output it does not
# recognise, and a tool that exits without saying why.

import collections
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ratchet  # noqa: E402


@pytest.fixture
def baseline_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ratchet, "BASELINE_DIR", tmp_path)
    return tmp_path


def emit(lines, code=1):
    # A command that prints `lines` and exits with `code`, without depending
    # on any lint tool being installed.
    script = "import sys; sys.stdout.write(sys.argv[1]); sys.exit(int(sys.argv[2]))"
    return ["--", sys.executable, "-c", script, "\n".join(lines), str(code)]


SC_A = "internal/a/a.go:10:2: var x is unused (U1000)"
SC_A2 = "internal/a/a.go:20:2: var y is unused (U1000)"
SC_B = "internal/b/b.go:5:1: should use time.Until (S1024)"


def run(name, lines, code=1, update=False):
    argv = [name] + (["--update"] if update else []) + emit(lines, code)
    return ratchet.main(argv)


def test_unchanged_counts_pass(baseline_dir):
    assert run("staticcheck", [SC_A, SC_B], update=True) == 0
    assert run("staticcheck", [SC_A, SC_B]) == 0


def test_a_rise_in_one_file_fails(baseline_dir):
    run("staticcheck", [SC_A], update=True)
    assert run("staticcheck", [SC_A, SC_A2]) == 1


def test_a_fix_elsewhere_does_not_pay_for_a_new_finding(baseline_dir):
    # Same total, different file: still a new finding.
    run("staticcheck", [SC_A, SC_A2], update=True)
    assert run("staticcheck", [SC_A, SC_B]) == 1


def test_a_new_file_with_findings_fails(baseline_dir):
    run("staticcheck", [SC_A], update=True)
    assert run("staticcheck", [SC_A, SC_B]) == 1


def test_a_fall_fails_until_the_baseline_is_lowered(baseline_dir, capsys):
    run("staticcheck", [SC_A, SC_A2], update=True)
    assert run("staticcheck", [SC_A]) == 1
    assert "--update" in capsys.readouterr().out
    assert run("staticcheck", [SC_A], update=True) == 0
    assert run("staticcheck", [SC_A]) == 0


def test_a_deleted_file_drops_out_of_the_baseline(baseline_dir):
    run("staticcheck", [SC_A, SC_B], update=True)
    run("staticcheck", [SC_A], update=True)
    text = (baseline_dir / "staticcheck.txt").read_text()
    assert "internal/b/b.go" not in text
    assert run("staticcheck", [SC_A]) == 0


def test_no_baseline_means_zero_allowed(baseline_dir):
    assert run("staticcheck", [], code=0) == 0
    assert run("staticcheck", [SC_A]) == 1


def test_unrecognised_output_fails(baseline_dir):
    # What staticcheck prints when a package does not load: no finding, and
    # it must not read as a clean tree.
    broken = "internal/x/x.go:13:2: could not import foo (compile)"
    run("staticcheck", [], code=0, update=True)
    assert run("staticcheck", [broken]) == 1


def test_a_silent_failure_fails(baseline_dir):
    run("staticcheck", [], code=0, update=True)
    assert run("staticcheck", [], code=2) == 1


def test_baseline_round_trip(tmp_path):
    counts = collections.Counter({("a.py", "F401"): 2, ("b.py", "BLE001"): 1})
    ratchet.write_baseline(tmp_path / "x.txt", counts)
    assert ratchet.read_baseline(tmp_path / "x.txt") == counts


@pytest.mark.parametrize(
    "fmt,line,key",
    [
        ("staticcheck", SC_B, ("internal/b/b.go", "S1024")),
        ("ruff", "controller/em_x.py:3:1: BLE001 Do not catch blind exception", ("controller/em_x.py", "BLE001")),
        ("mypy", "controller/em_x.py:3: error: Bad thing  [arg-type]", ("controller/em_x.py", "arg-type")),
        ("mypy", "controller/em_x.py:3:7: error: Bad thing  [union-attr]", ("controller/em_x.py", "union-attr")),
    ],
)
def test_each_format_parses(fmt, line, key):
    found, unknown = ratchet.parse(fmt, line)
    assert found == collections.Counter({key: 1}) and unknown == []


@pytest.mark.parametrize(
    "fmt,line",
    [
        ("ruff", "Found 3 errors."),
        ("ruff", "All checks passed!"),
        ("mypy", "Found 2 errors in 1 file (checked 9 source files)"),
        ("mypy", "Success: no issues found in 9 source files"),
        ("mypy", "controller/em_x.py:3: note: See https://mypy.rtfd.io"),
    ],
)
def test_summary_lines_are_not_findings(fmt, line):
    assert ratchet.parse(fmt, line) == (collections.Counter(), [])
