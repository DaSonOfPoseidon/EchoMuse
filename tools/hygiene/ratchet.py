# A lint gate that existing findings do not fail and new ones do (#822).
#
#   python tools/hygiene/ratchet.py <name> [--update] [--cwd DIR] -- <command...>
#
# Runs the command, reads each finding as (file, rule), and compares the count
# per (file, rule) with tools/hygiene/baseline/<name>.txt. A count that rises,
# or a pair that is new, fails. A count that FALLS also fails, with the
# command that lowers the baseline: otherwise the baseline stays where it was,
# and the next regression in that file passes against room somebody else's
# fix made.
#
# Keyed by file rather than by line, so moving code does not break it, and by
# file rather than as one total, so a fix in one file cannot pay for a new
# finding in another.
#
# Output the parser does not recognise fails the gate rather than being
# skipped. A tool that could not load a package prints an error and no
# findings, and that must not read as a clean tree.

import argparse
import collections
import pathlib
import re
import subprocess
import sys

BASELINE_DIR = pathlib.Path(__file__).resolve().parent / "baseline"

# One pattern per tool's plain-text output; each yields (path, rule).
FORMATS = {
    # internal/x/y.go:12:3: message (U1000)
    "staticcheck": re.compile(
        r"^(?P<path>[^:\s]+\.go):\d+:\d+: .* \((?P<rule>[A-Z]+\d+)\)$"
    ),
    # controller/em_x.py:12:3: BLE001 Do not catch blind exception: `Exception`
    "ruff": re.compile(r"^(?P<path>[^:\s]+):\d+:\d+: (?P<rule>[A-Z]+\d+)\b"),
    # controller/em_x.py:12: error: message  [arg-type]
    "mypy": re.compile(
        r"^(?P<path>[^:\s]+):\d+(?::\d+)?: error: .*\[(?P<rule>[a-z-]+)\]$"
    ),
}

# Lines a tool prints that are neither a finding nor a failure.
NOISE = {
    "ruff": re.compile(
        r"^(Found \d+ errors?\.|All checks passed!|\[\*\] .*fixable.*|No fixes available.*)$"
    ),
    "mypy": re.compile(
        r"^(Found \d+ errors? in \d+ files? .*|Success: .*|.*: note: .*)$"
    ),
}

Key = tuple[str, str]


def parse(fmt: str, text: str) -> tuple[collections.Counter[Key], list[str]]:
    pattern, noise = FORMATS[fmt], NOISE.get(fmt)
    found: collections.Counter[Key] = collections.Counter()
    unknown = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            found[(m["path"], m["rule"])] += 1
        elif not (noise and noise.match(line)):
            unknown.append(line)
    return found, unknown


def read_baseline(path: pathlib.Path) -> collections.Counter[Key]:
    counts: collections.Counter[Key] = collections.Counter()
    if not path.exists():
        return counts
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        n, file, rule = line.split("\t")
        counts[(file, rule)] = int(n)
    return counts


def write_baseline(path: pathlib.Path, counts: collections.Counter[Key]) -> None:
    lines = [
        "# count<TAB>file<TAB>rule. Written by tools/hygiene/ratchet.py --update;",
        "# lower it by fixing findings, never raise it by hand.",
    ]
    lines += [f"{n}\t{f}\t{r}" for (f, r), n in sorted(counts.items()) if n > 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def compare(
    baseline: collections.Counter[Key], found: collections.Counter[Key]
) -> tuple[list[str], list[str]]:
    worse, better = [], []
    for key in sorted(set(baseline) | set(found)):
        was, now = baseline.get(key, 0), found.get(key, 0)
        if now > was:
            worse.append(f"{key[0]}: {key[1]} {was} -> {now}")
        elif now < was:
            better.append(f"{key[0]}: {key[1]} {was} -> {now}")
    return worse, better


def main(argv: list[str]) -> int:
    if "--" not in argv:
        print(
            "usage: ratchet.py <name> [--update] [--cwd DIR] -- <command...>",
            file=sys.stderr,
        )
        return 2
    split = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--format", help="output format, defaults to the name's prefix")
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args(argv[:split])
    command = argv[split + 1 :]
    fmt = args.format or args.name.split("-")[0]
    if fmt not in FORMATS:
        print(f"unknown format {fmt!r}; known: {', '.join(FORMATS)}", file=sys.stderr)
        return 2

    proc = subprocess.run(command, cwd=args.cwd, capture_output=True, text=True)
    found, unknown = parse(fmt, proc.stdout + proc.stderr)
    if unknown:
        print("\n".join(unknown))
        print(
            f"::error::{args.name}: output that is not a finding -- the tool may have failed"
        )
        return 1
    # Every tool here exits non-zero when it reports findings, so only an
    # exit with nothing reported is suspicious. A crash that printed nothing
    # would otherwise pass as a clean tree.
    if proc.returncode != 0 and not found:
        print(
            f"::error::{args.name}: exited {proc.returncode} without reporting a finding"
        )
        return 1

    path = BASELINE_DIR / f"{args.name}.txt"
    if args.update:
        write_baseline(path, found)
        print(f"{path}: {sum(found.values())} findings recorded")
        return 0

    worse, better = compare(read_baseline(path), found)
    if worse:
        print((proc.stdout + proc.stderr).rstrip())
        print("\nNew findings, by file and rule:\n  " + "\n  ".join(worse))
        print(
            f"::error::{args.name}: new findings -- fix them; the baseline does not go up"
        )
        return 1
    if better:
        print("Fewer findings than the baseline records:\n  " + "\n  ".join(better))
        print(
            f"::error::{args.name}: lower the baseline to keep the gain -- run: "
            f"python tools/hygiene/ratchet.py {' '.join(argv[:split])} --update -- {' '.join(command)}"
        )
        return 1
    print(f"{args.name}: {sum(found.values())} findings, none new")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
