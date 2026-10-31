"""Run every lab in algorithms/ and report pass/fail and timing."""
import pathlib, subprocess, sys, time

root = pathlib.Path(__file__).resolve().parents[1] / "algorithms"
only = sys.argv[1:]  # optional filters, e.g. python scripts/run_all.py 01 07
fails = []
for script in sorted(root.glob("*/*.py")):
    if only and not any(script.parent.name.startswith(o) for o in only):
        continue
    t = time.time()
    r = subprocess.run([sys.executable, script.name], cwd=script.parent, capture_output=True, text=True)
    ok = r.returncode == 0
    print(f"{'PASS' if ok else 'FAIL'}  {time.time() - t:5.1f}s  {script.parent.name}/{script.name}", flush=True)
    if not ok:
        fails.append((script, r.stderr[-800:]))
for s, e in fails:
    print("\n---", s, "\n", e)
print(f"\n{len(fails)} failed")
sys.exit(1 if fails else 0)
