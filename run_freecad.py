FORBIDDEN_TOKENS = [
    "Sketcher",
    "PartDesign",
    "Draft",
    "Import",
]

def detect_contract_violations(script: str) -> list[str]:
    violations = []
    for tok in FORBIDDEN_TOKENS:
        if tok in script:
            violations.append(tok)
    return violations
# run_freecad.py
import os
import subprocess
import time
import json
import pathlib
import textwrap
from dotenv import load_dotenv

dotenv_path = 'enviromental.env'
load_dotenv(dotenv_path)

# FreeCAD headless CLI (macOS App Bundle)
FREECAD_CMD = os.environ.get("FREECAD_CMD")


WORKDIR = pathlib.Path(os.environ.get("CAD_WORKDIR")).resolve()
WORKDIR.mkdir(parents=True, exist_ok=True)

# OCC renderer (x86_64 conda env python that can import OCC)
OCC_PY = os.environ.get("OCC_PY")
OCC_RENDER_SCRIPT = os.environ.get(
    "OCC_RENDER_SCRIPT",
    os.path.join(os.path.dirname(__file__), "occ_render.py"),
)


def _run_proc(args, cwd: str, timeout_s: int = 120) -> tuple[int, str, str, bool]:
    """Run a process robustly. Returns (returncode, stdout, stderr, killed_by_timeout)."""
    killed_by_timeout = False
    proc = subprocess.Popen(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        killed_by_timeout = True
        try:
            os.killpg(proc.pid, 9)
        except Exception:
            proc.kill()
        stdout, stderr = proc.communicate()
    return proc.returncode, stdout or "", stderr or "", killed_by_timeout


def run_freecad_script(script: str, run_id: str) -> dict:
    """
    Run a generated FreeCAD Python script in a per-run folder.

    Expected script behavior:
      - writes result.step into cwd
      - prints exactly one JSON line at the end (we parse it as checks)
    """
    safe = "".join(c for c in (run_id or "") if c.isalnum() or c in "-_")[:64] or "run"
    out_dir = WORKDIR / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = out_dir / "job.py"
    script_path.write_text(textwrap.dedent(script), encoding="utf-8")

    # Write the runner.py wrapper that executes job.py and saves result.FCStd
    runner_path = out_dir / "runner.py"
    runner_path.write_text(
        """
import os
import runpy
import traceback

# Execute the generated job script in-process so we can access the created FreeCAD document.
# IMPORTANT: job.py is expected to create geometry and (optionally) export result.step.

try:
    runpy.run_path("job.py", run_name="__main__")
except SystemExit:
    # Allow job scripts to call sys.exit()
    raise
except Exception:
    # Re-raise after printing, so return code indicates failure.
    traceback.print_exc()
    raise

# Save native FreeCAD document if any doc exists.
# We intentionally do NOT print anything here, to preserve the job's final JSON line on stdout.
try:
    import FreeCAD as App

    out_dir = os.getcwd()
    fcstd_path = os.path.join(out_dir, "result.FCStd")

    docs = list(App.listDocuments().values())
    doc = App.ActiveDocument if getattr(App, "ActiveDocument", None) else (docs[-1] if docs else None)

    if doc is not None:
        doc.saveAs(fcstd_path)
except Exception:
    # Non-fatal: do not break CAD run if FCStd save fails.
    traceback.print_exc()
""".lstrip(),
        encoding="utf-8",
    )

    violations = detect_contract_violations(script)
    if violations:
        return {
            "run_id": safe,
            "seconds": 0,
            "returncode": -1,
            "killed_by_timeout": False,
            "stdout": "",
            "stderr": f"CONTRACT_VIOLATION: forbidden tokens used: {violations}",
            "checks": None,
            "artifacts": ["job.py"],
            "out_dir": str(out_dir),
            "step_exists": False,
            "cad_ok": False,
            "had_exception": True,
            "fcstd_exists": False,
        }

    t0 = time.time()
    rc, stdout, stderr, killed = _run_proc(
        [FREECAD_CMD, str(runner_path)],
        cwd=str(out_dir),
        timeout_s=240,
    )
    dt = round(time.time() - t0, 2)

    # Parse checks from last JSON line on stdout
    checks = None
    try:
        lines = [ln for ln in (stdout or "").splitlines() if ln.strip()]
        if lines:
            obj = json.loads(lines[-1])
            checks = obj.get("checks") or obj
    except Exception:
        checks = None

    stdout_s = stdout or ""
    stderr_s = stderr or ""

    # Detect exceptions even if rc==0 (FreeCAD does that sometimes)
    had_exception = (
        "Exception while processing file" in stderr_s
        or "Traceback (most recent call last)" in stderr_s
        or "Exception:" in stderr_s
        or "Error:" in stderr_s
        or "Traceback (most recent call last)" in stdout_s
        or "Exception while processing file" in stdout_s
    )

    step_exists = (out_dir / "result.step").exists()
    fcstd_exists = (out_dir / "result.FCStd").exists()

    # Consider CAD ok only if: rc==0, no exception, and step exists
    cad_ok = (rc == 0) and (not had_exception) and step_exists

    artifacts = sorted([p.name for p in out_dir.iterdir() if p.is_file()])

    return {
        "run_id": safe,
        "seconds": dt,
        "returncode": rc,
        "killed_by_timeout": killed,
        "stdout": stdout_s,
        "stderr": stderr_s,
        "checks": checks,
        "artifacts": artifacts,
        "out_dir": str(out_dir),
        "step_exists": step_exists,
        "cad_ok": cad_ok,
        "had_exception": had_exception,
        "fcstd_exists": fcstd_exists,
    }


def run_occ_render_images(step_path: str, out_dir: str) -> dict:
    """
    Render 4 views from STEP headless via pythonocc-core (x86_64 env).

    Expects occ_render.py to print ONE JSON line at the end:
      {"success": bool, "images": {...}, "bbox": [...], "volume": float, "error": str|null}

    Returns:
      {
        success, images, bbox, volume, error,
        returncode, stdout, stderr, killed_by_timeout
      }
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # On Apple Silicon: force x86_64 execution for pythonocc-core osx-64 env
    cmd = ["arch", "-x86_64", OCC_PY, OCC_RENDER_SCRIPT, step_path, str(out_dir)]

    rc, stdout, stderr, killed = _run_proc(
        cmd,
        cwd=str(out_dir),
        timeout_s=240,
    )

    payload = None
    try:
        lines = [ln for ln in (stdout or "").splitlines() if ln.strip()]
        if lines:
            payload = json.loads(lines[-1])
    except Exception:
        payload = None

    if not isinstance(payload, dict):
        return {
            "success": False,
            "returncode": rc,
            "stdout": stdout,
            "stderr": stderr,
            "killed_by_timeout": killed,
            "images": None,
            "bbox": None,
            "volume": None,
            "error": "OCC renderer returned no valid JSON on stdout",
        }

    return {
        "success": bool(payload.get("success")),
        "returncode": rc,
        "stdout": stdout,
        "stderr": stderr,
        "killed_by_timeout": killed,
        "images": payload.get("images") or {},
        "bbox": payload.get("bbox"),
        "volume": payload.get("volume"),
        "error": payload.get("error"),
    }


# Backward-compatible alias (if something still imports run_freecad_render_images)
def run_freecad_render_images(step_path: str, out_dir: str) -> dict:
    return run_occ_render_images(step_path, out_dir)
