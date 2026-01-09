# main3.1.py
import os
import json
import re
import asyncio
import base64
import csv
import hashlib
import sys
import ast
import shutil
import argparse
from datetime import datetime

from dotenv import load_dotenv
from azure.identity.aio import AzureCliCredential

from agent_framework import ChatMessage, TextContent, DataContent, Role
from agent_framework.azure import AzureAIClient

version = "01" # V34 base with  bugfixing agent

os.environ["CAD_WORKDIR"] = f"./cad_runs_v{version}"
from run_freecad import run_freecad_script, run_occ_render_images



dotenv_path = 'enviromental.env'
load_dotenv(dotenv_path)


# ----------------------------
# Runtime safety (timeouts/retries/heartbeats)
# ----------------------------

LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "180"))
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "3"))
LLM_RETRY_BACKOFF_S = float(os.getenv("LLM_RETRY_BACKOFF_S", "2"))

def _ts_local() -> str:
    # human readable local timestamp for console logs
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def call_agent(agent, prompt, *, label: str, timeout_s: int | None = None, retries: int | None = None):
    """Call an agent with a hard timeout + retries.

    Why: In long batch runs, network/API calls can occasionally stall without raising,
    which looks like the script "stops". This wrapper makes those failures visible
    and recoverable.
    """
    timeout_s = int(timeout_s or LLM_TIMEOUT_S)
    retries = int(retries or LLM_RETRIES)

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[{_ts_local()}] LLM call start: {label} (attempt {attempt}/{retries}, timeout={timeout_s}s)")
            sys.stdout.flush()
            res = await asyncio.wait_for(agent.run(prompt), timeout=timeout_s)
            print(f"[{_ts_local()}] LLM call ok: {label} (attempt {attempt}/{retries})")
            sys.stdout.flush()
            return res
        except asyncio.TimeoutError as e:
            last_exc = e
            print(f"[{_ts_local()}] LLM call TIMEOUT: {label} (attempt {attempt}/{retries})")
            sys.stdout.flush()
        except Exception as e:
            last_exc = e
            print(f"[{_ts_local()}] LLM call ERROR: {label} (attempt {attempt}/{retries}) -> {type(e).__name__}: {e}")
            sys.stdout.flush()

        # backoff before retry
        if attempt < retries:
            await asyncio.sleep(LLM_RETRY_BACKOFF_S * attempt)

    # Exhausted
    raise RuntimeError(f"LLM call failed after {retries} attempts: {label}") from last_exc

# ----------------------------
# Agent instructions
# ----------------------------

ARCHITECT_INSTRUCTIONS = """
You are an experienced designer (Senior CAD Engineer) for FreeCAD.
Respond ONLY with valid JSON (no Markdown, no explanations).

Goal: Produce a robust, parametric design and feature plan.
You do NOT write FreeCAD code. You decide feature order, references (datums), symmetries, patterns, and parameters.

EXACT output format:
{
  "run_id": "<only a-z0-9- max 20, start/end alphanumeric>",
  "intent": "<short type, e.g., bracket|plate|tube|wheel-rim|housing|gear-like|generic>",
  "params": {"key": "value", "...": "..."},
  "datums": ["..."],
  "feature_plan": [
    {
      "id": "f1",
      "type": "base|cut|pattern|detail",
      "op": "pad|revolve|extrude|pocket|hole|boolean_cut|fillet|chamfer|draft",
      "sketch": "<name or null>",
      "ref": "<datum/face/axis name>",
      "dims": {"key": "value"},
      "notes": "<brief>"
    }
  ],
  "acceptance": {
    "expected_bbox": null,
    "must_have": ["..."]
  },
  "plan": ["<short construction plan in steps>"]
}

Rules / designer logic:
- Think in feature tree/order: Base -> main cuts -> patterns -> details.
- Robustness: prefer sketch+pad/pocket/hole/pattern over many booleans.
  - If unclear: allow boolean_cut only as a last resort and sparingly.
- Parameterize everything important (dimensions, count, bolt circle, wall thicknesses, radii, symmetry).
- If the user is vague ("car wheel"), choose plausible default parameters and list them in params.
- For circular/bolt patterns, use parameters: n, pcd/bolt_circle_diameter, radius.
- Units: mm.
- IMPORTANT: All values in params/dims/acceptance must be JSON literals (numbers/strings/bool/null). NO expressions like "112 + 30"; compute it first (e.g., 142).
- run_id strict: [a-z0-9-], max 20.
"""

IMPLEMENTER_INSTRUCTIONS = """
You are a FreeCAD implementer (CAD scripter). You receive a feature plan (JSON) from the designer.
Your task: produce FreeCAD Python code that robustly implements this plan.

Respond ONLY with valid JSON (no Markdown, no explanations):
{
  "plan": ["<step>", "<step>", "..."],
  "run_id": "<only a-z0-9- max 20>",
  "script": "<FreeCAD Python Script>"
}

Requirements for "script" (MUST satisfy all):
- Only these imports (exactly these four lines, in exactly this order):
  import FreeCAD as App
  import Part
  import os, json
  import math
- Must create a document:
  doc = App.newDocument("Model")
- Must keep the result in variable "solid".
- Must create exactly ONE object (name: Result):
  obj = doc.addObject("Part::Feature","Result"); obj.Shape = solid; doc.recompute()
- Must export STEP (in cwd):
  out_path = os.path.join(os.getcwd(),"result.step")
  obj.Shape.exportStep(out_path)
- Must print EXACTLY ONE JSON line at the end, and it MUST look like this:
  print(json.dumps({"checks": {"bbox": [bb.XMin,bb.YMin,bb.ZMin,bb.XMax,bb.YMax,bb.ZMax], "volume": float(obj.Shape.Volume)}}))
  (bb is obj.Shape.BoundBox)
  - Do NOT use the variable name `step_path` in the FreeCAD script. The export path is strictly `out_path` (os.getcwd()/result.step) and is set only in the footer.
- Avoid `Base.*` types (e.g., Base.Placement/Base.Matrix). Use only Part shapes + Shape.rotate(...) + App.Vector(...).

Rules:
API CHEATSHEET (FreeCAD 1.0.x, allowed in this runner):
You may use ONLY Part + App.Vector + Shape.rotate. Use NO workbenches (no Sketcher/PartDesign/Draft/Import).

Allowed primitives (positional args only, no keyword args!):
- Part.makeBox(l, w, h)
- Part.makeCylinder(radius, height)
- Part.makeCone(r1, r2, height)
- Part.makeSphere(radius)
- Part.makeTorus(r1, r2)

Allowed boolean/shape operations (on shapes):
- solid = solid.fuse(other)        # union
- solid = solid.cut(other)         # subtract
- solid = solid.common(other)      # intersection (rare)
- solid = solid.copy()             # if needed

Positioning/rotation (without Base.*):
- other.translate(App.Vector(x,y,z))
- other.rotate(App.Vector(0,0,0), App.Vector(ax,ay,az), deg)

Sketch/hole/pattern logic (how to translate the feature plan):
- pad/extrude -> Part.makeBox(...) or Part.makeCylinder(...)
- hole/pocket/boolean_cut -> Part.makeCylinder(...); solid = solid.cut(hole)
- pattern (circular) -> loop over angles: create hole, translate, cut
- fillet/chamfer -> OMIT (only if explicitly needed and stable)

IMPORTANT: Variable discipline (prevents 80% of your bugs):
- ALWAYS keep exactly one running result variable: `solid`.
- At the start of the script, define ONE param dict `P = {...}` (numbers/strings/bool/null only).
  - Afterwards, use dimensions only as `P["..."]` or as literals.
  - Avoid invented variable names like `ear_hole_offset_z`, `channel_y_positions`, etc.
- ALWAYS create helper geometry as a local variable (e.g., `tool`) right before use.
- No access to variables before they are assigned (no free variables!).
- No additional imports (beyond the 4 above).
- No file paths outside os.getcwd().
- Units: mm.
- If angles/trigonometry are needed: use only `math`:
  - Degrees -> radians: `rad = math.radians(deg)`
  - `math.cos(rad)`, `math.sin(rad)`
  - Use NO `App.cos`, `App.sin`, `App.pi`, NO `App.Units.Quantity(...)`
  - Use NO Vector.rotate()
- For bolt circles / circular patterns: ONLY this pattern:
    rad = math.radians(deg)
    x = r * math.cos(rad)
    y = r * math.sin(rad)
    pos = App.Vector(x, y, 0)
- For shape rotations: ONLY Shape.rotate(...), e.g.:
    shp.rotate(App.Vector(0,0,0), App.Vector(0,0,1), deg)

IMPORTANT:
- Prefer stable primitives + cuts (cylinders/rings/boxes) and patterns.
- If an ellipse/oval is difficult: use a capsule slot (box + 2 cylinders) instead of an ellipse.
- No fillets if they often break: start without fillet, then optionally add as the last step.
- Do NOT use the variable name `step_path` in the FreeCAD script. The export path is strictly `out_path` (os.getcwd()/result.step) and is set only in the footer.
- Avoid `Base.*` types (e.g., Base.Placement/Base.Matrix). Use only Part shapes + Shape.rotate(...) + App.Vector(...).
=== LESSONS-LEARNED COMPLIANCE (HARD RULES) ===

You also receive a list "LESSONS LEARNED (persisted from previous runs)".

MANDATORY:
- Analyze these lessons BEFORE writing the script.
- Identify the listed root causes and suggestions.
- Your script MUST NOT contain any of those errors again.

IN PARTICULAR (not exhaustive):
- Never use the variable `step_path`.
- Never access variables before they are defined.
- Do NOT use FreeCAD APIs marked in lessons as "not available" or "version-incompatible" (e.g., makePolyline).
- Do NOT use keyword arguments for Part.makeSphere / Part.makeCylinder etc., if lessons forbid them.
- Do NOT use Base.Placement / Base.Matrix APIs if lessons list them as a root cause.

IMPORTANT:
If your script includes a known lesson error, your answer is INVALID,
even if it appears syntactically correct.
"""
# --- Undefined-name preflight (prevents NameError before FreeCAD run) ---

_ALLOWED_GLOBAL_NAMES = {
    # mandatory imports
    "App", "Part", "os", "json", "math",
    # runner contract variables
    "doc", "solid", "obj", "bb", "out_path",
    # common loop / util names
    "range", "len", "float", "int", "str", "min", "max", "abs", "sum",
    # typical local names we allow if author defines them; (kept minimal)
}

class _NameUseCollector(ast.NodeVisitor):
    def __init__(self):
        self.assigned: set[str] = set()
        self.used: list[tuple[str, int]] = []  # (name, lineno)

    def visit_Import(self, node):
        for alias in node.names:
            if alias.asname:
                self.assigned.add(alias.asname)
            else:
                # import X -> binds X
                self.assigned.add(alias.name.split(".")[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.assigned.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node):
        self.assigned.add(node.name)
        for arg in node.args.args:
            self.assigned.add(arg.arg)
        for arg in getattr(node.args, "posonlyargs", []):
            self.assigned.add(arg.arg)
        for arg in node.args.kwonlyargs:
            self.assigned.add(arg.arg)
        if node.args.vararg:
            self.assigned.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.assigned.add(node.args.kwarg.arg)
        self.generic_visit(node)

    def visit_Assign(self, node):
        for t in node.targets:
            self._collect_assigned_target(t)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._collect_assigned_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._collect_assigned_target(node.target)
        self.generic_visit(node)

    def visit_For(self, node):
        self._collect_assigned_target(node.target)
        self.generic_visit(node)

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars is not None:
                self._collect_assigned_target(item.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.assigned.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        # record usage sites for Load
        if isinstance(node.ctx, ast.Load):
            self.used.append((node.id, getattr(node, "lineno", 0) or 0))
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.assigned.add(node.id)

    def _collect_assigned_target(self, t):
        if isinstance(t, ast.Name):
            self.assigned.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                self._collect_assigned_target(elt)
        # ignore attributes/subscripts (obj.x, a[i])


def validate_no_undefined_names(script: str) -> tuple[bool, list[str], list[str]]:
    """Return (ok, issues, undefined_names).

    Detects variables used before being defined at module scope, which often leads to NameError
    in FreeCAD runs. This is intentionally conservative and allows common builtins.
    """
    if not isinstance(script, str) or not script.strip():
        return False, ["script empty"], []

    try:
        tree = ast.parse(script)
    except Exception as e:
        # Syntax handled elsewhere; keep this non-blocking but visible
        return False, [f"cannot parse script for undefined-name check: {type(e).__name__}: {e}"], []

    c = _NameUseCollector()
    c.visit(tree)

    # Add Python builtins we allow (conservative)
    allowed = set(_ALLOWED_GLOBAL_NAMES)
    allowed.update(dir(__builtins__))

    # Anything used (Load) that isn't assigned/imported/builtin is considered undefined.
    undefined: dict[str, int] = {}
    for name, lineno in c.used:
        if name in allowed:
            continue
        if name in c.assigned:
            continue
        # ignore dunder names
        if name.startswith("__") and name.endswith("__"):
            continue
        undefined.setdefault(name, lineno)

    if not undefined:
        return True, [], []

    # Build readable issues
    items = sorted(undefined.items(), key=lambda x: (x[1], x[0]))
    undefined_names = [n for n, _ in items]
    issues = [f"undefined name '{n}' used (line {ln})" for n, ln in items[:25]]
    return False, issues, undefined_names

EVALUATOR_INSTRUCTIONS = """
You are a geometric CAD reviewer with vision.

You receive:
- a JSON payload (text)
- 4 images as PNG (iso/front/right/top)

Your task: Assess whether the CAD result matches the prompt.

IMPORTANT:
- Respond ONLY with valid JSON.
- Respond ONLY with EXACTLY these keys (no others!):
  status, notes, issues, expected_dims, observed_dims, observed_bbox
- Return NO plan, NO run_id, NO script.

Format (exactly like this):
{
  "status": "PASS" | "FAIL",
  "notes": "<string>",
  "issues": ["..."],
  "expected_dims": null,
  "observed_dims": null,
  "observed_bbox": null
}

Notes:
- Use the payload (bbox/volume/render_bbox/render_volume) + images for plausibility checks.
- observed_bbox can mirror render_bbox or bbox from the payload.
- expected_dims can be e.g. {"radius":400,"thickness":5} (if derivable), otherwise null.
- observed_dims can be a rough estimate or null.
"""

REPLANNER_INSTRUCTIONS = """
You are a designer replanner.
You receive user_prompt + last_eval + last_geom_eval + last_arch_spec.

Your task: Correct the feature plan (not FreeCAD code).
Respond ONLY with valid JSON in the EXACT format of the ARCHITECT (same keys!).

Rules:
- run_id only [a-z0-9-], max 20.
- Make targeted changes: add missing features, correct dimensions, shift strategy if needed.
- IMPORTANT (Contract/API fails): If last_eval.notes or last_eval.issues point to script contract violations (e.g., "violates the runner contract", "forbidden token", "missing exact 4-line import block", "Sketcher", "PartDesign", "Import.export", "/tmp"), then this is NOT a geometry problem.
  In that case you must simplify the feature plan so the implementer can robustly execute it using Part primitives + boolean cuts:
  - prefer: cylinder/ring (cylinder minus inner cylinder), box, fuse, cut
  - do not assume Sketches/PartDesign pads/pockets/holes/patterns in the plan
  - describe elliptical/oval cutouts as a "capsule slot" (box + 2 cylinders)
  - fillets only optional and as the last step (or omit entirely)
  Goal: a plan that is stable in the pure `Part` workflow.

- Additional rule (geometry memory):
  - `last_geom_eval` is the last evaluator feedback from a run that reached the evaluator (render/evaluate).
  - If `last_eval` is a script/contract/runtime/render failure (i.e., NOT real geometry feedback), then for geometry corrections prefer `last_geom_eval` and use `last_eval` only for robustness/strategy changes.

Use last_failure_phase:
 - IMPLEMENTATION_CONTRACT / CAD_RUNTIME / RENDER: simplify the feature plan (primitives + cuts), NO dimension changes.
 - GEOMETRY_MISMATCH: correct dimensions / missing features.
 - NONE: replan normally.
"""
# --- Failure phase classification helper ---
def classify_failure_phase(last_eval: dict) -> str:
    """
    Classify the failure phase for targeted replanning.
    Returns one of: "NONE", "IMPLEMENTATION_CONTRACT", "CAD_RUNTIME", "RENDER", "GEOMETRY_MISMATCH"
    """
    if not last_eval:
        return "NONE"
    status = str(last_eval.get("status", "")).strip().upper()
    if status == "PASS":
        return "NONE"
    notes = (last_eval.get("notes") or "").lower()
    issues = [str(x).lower() for x in (last_eval.get("issues") or [])]
    # Implementation contract errors
    contract_keywords = ["contract", "runner", "import", "solid", "syntax"]
    if any(any(k in (notes or "") for k in contract_keywords) or any(k in (iss or "") for k in contract_keywords) for iss in issues):
        return "IMPLEMENTATION_CONTRACT"
    # CAD runtime errors
    cad_keywords = ["cad error", "exception", "runtime", "freecad"]
    if any(any(k in (iss or "") for k in cad_keywords) for iss in issues):
        return "CAD_RUNTIME"
    # Render errors
    if any("render" in (iss or "") for iss in issues):
        return "RENDER"
    # Geometry mismatch (default)
    return "GEOMETRY_MISMATCH"

DEBUGGER_INSTRUCTIONS = """
You are a FreeCAD script debugger.
You receive stderr/stdout and the current prompt/plan.
Respond ONLY with JSON:
{
  "root_cause": "<short>",
  "fix_type": "api" | "strategy" | "params",
  "suggestions": ["..."]
}

Rules:
- Do not output code.
- Focus: why the FreeCAD run failed.
  - If the script contract is violated (wrong imports, PartDesign/Sketcher/Draft/Import used, export not via os.getcwd()/result.step, missing 'solid', missing Result line, wrong print(json.dumps(...))): then fix_type MUST be "api".
  - If geometry/booleans are unstable: fix_type="strategy".
  - If dimensions/parameters are nonsensical: fix_type="params".
- Provide concrete, actionable guidance for the designer replanner/implementer.
"""

REPAIRER_INSTRUCTIONS = """
You are a FreeCAD engineering repair agent.

Goal: minimally repair an EXISTING FreeCAD script so it runs again.
You must NOT generate a completely new script.

You receive:
- failure_class: one of: SYNTAX | UNDEFINED_NAME | CONTRACT | CAD_RUNTIME
- policy: allows/forbids certain patch scopes
- script: the current FreeCAD script (as a string)
- errors: list of errors/issues (strings)
- stderr/stdout/root_cause_line (optional)

You MUST respond ONLY with valid JSON (no Markdown, no explanations):
{
  "failure_class": "SYNTAX|UNDEFINED_NAME|CONTRACT|CAD_RUNTIME",
  "edits": [
    {
      "op": "replace",
      "old": "<exact substring to replace>",
      "new": "<replacement>"
    },
    {
      "op": "insert_after",
      "anchor": "<exact substring anchor>",
      "text": "<text to insert after anchor>"
    },
    {
      "op": "delete",
      "old": "<exact substring to delete>"
    }
  ],
  "notes": "<short>"
}

PATCH POLICY (HARD):
A) failure_class = SYNTAX or UNDEFINED_NAME
- Allowed: only minimal text edits that fix syntax/indent/imports/undefined names.
- Forbidden: change geometry/parameters/feature strategy (no new features, no dimension changes).
- Typical fixes: fix indentation, define/replace variables, correct wrong names, fix missing/extra brackets/quotes.

B) failure_class = CONTRACT
- Allowed: only contract glue fixes (imports/doc/footer/export/print), no geometry changes.

C) failure_class = CAD_RUNTIME
- Allowed: small strategy fixes, but ONLY locally:
  - Change boolean order (e.g., fuse tools and cut once)
  - Slightly increase margin/clearance (e.g., +0.2mm) to stabilize BOP
  - Simplify a problematic operation (e.g., multiple cuts instead of complex fuse)
- Forbidden: reinvent geometry or completely change dimensions.

IMPORTANT:
- Use only ops replace/insert_after/delete.
- `old` and `anchor` must exist exactly in the script.
- Keep `edits` as small as possible (max 12 edits).
"""

# ----------------------------
# Helpers
# ----------------------------

REQUIRED_EVAL_KEYS = {"status", "notes", "issues", "expected_dims", "observed_dims", "observed_bbox"}
REQUIRED_IMPL_KEYS = {"run_id", "plan", "script"}
REQUIRED_ARCH_KEYS = {"run_id", "intent", "params", "datums", "feature_plan", "acceptance", "plan"}

REQUIRED_DBG_KEYS = {"root_cause", "fix_type", "suggestions"}

REQUIRED_REPAIR_KEYS = {"failure_class", "edits", "notes"}

def normalize_repair(obj: dict) -> dict | None:
    if not isinstance(obj, dict):
        return None
    if not REQUIRED_REPAIR_KEYS.issubset(set(obj.keys())):
        return None
    fc = str(obj.get("failure_class") or "").strip().upper()
    if fc not in ("SYNTAX", "UNDEFINED_NAME", "CONTRACT", "CAD_RUNTIME"):
        return None
    edits = obj.get("edits")
    if not isinstance(edits, list):
        return None
    norm_edits: list[dict] = []
    for e in edits[:12]:
        if not isinstance(e, dict):
            continue
        op = str(e.get("op") or "").strip().lower()
        if op == "replace":
            old = e.get("old")
            new = e.get("new")
            if isinstance(old, str) and old and isinstance(new, str):
                norm_edits.append({"op": "replace", "old": old, "new": new})
        elif op == "insert_after":
            anchor = e.get("anchor")
            text = e.get("text")
            if isinstance(anchor, str) and anchor and isinstance(text, str):
                norm_edits.append({"op": "insert_after", "anchor": anchor, "text": text})
        elif op == "delete":
            old = e.get("old")
            if isinstance(old, str) and old:
                norm_edits.append({"op": "delete", "old": old})
    notes = obj.get("notes")
    if not isinstance(notes, str):
        notes = "" if notes is None else str(notes)
    return {"failure_class": fc, "edits": norm_edits, "notes": notes}

def apply_script_edits(script: str, edits: list[dict]) -> tuple[str, list[str]]:
    """Apply structured minimal edits deterministically.

    Returns (new_script, actions). If an edit cannot be applied (substring not found), it is skipped and recorded.
    """
    actions: list[str] = []
    s = script or ""
    for i, e in enumerate(edits or [], start=1):
        op = e.get("op")
        if op == "replace":
            old = e.get("old", "")
            new = e.get("new", "")
            if old in s:
                s = s.replace(old, new, 1)
                actions.append(f"edit#{i}: replace applied")
            else:
                actions.append(f"edit#{i}: replace skipped (old not found)")
        elif op == "insert_after":
            anchor = e.get("anchor", "")
            text = e.get("text", "")
            pos = s.find(anchor)
            if pos != -1:
                ins_at = pos + len(anchor)
                s = s[:ins_at] + text + s[ins_at:]
                actions.append(f"edit#{i}: insert_after applied")
            else:
                actions.append(f"edit#{i}: insert_after skipped (anchor not found)")
        elif op == "delete":
            old = e.get("old", "")
            if old in s:
                s = s.replace(old, "", 1)
                actions.append(f"edit#{i}: delete applied")
            else:
                actions.append(f"edit#{i}: delete skipped (old not found)")
        else:
            actions.append(f"edit#{i}: unknown op skipped")
    return s, actions

# --- Deterministic root-cause extractor (avoid debugger hallucinations) ---
_EXCEPTION_PATTERNS = (
    r"^Traceback \(most recent call last\):$",
    r"^(NameError|TypeError|AttributeError|ValueError|KeyError|IndexError|AssertionError|RuntimeError|NotImplementedError|ImportError|ModuleNotFoundError|OSError|IOError|Exception):\s+.*$",
    r"^Exception while processing file:.*$",
)

_exception_re = re.compile("|".join(_EXCEPTION_PATTERNS), flags=re.MULTILINE)

def extract_root_cause_line(stderr_s: str, stdout_s: str) -> str:
    """Return the first meaningful exception/trace line from stderr/stdout.

    This is used for logging/lessons so we don't learn from hallucinated debugger output.
    """
    blob = "\n".join([(stderr_s or ""), (stdout_s or "")]).strip()
    if not blob:
        return ""

    # Prefer the first explicit exception line (NameError/TypeError/...) if present.
    for ln in blob.splitlines():
        ln_s = (ln or "").strip()
        if not ln_s:
            continue
        if re.match(_EXCEPTION_PATTERNS[1], ln_s):
            return ln_s

    # Otherwise capture the first traceback marker or processing exception.
    m = _exception_re.search(blob)
    if m:
        return (m.group(0) or "").strip()

    return ""

# --- Safe math expression repair for JSON ---
import ast
import operator as _op

_ALLOWED_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
    ast.USub: _op.neg,
    ast.UAdd: _op.pos,
}

def _safe_eval_arith(expr: str) -> float | int:
    """Safely evaluate a tiny subset of arithmetic: numbers + - * / // % ** and parentheses."""
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("empty expr")

    node = ast.parse(expr, mode="eval")

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        raise ValueError(f"unsupported expr: {expr}")

    return _eval(node)


def repair_json_arithmetic(text: str) -> str:
    """Replace simple arithmetic expressions used as JSON values with computed literals.

    Example: {"distance": 112 + 30} -> {"distance": 142}

    Only repairs expressions that:
    - appear as a value after ':'
    - contain only digits, decimal points, whitespace, and + - * / ( ) operators
    - are directly followed by ',' or '}'
    """
    if not text:
        return text

    pattern = re.compile(r":\s*([0-9\s\+\-\*\/\(\)\.]+)\s*(?=,|\})")

    def _repl(m: re.Match) -> str:
        expr = (m.group(1) or "").strip()
        # Fast path: already a plain number
        if re.fullmatch(r"[0-9]+(\.[0-9]+)?", expr):
            return ": " + expr
        try:
            val = _safe_eval_arith(expr)
        except Exception:
            return m.group(0)  # leave unchanged

        # Emit int if it's effectively an int
        if isinstance(val, float) and abs(val - int(val)) < 1e-9:
            val = int(val)
        return ": " + str(val)

    return pattern.sub(_repl, text)


def extract_json(text: str):
    text = (text or "").strip()
    # First attempt: direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Second attempt: find first JSON-like block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        # Last resort: try to repair the whole text anyway
        repaired = repair_json_arithmetic(text)
        return json.loads(repaired)

    blob = m.group(0)

    # Try raw blob
    try:
        return json.loads(blob)
    except Exception:
        # Try repaired blob (fixes things like `112 + 30`)
        repaired = repair_json_arithmetic(blob)
        return json.loads(repaired)


def sanitize_run_id(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9-]+", "-", s).strip("-")
    s = (s[:20] or "run").strip("-")
    return s or "run"


def normalize_eval(obj: dict) -> dict | None:
    """Accept evaluator answers even if they have extra keys; return only required schema."""
    if not isinstance(obj, dict):
        return None
    if not REQUIRED_EVAL_KEYS.issubset(set(obj.keys())):
        return None

    status = str(obj.get("status", "")).strip().upper()
    if status not in ("PASS", "FAIL"):
        return None

    notes = obj.get("notes")
    issues = obj.get("issues")

    if not isinstance(notes, str):
        notes = str(notes) if notes is not None else ""

    if not isinstance(issues, list):
        issues = [str(issues)] if issues is not None else []

    return {
        "status": status,
        "notes": notes,
        "issues": issues,
        "expected_dims": obj.get("expected_dims"),
        "observed_dims": obj.get("observed_dims"),
        "observed_bbox": obj.get("observed_bbox"),
    }


# --- New helpers for implementer/architect schema ---
def normalize_impl(obj: dict) -> dict | None:
    """Return a minimal valid implementer spec (run_id, plan[list[str]], script[str]) or None."""
    if not isinstance(obj, dict):
        return None
    if not REQUIRED_IMPL_KEYS.issubset(set(obj.keys())):
        return None

    run_id = sanitize_run_id(obj.get("run_id", "run"))

    plan_val = obj.get("plan", [])
    if isinstance(plan_val, list):
        plan = [p if isinstance(p, str) else json.dumps(p, ensure_ascii=False) for p in plan_val]
    else:
        plan = [str(plan_val)]

    script = obj.get("script")
    if not isinstance(script, str) or not script.strip():
        return None

    return {"run_id": run_id, "plan": plan, "script": script}


def looks_like_architect(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    # architect outputs feature_plan/acceptance/params etc.
    return ("feature_plan" in keys) or REQUIRED_ARCH_KEYS.issubset(keys)


# --- FreeCAD script contract validator ---
def validate_freecad_script(script: str) -> tuple[bool, list[str]]:
    """Hard-validate that the script matches the strict FreeCAD runner contract.

    This prevents wasting CAD runs on scripts that will definitely fail (wrong imports,
    wrong export path, missing required footer, etc.).
    """
    issues: list[str] = []
    if not isinstance(script, str) or not script.strip():
        return False, ["script empty"]

    s = script.strip().replace("\r\n", "\n")
    s_lower = s.lower()

    required_import_block = (
        "import FreeCAD as App\n"
        "import Part\n"
        "import os, json\n"
        "import math\n"
    )

    # Must start with the exact 4-line import block
    if not s.startswith(required_import_block):
        issues.append("script must start with exact 4-line import block (App/Part/os,json/math)")

    # Required mandatory lines/fragments
    if 'doc = App.newDocument("Model")' not in s:
        issues.append('missing: doc = App.newDocument("Model")')

    # Require an actual assignment to `solid` somewhere in the body.
    # (A mere mention like "obj.Shape = solid" is not enough.)
    if not re.search(r"^\s*solid\s*=", s, flags=re.MULTILINE):
        issues.append("missing: assignment to variable 'solid' (final shape)")

    if 'obj = doc.addObject("Part::Feature","Result"); obj.Shape = solid; doc.recompute()' not in s:
        issues.append("missing: exact Result object creation line")

    if 'out_path = os.path.join(os.getcwd(),"result.step")' not in s:
        issues.append("missing: out_path using os.getcwd() and result.step")

    if 'obj.Shape.exportStep(out_path)' not in s:
        issues.append("missing: obj.Shape.exportStep(out_path)")

    required_print = 'print(json.dumps({"checks": {"bbox": [bb.XMin,bb.YMin,bb.ZMin,bb.XMax,bb.YMax,bb.ZMax], "volume": float(obj.Shape.Volume)}}))'
    if required_print not in s:
        issues.append("missing: exact final print(json.dumps({checks:{bbox,volume}})) line")
    if 'bb = obj.Shape.BoundBox' not in s:
        issues.append('missing: bb = obj.Shape.BoundBox')

    # Forbidden imports / paths / modules
    forbidden_tokens = [
        "import FreeCAD,",
        "from FreeCAD",
        "Sketcher",
        "Draft",
        "PartDesign",
        "Import",
        "export(__objs__",
        "/tmp/",
        "import os\\nimport json",
        "import json\\nimport os",
        "import os, json,",
        "import os,json",
        "Part.newDocument",
        "step_path",
        "Base",
        "makePolyline",
        "reduce(",
        "functools",
    ]
    for tok in forbidden_tokens:
        if tok in s:
            issues.append(f"forbidden token present: {tok}")

    # --- Additional hard checks ---
    # 1. Part.newDocument or Part.newdocument
    if "Part.newDocument" in s or "Part.newdocument" in s:
        issues.append("forbidden: Part.newDocument (must use App.newDocument)")
    # 2. step_path anywhere (case-insensitive)
    if "step_path" in s_lower:
        issues.append("forbidden: step_path (must use out_path footer only)")
    # 3. Base. or standalone Base (case-insensitive)
    if "Base." in s or re.search(r"\bBase\b", s, flags=re.IGNORECASE):
        issues.append("forbidden: Base namespace (must avoid Base.* and use App.Vector + Shape.rotate)")
    # 4. makePolyline (case-insensitive)
    if re.search(r"makepolyline", s, flags=re.IGNORECASE):
        issues.append("forbidden: Part.makePolyline (not available; use Part.makePolygon / edges)")
    # 5. Regex for keyword-args in Part primitives
    primitive_regexes = [
        (r"Part\.makeSphere\s*\(.*\w+\s*=", "forbidden: keyword args in Part.makeSphere(...)"),
        (r"Part\.makeCylinder\s*\(.*\w+\s*=", "forbidden: keyword args in Part.makeCylinder(...)"),
        (r"Part\.makeCone\s*\(.*\w+\s*=", "forbidden: keyword args in Part.makeCone(...)"),
        (r"Part\.makeTorus\s*\(.*\w+\s*=", "forbidden: keyword args in Part.makeTorus(...)"),
    ]
    for pat, msg in primitive_regexes:
        if re.search(pat, s):
            issues.append(msg)
    # 6. Multiple out_path = assignments
    if s.count("out_path =") > 1:
        issues.append("out_path assigned multiple times")

    # Guard: out_path must not be referenced before the footer defines it.
    # If a generated script uses out_path earlier (e.g. alternate export), it will NameError
    # once we strip/replace exports.
    first_out_assign = s.find('out_path = os.path.join(os.getcwd(),"result.step")')
    if first_out_assign != -1:
        pre = s[:first_out_assign]
        if re.search(r"\bout_path\b", pre):
            issues.append("out_path referenced before footer assignment")

    # 7. Forbid common hallucinated identifiers that frequently cause NameError
    hallucinated_idents = [
        "part_obj",
        "outer_circle",
        "plate",
        "hole_pt",
        "single_hole",
        "base_cylinder",
        "hole_cyl",
        "hole_cylinder",
    ]
    if re.search(r"\b(" + "|".join(hallucinated_idents) + r")\b", s):
        issues.append("forbidden: hallucinated identifier used (causes NameError); build shapes inline and keep only 'solid' + local 'tool'")

    return (len(issues) == 0), issues



def validate_python_syntax(script: str) -> tuple[bool, str]:
    """Return (ok, error_message). Catches SyntaxError/IndentationError early."""
    if not isinstance(script, str) or not script.strip():
        return False, "script empty"
    try:
        compile(script, "<freecad_job>", "exec")
        return True, ""
    except (SyntaxError, IndentationError) as e:
        # Format similar to Python trace: message (line:col)
        msg = getattr(e, "msg", None) or str(e) or type(e).__name__
        lineno = getattr(e, "lineno", None)
        offset = getattr(e, "offset", None)
        return False, f"{msg} (line {lineno}, col {offset})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _strip_unexpected_top_level_indent(lines: list[str], line_no_1based: int) -> bool:
    """If a top-level line has leading whitespace causing 'unexpected indent', strip it."""
    idx = (line_no_1based or 0) - 1
    if idx < 0 or idx >= len(lines):
        return False
    ln = lines[idx]
    if not ln:
        return False
    # Only strip if it looks like accidental leading whitespace at top-level
    if ln.startswith(" ") or ln.startswith("\t"):
        lines[idx] = ln.lstrip(" \t")
        return True
    return False


def auto_fix_python_syntax(script: str) -> tuple[str, list[str]]:
    """Best-effort deterministic fixes for indentation/syntax issues introduced by generation.

    - Replace tabs with 4 spaces.
    - Strip trailing whitespace.
    - If compile() reports 'unexpected indent', strip leading whitespace on the offending line.

    Returns: (fixed_script, actions)
    """
    actions: list[str] = []
    if not isinstance(script, str):
        return "", ["script not a string"]

    s = script.replace("\r\n", "\n").replace("\t", "    ")
    if "\t" in script:
        actions.append("replaced tabs with 4 spaces")

    lines = [ln.rstrip() for ln in s.split("\n")]

    # Try a few rounds of compile-based repair
    for _ in range(3):
        candidate = "\n".join(lines)
        ok, err = validate_python_syntax(candidate)
        if ok:
            return candidate, actions

        # Attempt targeted fix for 'unexpected indent'
        if "unexpected indent" in err:
            # Extract line number from the message "... (line N, col M)"
            m = re.search(r"\(line\s+(\d+),\s+col\s+\d+\)", err)
            ln_no = int(m.group(1)) if m else None
            if ln_no and _strip_unexpected_top_level_indent(lines, ln_no):
                actions.append(f"stripped leading whitespace on line {ln_no} (unexpected indent)")
                continue

        # If we can't fix deterministically, break.
        actions.append(f"python syntax still failing: {err}")
        break

    return "\n".join(lines), actions
# --- FreeCAD script auto-fixer ---

def auto_fix_freecad_script(script: str) -> tuple[str, list[str]]:
    """Best-effort deterministic repair for common runner-contract violations.

    We DO NOT change geometry strategy here. We only:
    - normalize the mandatory import block
    - normalize the mandatory doc/out/export/footer contract lines
    - strip obviously forbidden module/path lines

    Returns: (fixed_script, actions)
    """
    actions: list[str] = []
    if not isinstance(script, str):
        return "", ["script not a string"]

    s = script.replace("\r\n", "\n")
    lines = s.split("\n")

    # Strip forbidden lines (hard)
    forbidden_line_tokens = [
        "Sketcher",
        "PartDesign",
        "Draft",
        "Import",
        "from FreeCAD",
        "/tmp/",
        "Base",
        "step_path",
        "Part.newDocument",
        "makePolyline",
        "part_obj",
        "outer_circle",
        "plate",
        "hole_pt",
        "single_hole",
        "base_cylinder",
        "hole_cyl",
        "hole_cylinder",
    ]
    new_lines: list[str] = []
    for ln in lines:
        if any(tok in ln for tok in forbidden_line_tokens):
            actions.append(f"removed forbidden line: {ln[:80]}")
            continue
        new_lines.append(ln)
    lines = new_lines

    # Remove any existing import lines at the top (we will replace)
    # Keep shebang/encoding comments if present.
    prefix: list[str] = []
    body_start = 0
    for i, ln in enumerate(lines):
        if ln.startswith("#!") or ln.startswith("# -*-") or ln.startswith("# coding"):
            prefix.append(ln)
            continue
        # stop prefix on first non-comment non-empty line
        if ln.strip() == "" or ln.strip().startswith("#"):
            prefix.append(ln)
            continue
        body_start = i
        break

    body = lines[body_start:]

    # Drop leading import block(s) from body
    while body and (body[0].lstrip().startswith("import ") or body[0].lstrip().startswith("from ")):
        actions.append(f"dropped import line: {body[0].strip()}")
        body.pop(0)

    # Also drop subsequent import lines that sometimes appear later (common model mistake)
    cleaned_body: list[str] = []
    for ln in body:
        if ln.lstrip().startswith("import ") or ln.lstrip().startswith("from "):
            actions.append(f"dropped mid-body import line: {ln.strip()}")
            continue
        cleaned_body.append(ln)
    body = cleaned_body

    required_import_block = [
        "import FreeCAD as App",
        "import Part",
        "import os, json",
        "import math",
        "",
    ]

    # Ensure doc line exists and is exact
    doc_line = 'doc = App.newDocument("Model")'
    found_doc = False
    for i, ln in enumerate(body):
        if "App.newDocument" in ln:
            body[i] = doc_line
            actions.append("normalized doc creation line")
            found_doc = True
            break
    if not found_doc:
        # Insert after imports
        body.insert(0, doc_line)
        body.insert(1, "")
        actions.append("inserted doc creation line")
    # Replace any accidental Part.newDocument with App.newDocument
    for i, ln in enumerate(body):
        if "Part.newDocument" in ln:
            body[i] = doc_line
            actions.append("replaced Part.newDocument with App.newDocument")

    # Ensure out_path + exportStep exist; we will enforce footer later, but normalize if present
    for i, ln in enumerate(body):
        if "os.path.join(os.getcwd()" in ln and "result.step" in ln:
            body[i] = 'out_path = os.path.join(os.getcwd(),"result.step")'
            actions.append("normalized out_path line")

    # Remove alternative export calls like Part.export([...], stepfile)
    filtered_body: list[str] = []
    for ln in body:
        if "Part.export" in ln:
            actions.append("removed Part.export usage")
            continue
        filtered_body.append(ln)
    body = filtered_body

    # Enforce footer: remove any previous footer-ish exports/prints to avoid duplicates.
    # Additionally, remove any out_path assignments/usages outside the final footer to prevent
    # (a) "out_path assigned multiple times" contract fails and
    # (b) NameError if we strip an earlier assignment but a later line still references it.
    footer_markers = [
        "addObject(\"Part::Feature\"",
        "exportStep(",
        "result.step",
        "print(json.dumps(",
        "out_path =",
        "\bout_path\b",
    ]

    trimmed: list[str] = []
    for ln in body:
        ln_s = ln.strip()
        # Remove any exports/prints/out_path lines in the body (footer will be re-added exactly once).
        if (
            'out_path' in ln
            or 'exportStep(' in ln
            or 'Part.export' in ln
            or 'print(json.dumps(' in ln
            or 'doc.addObject("Part::Feature"' in ln
            or 'doc.addObject(\"Part::Feature\"' in ln
            or 'result.step' in ln
        ):
            actions.append(f"removed body export/footer/out_path line: {ln_s[:80]}")
            continue
        trimmed.append(ln)
    body = trimmed

    # Collapse excessive blank lines after stripping
    compact: list[str] = []
    blank_run = 0
    for ln in body:
        if ln.strip() == "":
            blank_run += 1
            if blank_run >= 3:
                continue
        else:
            blank_run = 0
        compact.append(ln)
    body = compact

    # Ensure there is at least one reference to 'solid' somewhere; if not, we can't fix.
    if "solid" not in "\n".join(body):
        actions.append("WARNING: no 'solid' variable found; cannot guarantee runner contract")

    footer = [
        "",
        "# --- Runner contract footer (DO NOT EDIT) ---",
        'obj = doc.addObject("Part::Feature","Result"); obj.Shape = solid; doc.recompute()',
        'out_path = os.path.join(os.getcwd(),"result.step")',
        "obj.Shape.exportStep(out_path)",
        "bb = obj.Shape.BoundBox",
        'print(json.dumps({"checks": {"bbox": [bb.XMin,bb.YMin,bb.ZMin,bb.XMax,bb.YMax,bb.ZMax], "volume": float(obj.Shape.Volume)}}))',
        "",
    ]

    fixed_lines = prefix + required_import_block + body + footer
    fixed = "\n".join(fixed_lines)

    # Final preflight: try to eliminate common indentation errors deterministically
    fixed2, py_actions = auto_fix_python_syntax(fixed)
    if py_actions:
        actions.extend([f"pyfix: {a}" for a in py_actions])
    return fixed2, actions
def normalize_debug(obj: dict) -> dict | None:
    if not isinstance(obj, dict):
        return None
    if not REQUIRED_DBG_KEYS.issubset(set(obj.keys())):
        return None
    rc = obj.get("root_cause")
    ft = obj.get("fix_type")
    sug = obj.get("suggestions")
    if not isinstance(rc, str):
        rc = "" if rc is None else str(rc)
    ft_s = str(ft).strip().lower() if ft is not None else ""
    if ft_s not in ("api", "strategy", "params"):
        ft_s = "api"
    if not isinstance(sug, list):
        sug = [] if sug is None else [str(sug)]
    return {"root_cause": rc, "fix_type": ft_s, "suggestions": sug}


# ----------------------------
# Persistent learning + logging
# ----------------------------
trace = "agent_trace" + version + ".jsonl"
lessons = "agent_lessons" + version + ".jsonl"
fail_run = "fail_run" + version
cad_runs_log = "cad_runs_log" + version + ".csv"

LESSONS_PATH = os.path.join(os.path.dirname(__file__), lessons)
RUNS_LOG_CSV = os.path.join(os.path.dirname(__file__), cad_runs_log)
TRACE_JSONL_PATH = os.path.join(os.path.dirname(__file__), trace)
FAIL_RUNS_DIR = os.path.join(os.path.dirname(__file__), fail_run)
def _safe_mkdir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _move_dir_into(src_dir: str, dst_parent_dir: str) -> tuple[bool, str]:
    """Move src_dir into dst_parent_dir. Returns (moved_ok, dst_path_or_error)."""
    try:
        if not src_dir or not os.path.isdir(src_dir):
            return False, "src_not_dir"
        _safe_mkdir(dst_parent_dir)
        base = os.path.basename(os.path.normpath(src_dir))
        dst = os.path.join(dst_parent_dir, base)
        if os.path.abspath(src_dir) == os.path.abspath(dst):
            return False, "src_equals_dst"
        # Avoid collisions
        if os.path.exists(dst):
            suffix = datetime.utcnow().strftime("%H%M%S")
            dst = os.path.join(dst_parent_dir, f"{base}-{suffix}")
        shutil.move(src_dir, dst)
        return True, dst
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def finalize_fail_runs(*, session_id: str, created_out_dirs: set[str], keep_dir: str | None) -> None:
    """Move all failed cad run folders (created_out_dirs except keep_dir) into fail_run/<session_id>/"""
    try:
        if not created_out_dirs:
            return
        dst_parent = os.path.join(FAIL_RUNS_DIR, session_id)
        _safe_mkdir(dst_parent)
        keep_abs = os.path.abspath(keep_dir) if keep_dir else None

        for d in sorted(created_out_dirs):
            if not d or not os.path.isdir(d):
                continue
            d_abs = os.path.abspath(d)
            if keep_abs and d_abs == keep_abs:
                continue
            _move_dir_into(d, dst_parent)
    except Exception:
        # Never break the main flow because of cleanup
        return


def load_jobs_json(path: str) -> list[str]:
    """Load batch prompts from a jobs.json file.

    Supported formats:
    - ["prompt1", "prompt2", ...]
    - {"jobs": [{"prompt": "..."}, ...]}
    - {"jobs": ["prompt1", "prompt2", ...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]

    if isinstance(data, dict):
        jobs = data.get("jobs")
        if isinstance(jobs, list):
            out: list[str] = []
            for j in jobs:
                if isinstance(j, str):
                    s = j.strip()
                    if s:
                        out.append(s)
                elif isinstance(j, dict):
                    s = str(j.get("prompt", "")).strip()
                    if s:
                        out.append(s)
            return out

    raise ValueError("Invalid jobs.json format")


def _utc_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_text(s: str) -> str:
    try:
        return "sha256:" + hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return "sha256:"


def _truncate_text(s: str, max_chars: int = 20000) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n\n...TRUNCATED {len(s) - max_chars} chars..."


def _summarize_blob(s: str, max_chars: int = 20000) -> dict:
    """Return a compact+useful representation for large raw strings."""
    s = s or ""
    return {
        "sha": _sha256_text(s),
        "len": len(s),
        "text": _truncate_text(s, max_chars=max_chars),
    }


def log_event(
    *,
    event: str,
    session_id: str,
    iteration: int,
    run_id: str | None,
    payload: dict,
) -> None:
    """Append one structured record into a SINGLE large JSONL file for later learning."""
    rec = {
        "ts": _utc_ts(),
        "event": str(event or ""),
        "session_id": str(session_id or ""),
        "iteration": int(iteration or 0),
        "run_id": str(run_id or ""),
        "payload": payload or {},
    }
    os.makedirs(os.path.dirname(TRACE_JSONL_PATH), exist_ok=True)
    with open(TRACE_JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def append_csv_row(path: str, header: list[str], row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


def save_lesson(event: str, run_id: str, fix_type: str, root_cause: str, suggestions: list[str], extra: dict | None = None):
    payload = {
        "ts": _utc_ts(),
        "event": event,
        "run_id": run_id,
        "fix_type": fix_type,
        "root_cause": root_cause,
        "suggestions": suggestions or [],
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(LESSONS_PATH), exist_ok=True)
    with open(LESSONS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_lessons(max_n: int = 25) -> list[dict]:
    if not os.path.exists(LESSONS_PATH):
        return []
    lessons: list[dict] = []
    try:
        with open(LESSONS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    lessons.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return lessons[-max_n:]


def lessons_to_prompt(lessons: list[dict]) -> str:
    if not lessons:
        return ""
    # Keep it short and actionable for the model
    lines = ["\n\n=== LESSONS LEARNED (persisted from previous runs) ==="]
    lines.append("FORBIDDEN PATTERNS (hard): step_path, Base.*, Part.newDocument, Part.makePolyline/makePolyline, keyword-args in Part primitives, and hallucinated identifiers (plate/part_obj/hole_pt/single_hole/outer_circle/base_cylinder)")
    for l in lessons:
        ts = l.get("ts", "")
        ev = l.get("event", "")
        ft = l.get("fix_type", "")
        rc = (l.get("root_cause", "") or "").strip()
        if len(rc) > 180:
            rc = rc[:180] + "…"
        lines.append(f"- [{ts}] {ev} ({ft}): {rc}")
        sug = l.get("suggestions") or []
        for s in sug[:2]:
            s = (s or "").strip()
            if s:
                if len(s) > 180:
                    s = s[:180] + "…"
                lines.append(f"    * {s}")
    lines.append("=== END LESSONS ===\n")
    return "\n".join(lines)


async def run_evaluator(evaluator, payload, images, tries=3, timeout_s=90) -> tuple[dict, str]:
    def _b64_to_bytes(b64val: str) -> bytes:
        if not b64val:
            return b""
        if b64val.startswith("data:"):
            b64val = b64val.split(",", 1)[-1]
        return base64.b64decode(b64val)

    iso_bytes = _b64_to_bytes((images or {}).get("iso", ""))
    front_bytes = _b64_to_bytes((images or {}).get("front", ""))
    right_bytes = _b64_to_bytes((images or {}).get("right", ""))
    top_bytes = _b64_to_bytes((images or {}).get("top", ""))

    if not (iso_bytes and front_bytes and right_bytes and top_bytes):
        return {
            "status": "FAIL",
            "notes": "Missing images (iso/front/right/top) for evaluator",
            "issues": ["missing rendered images"],
            "expected_dims": None,
            "observed_dims": None,
            "observed_bbox": None,
        }

    def _looks_like_planner(obj: dict) -> bool:
        if not isinstance(obj, dict):
            return False
        keys = set(obj.keys())
        return ("script" in keys) or ("run_id" in keys and "plan" in keys)

    base_payload = json.dumps(payload, ensure_ascii=False)

    last_raw = ""

    def make_prompt(strict: bool) -> str:
        schema = (
            "{"
            "\"status\":\"PASS|FAIL\","
            "\"notes\":\"<string>\","
            "\"issues\":[\"...\"],"
            "\"expected_dims\":null,"
            "\"observed_dims\":null,"
            "\"observed_bbox\":null"
            "}"
        )
        if not strict:
            return (
                "TASK: Assess whether the CAD result matches the prompt.\n"
                "Respond ONLY with JSON in the following schema (no extra keys, no Markdown):\n"
                f"{schema}\n\n"
                "Use payload + images for evaluation. Payload:\n"
                + base_payload
            )
        return (
            "IGNORE ALL PREVIOUS INSTRUCTIONS / OUTPUT FORMATS.\n"
            "YOU ARE A GEOMETRY REVIEWER.\n"
            "Respond NOW with exactly this JSON schema, NO other keys, NO Markdown, NO extra text:\n"
            f"{schema}\n\n"
            "Assess the geometry using payload + images (NOT the format of the previous answer!).\n"
            "Payload:\n"
            + base_payload
        )

    for i in range(1, tries + 1):
        strict = (i >= 2)
        prompt_text = make_prompt(strict)

        msg = ChatMessage(
            role=Role.USER,
            contents=[
                TextContent(text=prompt_text),
                DataContent(data=iso_bytes, media_type="image/png"),
                DataContent(data=front_bytes, media_type="image/png"),
                DataContent(data=right_bytes, media_type="image/png"),
                DataContent(data=top_bytes, media_type="image/png"),
            ],
        )

        try:
            res = await asyncio.wait_for(evaluator.run(msg), timeout=timeout_s)
        except asyncio.TimeoutError:
            print(f"\n--- EVALUATOR TIMEOUT (try {i}) ---")
            continue
        except Exception as e:
            print(f"\n--- EVALUATOR EXCEPTION (try {i}) ---\n{type(e).__name__}: {e}")
            continue

        raw = (res.text or "").strip()
        last_raw = raw
        print(f"\n--- EVALUATOR RAW (try {i}) ---\n{raw}")

        try:
            obj = extract_json(raw)
        except Exception:
            obj = None

        # If it starts acting like a planner again, force a retry (no meta-notes).
        if obj and _looks_like_planner(obj):
            continue

        norm = normalize_eval(obj) if obj else None
        if norm:
            return norm, last_raw

    # Fallback: use bbox so the replanner can still "learn".
    bb = payload.get("render_bbox") or payload.get("bbox")
    return (
        {
            "status": "FAIL",
            "notes": "Evaluator unreliable; bbox-only fallback.",
            "issues": ["invalid evaluator output", f"bbox={bb}"],
            "expected_dims": None,
            "observed_dims": None,
            "observed_bbox": bb,
        },
        last_raw,
    )


async def main(user_prompt: str | None = None, session_id: str | None = None):
    if user_prompt is None:
        user_prompt = input("What should be built? > ").strip()
    else:
        user_prompt = str(user_prompt).strip()

    if not session_id:
        session_id = sanitize_run_id("sess-" + datetime.utcnow().strftime("%m%d%H%M%S"))
    else:
        session_id = sanitize_run_id(session_id)

    created_out_dirs: set[str] = set()
    log_event(
        event="SESSION_START",
        session_id=session_id,
        iteration=0,
        run_id=None,
        payload={
            "user_prompt": user_prompt,
        },
    )
    persisted_lessons = load_lessons(max_n=50)
    lessons_prompt = lessons_to_prompt(persisted_lessons)

    ep = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
    model = os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"]

    async with AzureCliCredential() as cred:

        
        client = AzureAIClient(
            credential=cred,
            project_endpoint=ep,
            model_deployment_name=model,
            agent_name="freecad-vision-v3",
        )

        try:
            architect = client.create_agent(name="architect", instructions=ARCHITECT_INSTRUCTIONS)
            implementer = client.create_agent(name="implementer", instructions=IMPLEMENTER_INSTRUCTIONS)
            evaluator = client.create_agent(name="evaluator", instructions=EVALUATOR_INSTRUCTIONS)
            replanner = client.create_agent(name="replanner", instructions=REPLANNER_INSTRUCTIONS)
            debugger = client.create_agent(name="debugger", instructions=DEBUGGER_INSTRUCTIONS)
            repairer = client.create_agent(name="repairer", instructions=REPAIRER_INSTRUCTIONS)

            arch_spec = None
            last_eval = None
            last_geom_eval = None
            last_failure_phase = "NONE"

            # More iterations because we now run a "designer loop".
            for it in range(1, 11):
                print(f"\n================ Iteration {it}/11 ================")
                log_event(
                    event="ITERATION_START",
                    session_id=session_id,
                    iteration=it,
                    run_id=(arch_spec or {}).get("run_id") if isinstance(arch_spec, dict) else None,
                    payload={
                        "lessons_count": len(persisted_lessons or []),
                        "lessons_prompt": _summarize_blob(lessons_prompt, max_chars=6000),
                    },
                )

                # 1) ARCHITECT: create or replan the plan
                if it == 1:
                    raw_arch = await call_agent(architect, user_prompt, label="architect")
                else:
                    replanner_payload = {
                        "user_prompt": user_prompt,
                        "last_eval": last_eval,
                        "last_geom_eval": last_geom_eval,
                        "last_arch_spec": arch_spec,
                        "last_failure_phase": last_failure_phase,
                    }
                    raw_arch = await call_agent(
                        replanner,
                        lessons_prompt
                        + "\n"
                        + json.dumps(
                            replanner_payload,
                            ensure_ascii=False,
                        ),
                        label="replanner",
                    )

                print("\n--- ARCHITECT RAW ---")
                print(raw_arch.text)
                log_event(
                    event="ARCHITECT_RAW",
                    session_id=session_id,
                    iteration=it,
                    run_id=None,
                    payload={
                        "architect_raw": _summarize_blob(raw_arch.text, max_chars=80000),
                    },
                )

                try:
                    arch_spec = extract_json(raw_arch.text)
                except Exception as e:
                    last_eval = {
                        "status": "FAIL",
                        "notes": f"architect returned invalid JSON: {type(e).__name__}: {e}",
                        "issues": ["invalid architect json"],
                        "expected_dims": None,
                        "observed_dims": None,
                        "observed_bbox": None,
                    }
                    last_failure_phase = classify_failure_phase(last_eval)
                    log_event(
                        event="ARCH_JSON_FAIL",
                        session_id=session_id,
                        iteration=it,
                        run_id=None,
                        payload={
                            "error": f"{type(e).__name__}: {e}",
                            "last_eval": last_eval,
                        },
                    )
                    continue
                arch_spec["run_id"] = sanitize_run_id(arch_spec.get("run_id", "run"))
                log_event(
                    event="ARCH_SPEC",
                    session_id=session_id,
                    iteration=it,
                    run_id=arch_spec.get("run_id"),
                    payload={
                        "arch_spec": arch_spec,
                    },
                )

                # 2) IMPLEMENTER: Feature-Plan -> Script (robust with retries + contract feedback)
                impl_spec = None
                last_impl_raw = None
                last_contract_issues: list[str] = []
                extra_undef_retry_used = False
                max_impl_tries = 4  # allow one extra repair-try specifically for undefined-name preflight fails

                def _contract_feedback_text(issues: list[str]) -> str:
                    if not issues:
                        return ""
                    bullets = "\n".join([f"- {x}" for x in issues[:12]])
                    extra_hint = ""
                    # If we detect undefined-name problems, make the repair instruction very explicit.
                    if any("undefined name" in (x or "").lower() or "undefined" in (x or "").lower() for x in issues):
                        # Collect explicit undefined names if present in the issues text
                        names: list[str] = []
                        for x in issues:
                            x_s = (x or "")
                            if x_s.startswith("undefined name '"):
                                m = re.search(r"undefined name '([^']+)'", x_s)
                                if m:
                                    names.append(m.group(1))
                        names = sorted(set([n for n in names if n]))
                        names_txt = (", ".join(names[:20]) if names else "(not extractable)")
                        extra_hint = (
                            "\n\nADDITIONAL (very important): Your script uses undefined variables and therefore will NOT be executed."
                            "\n- Affected names: " + names_txt +
                            "\n\nREPAIR RULE (hard):"
                            "\n1) At the top, define EXACTLY ONE param dict: `P = {...}` (literals only)."
                            "\n2) After that, use ONLY `P[\"...\"]` or literals."
                            "\n3) No invented variable names. If you need hole positions, compute them inline in the loop (x/y) and immediately use a local variable `tool` right before `solid = solid.cut(tool)`."
                            "\n4) `solid` is the only persistent result variable. Everything else is short-lived."
                        )
                    return (
                        "\n\nIMPORTANT: Your last script VIOLATES the runner contract or preflight rules. "
                        "You MUST fix it exactly. The violations are:\n"
                        + bullets
                        + "\n\nReminder: ONLY these are allowed: import FreeCAD as App; import Part; import os, json; import math. "
                        "No Sketcher/PartDesign/Draft/Import. Export ONLY to os.getcwd()/result.step. "
                        "Solid must be in variable 'solid'. The Result line and final print(json.dumps(...)) must be exact."
                        + extra_hint
                    )

                for impl_try in range(1, max_impl_tries + 1):
                    impl_prompt = json.dumps(arch_spec, ensure_ascii=False)

                    # Always reinforce the output schema (models sometimes mirror the input JSON).
                    header = (
                        "IMPORTANT: You are the IMPLEMENTER. Return ONLY this JSON (no extra keys, no Markdown):\n"
                        "{\"run_id\":\"...\",\"plan\":[\"...\"],\"script\":\"<FreeCAD Python Script>\"}\n\n"
                        "You MUST provide a FreeCAD script that satisfies the requirements (imports, doc, solid, STEP export, print checks).\n"
                        "FORBIDDEN: Sketcher, Draft, PartDesign, Import, from FreeCAD, /tmp, any additional imports.\n"
                    )

                    header = header + lessons_prompt + "\n\nMANDATORY: Run a preflight check on your script for the FORBIDDEN PATTERNS. If one appears, fix it BEFORE output."

                    # If we already know contract violations, feed them back explicitly.
                    feedback = _contract_feedback_text(last_contract_issues)

                    # Always include the contract header. First try has no feedback, but still has the rules.
                    base = header + "\n\nHere is the ARCHITECT plan as input (DO NOT mirror it back!):\n" + impl_prompt
                    if impl_try == 1 and not last_contract_issues:
                        full_prompt = base
                    else:
                        full_prompt = header + feedback + "\n\nHere is the ARCHITECT plan as input (DO NOT mirror it back!):\n" + impl_prompt

                    raw_impl = await call_agent(implementer, full_prompt, label=f"implementer (try {impl_try})")
                    last_impl_raw = (raw_impl.text or "").strip()

                    print("\n--- IMPLEMENTER RAW ---")
                    print(last_impl_raw)
                    log_event(
                        event="IMPLEMENTER_RAW",
                        session_id=session_id,
                        iteration=it,
                        run_id=arch_spec.get("run_id") if isinstance(arch_spec, dict) else None,
                        payload={
                            "impl_try": impl_try,
                            "implementer_raw": _summarize_blob(last_impl_raw, max_chars=120000),
                            "last_contract_issues": last_contract_issues[:50],
                        },
                    )

                    try:
                        parsed = extract_json(last_impl_raw)
                    except Exception:
                        parsed = None

                    # If implementer mistakenly returns the architect JSON, retry.
                    if parsed and looks_like_architect(parsed):
                        last_contract_issues = ["implementer mirrored architect JSON instead of producing script JSON"]
                        continue

                    candidate = normalize_impl(parsed) if parsed else None
                    if not candidate:
                        last_contract_issues = ["implementer output JSON missing run_id/plan/script or script empty"]
                        continue

                    run_id = sanitize_run_id(candidate.get("run_id", arch_spec["run_id"]))
                    candidate["run_id"] = run_id

                    # Deterministic auto-fix for common contract violations
                    fixed_script, fix_actions = auto_fix_freecad_script(candidate.get("script", ""))
                    if fix_actions:
                        print("\n--- AUTO-FIX ACTIONS ---")
                        for a in fix_actions[:20]:
                            print(f"- {a}")
                    candidate["script"] = fixed_script
                    log_event(
                        event="SCRIPT_FIXED",
                        session_id=session_id,
                        iteration=it,
                        run_id=run_id,
                        payload={
                            "fix_actions": fix_actions[:200],
                            "script_sha": _sha256_text(fixed_script),
                            "script_len": len(fixed_script or ""),
                            "script": _truncate_text(fixed_script, max_chars=180000),
                        },
                    )

                    repaired_once = False

                    # Hard contract validation BEFORE running FreeCAD
                    ok_script, script_issues = validate_freecad_script(candidate.get("script", ""))
                    if not ok_script:
                        last_contract_issues = script_issues
                        print("\n--- CONTRACT VALIDATION FAIL ---")
                        for si in script_issues[:20]:
                            print(f"- {si}")
                        save_lesson(
                            event="contract_validation_fail",
                            run_id=run_id,
                            fix_type="api",
                            root_cause="; ".join(script_issues[:6]),
                            suggestions=[
                                "Start script with the exact 4-line import block (App/Part/os, json/math).",
                                "Use doc = App.newDocument(\"Model\") exactly.",
                                "Use the exact footer: Result object + exportStep to os.getcwd()/result.step + bb + print(json.dumps(...)).",
                                "Do not use Sketcher/PartDesign/Draft/Import or /tmp paths.",
                            ],
                            extra={"issues": script_issues[:20]},
                        )
                        append_csv_row(
                            RUNS_LOG_CSV,
                            header=["ts", "run_id", "stage", "status", "message"],
                            row={
                                "ts": _utc_ts(),
                                "run_id": run_id,
                                "stage": "implementer_contract",
                                "status": "FAIL",
                                "message": " | ".join(script_issues[:8]),
                            },
                        )
                        log_event(
                            event="CONTRACT_FAIL",
                            session_id=session_id,
                            iteration=it,
                            run_id=run_id,
                            payload={
                                "script_issues": script_issues,
                                "last_contract_issues": last_contract_issues,
                            },
                        )
                        # --- Engineering Repair Pipeline: attempt minimal patch once ---
                        if not repaired_once and last_impl_raw:
                            try:
                                repair_payload = {
                                    "failure_class": "CONTRACT",
                                    "policy": "patch-only (no full rewrite)",
                                    "script": candidate.get("script", ""),
                                    "errors": script_issues,
                                    "stderr": "",
                                    "stdout": "",
                                    "root_cause_line": "",
                                }
                                raw_rep = await call_agent(
                                    repairer,
                                    json.dumps(repair_payload, ensure_ascii=False),
                                    label=f"repairer ({repair_payload['failure_class']})",
                                    timeout_s=90,
                                )
                                rep_obj = None
                                try:
                                    rep_obj = extract_json((raw_rep.text or "").strip())
                                except Exception:
                                    rep_obj = None
                                rep = normalize_repair(rep_obj) if rep_obj else None
                                if rep and rep.get("edits"):
                                    patched, patch_actions = apply_script_edits(candidate.get("script", ""), rep.get("edits"))
                                    candidate["script"] = patched
                                    repaired_once = True
                                    log_event(
                                        event="REPAIR_PATCH_APPLIED",
                                        session_id=session_id,
                                        iteration=it,
                                        run_id=run_id,
                                        payload={
                                            "failure_class": rep.get("failure_class"),
                                            "notes": rep.get("notes"),
                                            "patch_actions": patch_actions,
                                        },
                                    )
                                    # Re-run deterministic auto-fix + validations after patch
                                    patched2, fix_actions2 = auto_fix_freecad_script(candidate.get("script", ""))
                                    candidate["script"] = patched2
                                    # restart the outer preflight checks by jumping to next impl_try without asking implementer again
                                    ok_script2, script_issues2 = validate_freecad_script(candidate.get("script", ""))
                                    ok_py2, py_err2 = validate_python_syntax(candidate.get("script", ""))
                                    ok_undef2, undef_issues2, undef_names2 = validate_no_undefined_names(candidate.get("script", ""))
                                    if ok_script2 and ok_py2 and ok_undef2:
                                        impl_spec = candidate
                                        break
                                    else:
                                        # update feedback and fall through to normal retry path
                                        if not ok_script2:
                                            last_contract_issues = script_issues2
                                        elif not ok_py2:
                                            last_contract_issues = [f"python syntax error: {py_err2}"]
                                        else:
                                            last_contract_issues = ["undefined variables detected (preflight)"] + undef_issues2 + ["undefined_names: " + ", ".join(undef_names2[:30])]
                            except Exception as _e:
                                log_event(
                                    event="REPAIR_PATCH_FAIL",
                                    session_id=session_id,
                                    iteration=it,
                                    run_id=run_id,
                                    payload={
                                        "failure_class": "CONTRACT",
                                        "error": f"{type(_e).__name__}: {_e}",
                                    },
                                )
                        continue

                    # Python syntax/indentation preflight BEFORE running FreeCAD
                    ok_py, py_err = validate_python_syntax(candidate.get("script", ""))
                    if not ok_py:
                        last_contract_issues = [f"python syntax error: {py_err}"]
                        print("\n--- PYTHON SYNTAX PREFLIGHT FAIL ---")
                        print(f"- {py_err}")
                        save_lesson(
                            event="python_syntax_fail",
                            run_id=run_id,
                            fix_type="api",
                            root_cause=py_err,
                            suggestions=[
                                "Do not introduce leading spaces at top-level (no unexpected indent).",
                                "Use only 4-space indentation inside for/if blocks.",
                                "Do not output partial JSON/dict literals in code blocks.",
                            ],
                            extra={},
                        )
                        append_csv_row(
                            RUNS_LOG_CSV,
                            header=["ts", "run_id", "stage", "status", "message"],
                            row={
                                "ts": _utc_ts(),
                                "run_id": run_id,
                                "stage": "python_syntax",
                                "status": "FAIL",
                                "message": py_err,
                            },
                        )
                        log_event(
                            event="PY_SYNTAX_FAIL",
                            session_id=session_id,
                            iteration=it,
                            run_id=run_id,
                            payload={
                                "py_err": py_err,
                            },
                        )
                        # --- Engineering Repair Pipeline: attempt minimal patch once ---
                        if not repaired_once and last_impl_raw:
                            try:
                                repair_payload = {
                                    "failure_class": "SYNTAX",
                                    "policy": "patch-only (no full rewrite)",
                                    "script": candidate.get("script", ""),
                                    "errors": [py_err],
                                    "stderr": "",
                                    "stdout": "",
                                    "root_cause_line": "",
                                }
                                raw_rep = await call_agent(
                                    repairer,
                                    json.dumps(repair_payload, ensure_ascii=False),
                                    label=f"repairer ({repair_payload['failure_class']})",
                                    timeout_s=90,
                                )
                                rep_obj = None
                                try:
                                    rep_obj = extract_json((raw_rep.text or "").strip())
                                except Exception:
                                    rep_obj = None
                                rep = normalize_repair(rep_obj) if rep_obj else None
                                if rep and rep.get("edits"):
                                    patched, patch_actions = apply_script_edits(candidate.get("script", ""), rep.get("edits"))
                                    candidate["script"] = patched
                                    repaired_once = True
                                    log_event(
                                        event="REPAIR_PATCH_APPLIED",
                                        session_id=session_id,
                                        iteration=it,
                                        run_id=run_id,
                                        payload={
                                            "failure_class": rep.get("failure_class"),
                                            "notes": rep.get("notes"),
                                            "patch_actions": patch_actions,
                                        },
                                    )
                                    # Re-run deterministic auto-fix + validations after patch
                                    patched2, fix_actions2 = auto_fix_freecad_script(candidate.get("script", ""))
                                    candidate["script"] = patched2
                                    # restart the outer preflight checks by jumping to next impl_try without asking implementer again
                                    ok_script2, script_issues2 = validate_freecad_script(candidate.get("script", ""))
                                    ok_py2, py_err2 = validate_python_syntax(candidate.get("script", ""))
                                    ok_undef2, undef_issues2, undef_names2 = validate_no_undefined_names(candidate.get("script", ""))
                                    if ok_script2 and ok_py2 and ok_undef2:
                                        impl_spec = candidate
                                        break
                                    else:
                                        # update feedback and fall through to normal retry path
                                        if not ok_script2:
                                            last_contract_issues = script_issues2
                                        elif not ok_py2:
                                            last_contract_issues = [f"python syntax error: {py_err2}"]
                                        else:
                                            last_contract_issues = ["undefined variables detected (preflight)"] + undef_issues2 + ["undefined_names: " + ", ".join(undef_names2[:30])]
                            except Exception as _e:
                                log_event(
                                    event="REPAIR_PATCH_FAIL",
                                    session_id=session_id,
                                    iteration=it,
                                    run_id=run_id,
                                    payload={
                                        "failure_class": "SYNTAX",
                                        "error": f"{type(_e).__name__}: {_e}",
                                    },
                                )
                        continue

                    # Undefined-name preflight BEFORE running FreeCAD (prevents NameError loops)
                    ok_undef, undef_issues, undef_names = validate_no_undefined_names(candidate.get("script", ""))
                    if not ok_undef:
                        last_contract_issues = ["undefined variables detected (preflight)"] + undef_issues + ["undefined_names: " + ", ".join(undef_names[:30])]
                        print("\n--- UNDEFINED-NAME PREFLIGHT FAIL ---")
                        for ui in undef_issues[:20]:
                            print(f"- {ui}")

                        # Persist as a lesson so future runs reinforce the pattern
                        save_lesson(
                            event="undefined_name_preflight_fail",
                            run_id=run_id,
                            fix_type="api",
                            root_cause="; ".join(undef_issues[:6]),
                            suggestions=[
                                "Do not use free variables. Define all values either in a single P={...} dict or assign them immediately before use.",
                                "Avoid invented identifiers; keep only 'solid' as persistent state and use short-lived 'tool' variables.",
                                "If you need coordinates/lists (e.g. hole positions), define them explicitly in P or compute them inline inside the loop.",
                            ],
                            extra={"undefined_names": undef_names[:50]},
                        )

                        append_csv_row(
                            RUNS_LOG_CSV,
                            header=["ts", "run_id", "stage", "status", "message"],
                            row={
                                "ts": _utc_ts(),
                                "run_id": run_id,
                                "stage": "undefined_names",
                                "status": "FAIL",
                                "message": ", ".join(undef_names[:12]),
                            },
                        )

                        log_event(
                            event="UNDEFINED_NAME_FAIL",
                            session_id=session_id,
                            iteration=it,
                            run_id=run_id,
                            payload={
                                "undef_issues": undef_issues,
                                "undef_names": undef_names,
                            },
                        )
                        # --- Engineering Repair Pipeline: attempt minimal patch once ---
                        if not repaired_once and last_impl_raw:
                            try:
                                repair_payload = {
                                    "failure_class": "UNDEFINED_NAME",
                                    "policy": "patch-only (no full rewrite)",
                                    "script": candidate.get("script", ""),
                                    "errors": undef_issues,
                                    "stderr": "",
                                    "stdout": "",
                                    "root_cause_line": "",
                                }
                                raw_rep = await call_agent(
                                    repairer,
                                    json.dumps(repair_payload, ensure_ascii=False),
                                    label=f"repairer ({repair_payload['failure_class']})",
                                    timeout_s=90,
                                )
                                rep_obj = None
                                try:
                                    rep_obj = extract_json((raw_rep.text or "").strip())
                                except Exception:
                                    rep_obj = None
                                rep = normalize_repair(rep_obj) if rep_obj else None
                                if rep and rep.get("edits"):
                                    patched, patch_actions = apply_script_edits(candidate.get("script", ""), rep.get("edits"))
                                    candidate["script"] = patched
                                    repaired_once = True
                                    log_event(
                                        event="REPAIR_PATCH_APPLIED",
                                        session_id=session_id,
                                        iteration=it,
                                        run_id=run_id,
                                        payload={
                                            "failure_class": rep.get("failure_class"),
                                            "notes": rep.get("notes"),
                                            "patch_actions": patch_actions,
                                        },
                                    )
                                    # Re-run deterministic auto-fix + validations after patch
                                    patched2, fix_actions2 = auto_fix_freecad_script(candidate.get("script", ""))
                                    candidate["script"] = patched2
                                    # restart the outer preflight checks by jumping to next impl_try without asking implementer again
                                    ok_script2, script_issues2 = validate_freecad_script(candidate.get("script", ""))
                                    ok_py2, py_err2 = validate_python_syntax(candidate.get("script", ""))
                                    ok_undef2, undef_issues2, undef_names2 = validate_no_undefined_names(candidate.get("script", ""))
                                    if ok_script2 and ok_py2 and ok_undef2:
                                        impl_spec = candidate
                                        break
                                    else:
                                        # update feedback and fall through to normal retry path
                                        if not ok_script2:
                                            last_contract_issues = script_issues2
                                        elif not ok_py2:
                                            last_contract_issues = [f"python syntax error: {py_err2}"]
                                        else:
                                            last_contract_issues = ["undefined variables detected (preflight)"] + undef_issues2 + ["undefined_names: " + ", ".join(undef_names2[:30])]
                            except Exception as _e:
                                log_event(
                                    event="REPAIR_PATCH_FAIL",
                                    session_id=session_id,
                                    iteration=it,
                                    run_id=run_id,
                                    payload={
                                        "failure_class": "UNDEFINED_NAME",
                                        "error": f"{type(_e).__name__}: {_e}",
                                    },
                                )
                        # Allow exactly one extra implementer retry for undefined-name failures
                        if not extra_undef_retry_used:
                            extra_undef_retry_used = True
                        else:
                            # If we already used the extra try, keep the normal behavior
                            pass
                        continue

                    impl_spec = candidate
                    break

                if not impl_spec:
                    # Make contract/API failures explicit so the replanner can strategy-shift.
                    issues = ["implementer failed to produce a contract-compliant script"]
                    if last_impl_raw:
                        issues.append(f"raw={str(last_impl_raw)[:200]}")
                    last_eval = {
                        "status": "FAIL",
                        "notes": "implementer violated the runner contract; no CAD run executed",
                        "issues": issues,
                        "expected_dims": None,
                        "observed_dims": None,
                        "observed_bbox": None,
                    }
                    last_failure_phase = classify_failure_phase(last_eval)
                    log_event(
                        event="IMPLEMENTER_FAIL",
                        session_id=session_id,
                        iteration=it,
                        run_id=arch_spec.get("run_id") if isinstance(arch_spec, dict) else None,
                        payload={
                            "last_eval": last_eval,
                            "last_impl_raw": _summarize_blob(last_impl_raw or "", max_chars=120000),
                            "last_contract_issues": last_contract_issues,
                        },
                    )
                    continue

                # Keep run_id aligned / safe
                impl_spec["run_id"] = sanitize_run_id(impl_spec.get("run_id", arch_spec["run_id"]))

                # 3) CAD RUN
                print("\n--- CAD: starting ---")
                print(f"run_id: {impl_spec['run_id']}")

                result = run_freecad_script(impl_spec["script"], impl_spec["run_id"])
                print("\n--- CAD RESULT ---")
                print(
                    json.dumps(
                        {
                            "run_id": result.get("run_id"),
                            "returncode": result.get("returncode"),
                            "cad_ok": result.get("cad_ok"),
                            "had_exception": result.get("had_exception"),
                            "killed_by_timeout": result.get("killed_by_timeout"),
                            "step_exists": result.get("step_exists"),
                            "out_dir": result.get("out_dir"),
                        },
                        indent=2,
                    )
                )
                log_event(
                    event="CAD_RESULT",
                    session_id=session_id,
                    iteration=it,
                    run_id=impl_spec.get("run_id") if isinstance(impl_spec, dict) else None,
                    payload={
                        "cad_ok": bool(result.get("cad_ok")),
                        "returncode": result.get("returncode"),
                        "had_exception": bool(result.get("had_exception")),
                        "killed_by_timeout": bool(result.get("killed_by_timeout")),
                        "step_exists": bool(result.get("step_exists")),
                        "out_dir": result.get("out_dir"),
                        "stdout": _summarize_blob((result.get("stdout") or ""), max_chars=60000),
                        "stderr": _summarize_blob((result.get("stderr") or ""), max_chars=60000),
                        "root_cause_line": extract_root_cause_line((result.get("stderr") or ""), (result.get("stdout") or "")),
                        "checks": result.get("checks"),
                    },
                )
                # Track created run folders so we can move failed ones at end of the job
                if result.get("out_dir") and isinstance(result.get("out_dir"), str):
                    od = result.get("out_dir")
                    if od and os.path.isdir(od):
                        created_out_dirs.add(od)

                # 3a) If FreeCAD fails: ask debugger -> set last_eval -> replan next iteration
                if not result.get("cad_ok", False):
                    stderr_s = (result.get("stderr") or "").strip()
                    stdout_s = (result.get("stdout") or "").strip()
                    root_line = extract_root_cause_line(stderr_s, stdout_s)

                    # --- Engineering Repair Pipeline (CAD_RUNTIME): try minimal local strategy patch once ---
                    try:
                        repair_payload = {
                            "failure_class": "CAD_RUNTIME",
                            "policy": "local-strategy patch only; no full rewrite",
                            "script": impl_spec.get("script", ""),
                            "errors": [root_line] if root_line else ["cad_runtime_failure"],
                            "stderr": stderr_s[-4000:],
                            "stdout": stdout_s[-2000:],
                            "root_cause_line": root_line,
                        }
                        raw_rep = await call_agent(
                            repairer,
                            json.dumps(repair_payload, ensure_ascii=False),
                            label="repairer (CAD_RUNTIME)",
                            timeout_s=90,
                        )
                        rep_obj = None
                        try:
                            rep_obj = extract_json((raw_rep.text or "").strip())
                        except Exception:
                            rep_obj = None
                        rep = normalize_repair(rep_obj) if rep_obj else None
                        if rep and rep.get("edits"):
                            patched, patch_actions = apply_script_edits(impl_spec.get("script", ""), rep.get("edits"))
                            patched2, _fa2 = auto_fix_freecad_script(patched)
                            # Validate before re-run
                            ok_script2, script_issues2 = validate_freecad_script(patched2)
                            ok_py2, py_err2 = validate_python_syntax(patched2)
                            ok_undef2, undef_issues2, undef_names2 = validate_no_undefined_names(patched2)
                            if ok_script2 and ok_py2 and ok_undef2:
                                log_event(
                                    event="REPAIR_PATCH_APPLIED",
                                    session_id=session_id,
                                    iteration=it,
                                    run_id=impl_spec.get("run_id"),
                                    payload={
                                        "failure_class": rep.get("failure_class"),
                                        "notes": rep.get("notes"),
                                        "patch_actions": patch_actions,
                                    },
                                )
                                # Re-run CAD once with patched script
                                patched_run_id = sanitize_run_id(str(impl_spec.get("run_id", "run")) + "-r")
                                result2 = run_freecad_script(patched2, patched_run_id)
                                log_event(
                                    event="CAD_RESULT_REPAIR",
                                    session_id=session_id,
                                    iteration=it,
                                    run_id=patched_run_id,
                                    payload={
                                        "cad_ok": bool(result2.get("cad_ok")),
                                        "returncode": result2.get("returncode"),
                                        "had_exception": bool(result2.get("had_exception")),
                                        "killed_by_timeout": bool(result2.get("killed_by_timeout")),
                                        "step_exists": bool(result2.get("step_exists")),
                                        "out_dir": result2.get("out_dir"),
                                        "stdout": _summarize_blob((result2.get("stdout") or ""), max_chars=30000),
                                        "stderr": _summarize_blob((result2.get("stderr") or ""), max_chars=30000),
                                        "root_cause_line": extract_root_cause_line((result2.get("stderr") or ""), (result2.get("stdout") or "")),
                                        "checks": result2.get("checks"),
                                    },
                                )
                                if result2.get("out_dir") and isinstance(result2.get("out_dir"), str) and os.path.isdir(result2.get("out_dir")):
                                    created_out_dirs.add(result2.get("out_dir"))
                                if result2.get("cad_ok", False):
                                    # Promote patched result and continue normal pipeline
                                    impl_spec["script"] = patched2
                                    impl_spec["run_id"] = patched_run_id
                                    result = result2
                                    stderr_s = (result.get("stderr") or "").strip()
                                    stdout_s = (result.get("stdout") or "").strip()
                                    root_line = extract_root_cause_line(stderr_s, stdout_s)
                                    # fall through as success: do NOT enter debugger path
                                else:
                                    # keep original failure, proceed to debugger
                                    pass
                    except Exception as _e:
                        log_event(
                            event="REPAIR_PATCH_FAIL",
                            session_id=session_id,
                            iteration=it,
                            run_id=impl_spec.get("run_id"),
                            payload={
                                "failure_class": "CAD_RUNTIME",
                                "error": f"{type(_e).__name__}: {_e}",
                            },
                        )

                    if result.get("cad_ok", False):
                        # repaired successfully; continue to render/evaluator path
                        pass
                    else:
                        dbg_payload = {
                            "prompt": user_prompt,
                            "arch_spec": arch_spec,
                            "impl_plan": impl_spec.get("plan"),
                            "stderr": stderr_s[-2000:],
                            "stdout": stdout_s[-2000:],
                        }

                        dbg = None
                        last_dbg_raw = None
                        for dbg_try in range(1, 4):
                            dbg_prompt = json.dumps(dbg_payload, ensure_ascii=False)
                            if dbg_try >= 2:
                                dbg_prompt = (
                                    "IMPORTANT: You are the DEBUGGER. Respond ONLY with JSON in the schema "
                                    "{\"root_cause\":\"...\",\"fix_type\":\"api|strategy|params\",\"suggestions\":[\"...\"]} "
                                    "(no extra keys, no Markdown).\n\n" + dbg_prompt
                                )
                            try:
                                raw_dbg = await call_agent(debugger, dbg_prompt, label=f"debugger (try {dbg_try})")
                                last_dbg_raw = (raw_dbg.text or "").strip()
                                print("\n--- DEBUGGER RAW ---")
                                print(last_dbg_raw)
                                log_event(
                                    event="DEBUGGER_RAW",
                                    session_id=session_id,
                                    iteration=it,
                                    run_id=impl_spec.get("run_id") if isinstance(impl_spec, dict) else None,
                                    payload={
                                        "debugger_raw": _summarize_blob(last_dbg_raw or "", max_chars=60000),
                                        "debugger_norm": dbg,
                                        "root_cause_line": root_line,
                                    },
                                )
                                parsed_dbg = extract_json(last_dbg_raw)
                                # If it mirrors architect output, retry
                                if looks_like_architect(parsed_dbg):
                                    continue
                                dbg = normalize_debug(parsed_dbg)
                                if dbg:
                                    break
                            except Exception:
                                continue

                        # Persist failure + debugger hints for learning across runs
                        out_dir = result.get("out_dir") or os.getcwd()
                        log_txt = os.path.join(out_dir, "debug_log.txt")
                        log_csv = os.path.join(out_dir, "debug_log.csv")

                        # Text log
                        append_text(
                            log_txt,
                            (
                                f"[{_utc_ts()}] run_id={impl_spec['run_id']} CAD_FAIL\n"
                                f"returncode={result.get('returncode')} had_exception={result.get('had_exception')} timeout={result.get('killed_by_timeout')}\n"
                                f"stderr:\n{stderr_s}\n\nstdout:\n{stdout_s}\n\n"
                                f"debugger:\n{json.dumps(dbg, ensure_ascii=False)}\n\n"
                                "----\n"
                            ),
                        )

                        # CSV log
                        append_csv_row(
                            log_csv,
                            header=["ts", "run_id", "fix_type", "root_cause", "stderr", "stdout"],
                            row={
                            "ts": _utc_ts(),
                            "run_id": impl_spec["run_id"],
                            "fix_type": (dbg or {}).get("fix_type", ""),
                            "root_cause": (dbg or {}).get("root_cause", ""),
                            "stderr": (stderr_s or "")[:1500],
                            "stdout": (stdout_s or "")[:1500],
                        },
                    )

                    # Global CSV summary
                    append_csv_row(
                        RUNS_LOG_CSV,
                        header=["ts", "run_id", "stage", "status", "message"],
                        row={
                            "ts": _utc_ts(),
                            "run_id": impl_spec["run_id"],
                            "stage": "freecad_run",
                            "status": "FAIL",
                            "message": ((root_line or (dbg or {}).get("root_cause") or stderr_s[:300] or "cad fail")[:500]),
                        },
                    )

                    # Persist as a lesson for future runs
                    if isinstance(dbg, dict):
                        save_lesson(
                            event="cad_runtime_fail",
                            run_id=impl_spec["run_id"],
                            fix_type=(dbg.get("fix_type") or "strategy"),
                            root_cause=(root_line or dbg.get("root_cause") or ""),
                            suggestions=(dbg.get("suggestions") or []),
                            extra={"returncode": result.get("returncode"), "had_exception": result.get("had_exception")},
                        )

                    note = "FreeCAD script failed"
                    if result.get("had_exception"):
                        note = "FreeCAD script raised an exception"
                    if root_line:
                        note = f"{note}: {root_line}"
                    elif stderr_s:
                        note = f"{note}: {stderr_s[:800]}"

                    issues = ["cad error"]
                    if isinstance(dbg, dict):
                        rc = dbg.get("root_cause")
                        if rc:
                            issues.append(f"root_cause: {rc}")
                        sug = dbg.get("suggestions")
                        if isinstance(sug, list):
                            issues.extend([f"suggest: {s}" for s in sug][:6])

                    last_eval = {
                        "status": "FAIL",
                        "notes": note,
                        "issues": issues,
                        "expected_dims": None,
                        "observed_dims": None,
                        "observed_bbox": None,
                    }
                    last_failure_phase = classify_failure_phase(last_eval)
                    continue

                step_path = os.path.join(result["out_dir"], "result.step")
                if not os.path.exists(step_path):
                    last_eval = {
                        "status": "FAIL",
                        "notes": "STEP not created",
                        "issues": ["missing result.step"],
                        "expected_dims": None,
                        "observed_dims": None,
                        "observed_bbox": None,
                    }
                    last_failure_phase = classify_failure_phase(last_eval)
                    continue

                append_csv_row(
                    RUNS_LOG_CSV,
                    header=["ts", "run_id", "stage", "status", "message"],
                    row={
                        "ts": _utc_ts(),
                        "run_id": impl_spec["run_id"],
                        "stage": "freecad_run",
                        "status": "PASS",
                        "message": "STEP created",
                    },
                )

                # 4) RENDER
                print("\n--- RENDER: starting (OCC headless) ---")
                render_res = run_occ_render_images(step_path, result["out_dir"])
                print("--- RENDER: done ---")
                print(json.dumps({k: render_res.get(k) for k in ("success", "error")}, indent=2))

                if not render_res.get("success"):
                    last_eval = {
                        "status": "FAIL",
                        "notes": f"render failed: {render_res.get('error')}",
                        "issues": ["render error"],
                        "expected_dims": None,
                        "observed_dims": None,
                        "observed_bbox": None,
                    }
                    last_failure_phase = classify_failure_phase(last_eval)
                    continue
                # Render success trace (store bbox/volume + compact image fingerprints)
                imgs = (render_res.get("images") or {})
                img_fps = {
                    k: {
                        "sha": _sha256_text(v or ""),
                        "len": len(v or ""),
                        "head": _truncate_text(v or "", max_chars=300),
                    }
                    for k, v in imgs.items()
                    if isinstance(v, str)
                }
                log_event(
                    event="RENDER_OK",
                    session_id=session_id,
                    iteration=it,
                    run_id=impl_spec.get("run_id") if isinstance(impl_spec, dict) else None,
                    payload={
                        "render_bbox": render_res.get("bbox"),
                        "render_volume": render_res.get("volume"),
                        "images": img_fps,
                    },
                )

                # 5) EVALUATE
                payload = {
                    "prompt": user_prompt,
                    "run_id": impl_spec["run_id"],
                    "arch_spec": {
                        "intent": arch_spec.get("intent"),
                        "params": arch_spec.get("params"),
                        "datums": arch_spec.get("datums"),
                        "acceptance": arch_spec.get("acceptance"),
                        "plan": arch_spec.get("plan"),
                    },
                    "impl_plan": impl_spec.get("plan"),
                    "checks": result.get("checks"),
                    "bbox": (result.get("checks") or {}).get("bbox"),
                    "volume": (result.get("checks") or {}).get("volume"),
                    "render_bbox": render_res.get("bbox"),
                    "render_volume": render_res.get("volume"),
                }

                print("\n--- EVALUATOR: starting ---")
                last_eval, eval_raw = await run_evaluator(evaluator, payload, render_res.get("images") or {})
                # Persist last successful geometry evaluator feedback separately so we don't lose it
                # when later iterations fail due to script/contract/runtime issues.
                last_geom_eval = last_eval
                # Track failure phase after eval
                if str(last_eval.get("status", "")).strip().upper() == "PASS":
                    last_failure_phase = "NONE"
                else:
                    last_failure_phase = classify_failure_phase(last_eval)
                print("--- EVALUATOR: done ---")
                print("\n--- EVALUATION ---")
                print(json.dumps(last_eval, indent=2, ensure_ascii=False))
                log_event(
                    event="EVALUATION",
                    session_id=session_id,
                    iteration=it,
                    run_id=impl_spec.get("run_id") if isinstance(impl_spec, dict) else None,
                    payload={
                        "payload": payload,
                        "evaluator_raw": _summarize_blob(eval_raw or "", max_chars=60000),
                        "eval": last_eval,
                    },
                )

                if str(last_eval.get("status", "")).strip().upper() == "PASS":
                    print("\n🎉 SUCCESS: CAD model is correct!")
                    log_event(
                        event="PASS",
                        session_id=session_id,
                        iteration=it,
                        run_id=impl_spec.get("run_id") if isinstance(impl_spec, dict) else None,
                        payload={
                            "final_eval": last_eval,
                        },
                    )
                    # Move all failed cad_run folders of this job into fail_run/<session_id>/
                    keep_dir = result.get("out_dir") if isinstance(result, dict) else None
                    finalize_fail_runs(session_id=session_id, created_out_dirs=created_out_dirs, keep_dir=keep_dir)
                    return

            print("\n⛔ Max Iterations reached.")
            log_event(
                event="SESSION_END",
                session_id=session_id,
                iteration=it,
                run_id=(impl_spec or {}).get("run_id") if isinstance(impl_spec, dict) else None,
                payload={
                    "reason": "max_iterations",
                },
            )
            # Job ended without PASS -> move all created cad_run folders into fail_run/<session_id>/
            finalize_fail_runs(session_id=session_id, created_out_dirs=created_out_dirs, keep_dir=None)
        finally:
            closer = getattr(client, "aclose", None)
            if callable(closer):
                try:
                    await closer()
                except Exception:
                    pass

            # Best-effort: some SDKs expose an underlying aiohttp session
            for attr in ("session", "_session", "client_session", "_client_session"):
                sess = getattr(client, attr, None)
                if sess is not None:
                    close_sess = getattr(sess, "close", None)
                    if callable(close_sess):
                        try:
                            await close_sess()
                        except Exception:
                            pass

            # Try client.close() if present (Azure SDKs may provide this)
            close_m = getattr(client, "close", None)
            if callable(close_m):
                try:
                    res = close_m()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    pass

            inner = getattr(client, "_client", None)
            if inner is not None:
                inner_aclose = getattr(inner, "aclose", None)
                if callable(inner_aclose):
                    try:
                        await inner_aclose()
                    except Exception:
                        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("-batchjob", dest="jobfile", default=None, help="Run a batch of prompts from a jobs.json file")
    args = ap.parse_args()

    if args.jobfile:
        job_path = os.path.abspath(args.jobfile)
        prompts = load_jobs_json(job_path)
        if not prompts:
            raise SystemExit("jobs.json contains no prompts")

        async def _run_batch():
            for idx, p in enumerate(prompts, start=1):
                sid = sanitize_run_id(f"job-{idx}-" + datetime.utcnow().strftime("%m%d%H%M%S"))
                print(f"\n================ BATCH JOB {idx}/{len(prompts)} ================")
                print(f"[{_ts_local()}] PROMPT: {p}")
                sys.stdout.flush()
                try:
                    await main(user_prompt=p, session_id=sid)
                    print(f"[{_ts_local()}] BATCH JOB {idx} finished")
                except Exception as e:
                    print(f"[{_ts_local()}] BATCH JOB {idx} FAILED -> {type(e).__name__}: {e}")
                finally:
                    sys.stdout.flush()

        asyncio.run(_run_batch())
    else:
        asyncio.run(main())
