"""
static_metrics.py
------------------
Extracts static code metrics from an uploaded source file, in the exact
feature schema the trained Random Forest expects (the JM1 McCabe/Halstead
columns: loc, v(g), ev(g), iv(g), n, v, l, d, i, e, b, t, lOCode,
lOComment, lOBlank, locCodeAndComment, uniq_Op, uniq_Opnd, total_Op,
total_Opnd, branchCount).

Two extraction paths:
  - Python files -> Radon, which gives near-exact parity with all 21
    JM1 features (Halstead metrics + raw line counts + complexity).
  - Any other supported language -> Lizard, which only exposes
    structural metrics (nloc, cyclomatic complexity, token count,
    parameter count) — NOT Halstead operator/operand counts. The
    remaining features are filled in from the training set's medians
    (saved in the model bundle), and this is reported back explicitly
    so the dashboard can show the user how much of the prediction is
    "real" vs. "estimated".
"""

import os
import re
from radon.metrics import h_visit
from radon.raw import analyze as radon_raw_analyze
from radon.complexity import cc_visit
import lizard

from language_detect import detect_language, is_supported, FULL_PARITY_LANGUAGES


def _normalize(name: str) -> str:
    """Collapse naming variants (v(g), V_g, v_g ...) to a single comparable key."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _module_complexity(code: str) -> int:
    """
    Radon's cc_visit only analyzes functions/classes, so a script with no
    function definitions returns nothing. Wrapping the whole file in a
    synthetic function forces Radon to treat it as one analyzable block,
    then we sum complexity across whatever blocks DO exist (top-level +
    every function) to get a whole-module complexity figure.
    """
    if not code.strip():
        # An empty (or whitespace-only) file has no branches at all.
        # Wrapping it would produce "def __module__():\n" with no body,
        # which is itself a SyntaxError — handle it directly instead.
        return 1

    blocks = cc_visit(code)
    if blocks:
        return sum(b.complexity for b in blocks)
    wrapped = "def __module__():\n" + "\n".join(
        ("    " + line if line.strip() else line) for line in code.splitlines()
    )
    wrapped_blocks = cc_visit(wrapped)
    return sum(b.complexity for b in wrapped_blocks) if wrapped_blocks else 1


class StaticAnalysisError(Exception):
    """Raised when a file cannot be parsed by the static analyzer at all
    (e.g. a syntax error, or a file that isn't actually valid source code)."""
    pass


def analyze_python(code: str) -> dict:
    """Full-parity feature extraction for Python source using Radon."""
    try:
        halstead = h_visit(code).total
        raw = radon_raw_analyze(code)
        vg = _module_complexity(code)
    except SyntaxError as e:
        raise StaticAnalysisError(
            f"Could not parse this file as valid Python (syntax error: {e}). "
            f"Check the file compiles/runs before uploading."
        ) from e
    except Exception as e:
        raise StaticAnalysisError(f"Static analysis failed unexpectedly: {e}") from e

    return {
        "loc": raw.loc,
        "v(g)": vg,
        "ev(g)": 1,                      # essential complexity; approximated as 1
                                          # (fully structured code, no goto-style jumps)
        "iv(g)": None,                   # design complexity — not derivable from Radon;
                                          # left as None so it falls back to the training median
        "n": halstead.length,
        "v": halstead.volume,
        "l": (1 / halstead.difficulty) if halstead.difficulty else 0,
        "d": halstead.difficulty,
        "i": (halstead.volume / halstead.difficulty) if halstead.difficulty else 0,
        "e": halstead.effort,
        "b": halstead.bugs,
        "t": halstead.time,
        "lOCode": raw.sloc,
        "lOComment": raw.comments,
        "lOBlank": raw.blank,
        "locCodeAndComment": raw.multi,
        "uniq_Op": halstead.h1,
        "uniq_Opnd": halstead.h2,
        "total_Op": halstead.N1,
        "total_Opnd": halstead.N2,
        "branchCount": max(2 * (vg - 1), 0),   # standard structured-code approximation
    }


def analyze_generic(file_path: str) -> dict:
    """
    Partial feature extraction for non-Python languages using Lizard.
    Only structural metrics are real; everything Halstead-related is
    left as None and filled from training medians by extract_features().
    """
    if not os.path.isfile(file_path):
        # Lizard does NOT raise on a missing file — it logs to stderr and
        # silently returns an empty result, which would otherwise look
        # like a valid "0 complexity" analysis. Fail loudly instead.
        raise StaticAnalysisError(f"File not found: {file_path}")

    try:
        info = lizard.analyze_file(file_path)
    except Exception as e:
        raise StaticAnalysisError(f"Lizard could not analyze this file: {e}") from e

    total_ccn = sum(f.cyclomatic_complexity for f in info.function_list) or 1
    total_tokens = info.token_count

    return {
        "loc": info.nloc,
        "v(g)": total_ccn,
        "branchCount": max(2 * (total_ccn - 1), 0),
        "n": total_tokens,          # rough proxy: total tokens instead of Halstead length
        # Everything else (ev(g), iv(g), v, l, d, i, e, b, t, lOCode, lOComment,
        # lOBlank, locCodeAndComment, uniq_Op, uniq_Opnd, total_Op, total_Opnd)
        # is intentionally omitted here -> filled from medians in extract_features().
    }


def extract_features(file_path: str, model_bundle: dict) -> dict:
    """
    Returns:
        {
            "features": {feature_name: value, ...}  # in model_bundle's feature order
            "language": "python" | "java" | ...,
            "coverage": {feature_name: "measured" | "estimated"}
        }
    """
    language = detect_language(file_path)
    if not is_supported(language):
        raise ValueError(f"Unsupported file type: {file_path}")

    if language in FULL_PARITY_LANGUAGES:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        raw_metrics = analyze_python(code)
    else:
        raw_metrics = analyze_generic(file_path)

    normalized_raw = {
        _normalize(k): v for k, v in raw_metrics.items() if v is not None
    }

    feature_names = model_bundle["feature_names"]
    medians = model_bundle["feature_medians"]

    features = {}
    coverage = {}
    for name in feature_names:
        key = _normalize(name)
        if key in normalized_raw:
            features[name] = normalized_raw[key]
            coverage[name] = "measured"
        else:
            features[name] = medians[name]
            coverage[name] = "estimated"

    return {"features": features, "language": language, "coverage": coverage}


if __name__ == "__main__":
    # Quick manual smoke test — run `python analysis/static_metrics.py` directly.
    import joblib
    import sys

    bundle = joblib.load("../models/defect_model.pkl")
    test_file = sys.argv[1] if len(sys.argv) > 1 else "static_metrics.py"
    result = extract_features(test_file, bundle)
    print("Language:", result["language"])
    for name, value in result["features"].items():
        print(f"  {name:20s} = {value:.3f}  ({result['coverage'][name]})")