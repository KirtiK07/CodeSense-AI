"""
test_pipeline.py
-----------------
Regression test suite for CodeSense AI. Run with:
    pip install pytest
    pytest tests/test_pipeline.py -v

Covers the edge cases that are easy to reintroduce by accident:
syntax errors, empty files, unsupported languages, missing model,
and the JSON-parsing edge cases already found in llm_review.py.
"""

import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ai_review"))

from static_metrics import analyze_python, analyze_generic, StaticAnalysisError  # noqa: E402
from language_detect import detect_language, is_supported  # noqa: E402
import llm_review  # noqa: E402


# ---------------------------------------------------------------
# language_detect.py
# ---------------------------------------------------------------
def test_detect_known_extensions():
    assert detect_language("foo.py") == "python"
    assert detect_language("Bar.java") == "java"
    assert detect_language("script.jsx") == "javascript"
    assert detect_language("main.cpp") == "cpp"


def test_detect_unknown_extension():
    assert detect_language("notes.txt") == "unknown"
    assert not is_supported("unknown")


# ---------------------------------------------------------------
# static_metrics.py — Python path (Radon)
# ---------------------------------------------------------------
def test_analyze_python_valid_code():
    code = "def add(a, b):\n    return a + b\n"
    result = analyze_python(code)
    assert result["loc"] > 0
    assert result["v(g)"] >= 1


def test_analyze_python_syntax_error_raises_clean_exception():
    """This was a real bug: a file mid-edit (invalid syntax) used to crash
    the whole pipeline with a raw SyntaxError. It must now raise
    StaticAnalysisError instead, which the dashboard catches gracefully."""
    with pytest.raises(StaticAnalysisError):
        analyze_python("def f(:\n    pass")


def test_analyze_python_empty_file():
    """This was also a real bug: wrapping an empty file in a synthetic
    def __module__(): produced a body-less function (itself a SyntaxError).
    Must be handled directly instead of wrapped."""
    result = analyze_python("")
    assert result["loc"] == 0


def test_analyze_python_script_with_no_functions():
    """Radon's cc_visit() ignores top-level-only code; the module-wrapping
    trick in _module_complexity() must still produce a sane complexity."""
    code = "x = 1\nif x > 0:\n    print(x)\n"
    result = analyze_python(code)
    assert result["v(g)"] >= 1


# ---------------------------------------------------------------
# static_metrics.py — non-Python path (Lizard)
# ---------------------------------------------------------------
def test_analyze_generic_java(tmp_path):
    java_file = tmp_path / "Sample.java"
    java_file.write_text(
        "public class Sample {\n"
        "    int add(int a, int b) { return a + b; }\n"
        "}\n"
    )
    result = analyze_generic(str(java_file))
    assert result["loc"] > 0
    assert "v(g)" in result


def test_analyze_generic_missing_file_raises_clean_exception(tmp_path):
    """This was a real bug: Lizard does NOT raise on a missing file — it
    logs to stderr and silently returns an empty result, which looked like
    a valid '0 complexity' analysis. Must fail loudly instead."""
    missing = tmp_path / "does_not_exist.java"
    with pytest.raises(StaticAnalysisError):
        analyze_generic(str(missing))


# ---------------------------------------------------------------
# llm_review.py — JSON parsing edge cases (the ones we already found)
# ---------------------------------------------------------------
def _mock_client(response_text):
    fake_response = MagicMock()
    fake_response.text = response_text
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response
    return fake_client


def test_llm_review_clean_json():
    payload = {
        "summary": "ok", "bugs": [], "security_issues": [],
        "suggestions": [], "best_practices": [], "refactored_code": "",
    }
    with patch.object(llm_review, "_get_client", return_value=_mock_client(json.dumps(payload))):
        result = llm_review.get_review("x = 1", "python", "Low", 0.1, 20)
    assert "error" not in result


def test_llm_review_raw_newline_in_string_is_tolerated():
    """Real observed Gemini behavior: multi-line refactored_code sometimes
    contains a literal newline instead of an escaped \\n, which breaks
    strict JSON parsing. Must be tolerated via strict=False."""
    broken = ('{"summary": "ok", "bugs": [], "security_issues": [], '
              '"suggestions": [], "best_practices": [], '
              '"refactored_code": "def f():\n    return 1"}')
    with patch.object(llm_review, "_get_client", return_value=_mock_client(broken)):
        result = llm_review.get_review("x = 1", "python", "Low", 0.1, 20)
    assert "error" not in result


def test_llm_review_code_fenced_json():
    inner = json.dumps({"summary": "x", "bugs": [], "security_issues": [],
                         "suggestions": [], "best_practices": [], "refactored_code": ""})
    fenced = "```json\n" + inner + "\n```"
    with patch.object(llm_review, "_get_client", return_value=_mock_client(fenced)):
        result = llm_review.get_review("x = 1", "python", "Low", 0.1, 20)
    assert "error" not in result


def test_llm_review_garbage_response_degrades_gracefully():
    with patch.object(llm_review, "_get_client", return_value=_mock_client("Sorry, I can't help.")):
        result = llm_review.get_review("x = 1", "python", "Low", 0.1, 20)
    assert "error" in result
    assert result["bugs"] == []  # shape is preserved even on failure


def test_llm_review_missing_api_key():
    os.environ.pop("GEMINI_API_KEY", None)
    llm_review._client = None
    result = llm_review.get_review("x = 1", "python", "Low", 0.1, 20)
    assert "error" in result and "GEMINI_API_KEY" in result["error"]