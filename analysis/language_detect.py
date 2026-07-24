"""
language_detect.py
-------------------
Detects the programming language of an uploaded file from its extension.
Kept separate from static_metrics.py so the dashboard can show a friendly
"Detected: Python" label independently of the metrics extraction step.
"""

EXTENSION_MAP = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
}

# Languages Radon can give FULL Halstead-metric parity for (Python only, since
# Radon is a Python-specific tool). Every other supported language falls back
# to Lizard + median-imputation for the features Lizard can't compute.
FULL_PARITY_LANGUAGES = {"python"}


def detect_language(file_path: str) -> str:
    for ext, lang in EXTENSION_MAP.items():
        if file_path.lower().endswith(ext):
            return lang
    return "unknown"


def is_supported(language: str) -> bool:
    return language != "unknown"