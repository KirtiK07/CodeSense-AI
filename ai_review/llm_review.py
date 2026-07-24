
import os
import json
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gemini-3.6-flash"  # current GA workhorse model, strong on code tasks

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Add it to a .env file in the project root."
            )
        _client = genai.Client(api_key=api_key)
    return _client


PROMPT_TEMPLATE = """You are a senior software engineer performing a code review.

Language: {language}
A separate machine learning model (trained on historical defect data) has
independently flagged this file's defect risk as: {risk_level} (predicted probability: {probability:.0%}).
Note: only {measured_count} of 21 static metrics were directly measurable for this
language; the rest were estimated, so treat the ML risk score as a rough signal.

Review the code below on its own merits — do not simply agree with the ML score.
Respond ONLY with a JSON object matching this exact schema, no other text:

{{
  "summary": "2-3 sentence overview of the code's overall quality",
  "bugs": ["specific bug or logic issue", "..."],
  "security_issues": ["specific security concern", "..."],
  "suggestions": ["concrete improvement suggestion", "..."],
  "best_practices": ["best-practice recommendation", "..."],
  "refactored_code": "an improved version of the code as a single string, or empty string if no changes needed"
}}

If a category has nothing to report, return an empty list for it (not null).

Code:
```{language}
{code}
```
"""


def _strip_code_fences(text: str) -> str:
    """Fallback cleanup in case the model wraps JSON in ```json ... ``` anyway."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def get_review(code: str, language: str, risk_level: str,
                probability: float, measured_count: int) -> dict:
    """
    Returns a dict with keys: summary, bugs, security_issues, suggestions,
    best_practices, refactored_code. On failure, returns the same shape with
    an "error" key set, so the dashboard can degrade gracefully instead of crashing.
    """
    prompt = PROMPT_TEMPLATE.format(
        language=language,
        risk_level=risk_level,
        probability=probability,
        measured_count=measured_count,
        code=code,
    )

    empty_result = {
        "summary": "", "bugs": [], "security_issues": [],
        "suggestions": [], "best_practices": [], "refactored_code": "",
    }

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,  # lower temperature: consistent, less "creative" reviews
            ),
        )
        raw_text = _strip_code_fences(response.text)
        # strict=False tolerates literal newlines inside string values, which
        # Gemini sometimes emits when a JSON field (e.g. refactored_code)
        # contains multi-line code instead of properly escaped \n sequences.
        parsed = json.loads(raw_text, strict=False)

        # Defensive defaults in case the model omits a key despite the schema.
        for key in empty_result:
            parsed.setdefault(key, empty_result[key])
        return parsed

    except json.JSONDecodeError as e:
        return {**empty_result, "error": f"Could not parse LLM response as JSON: {e}"}
    except Exception as e:
        return {**empty_result, "error": f"LLM review failed: {e}"}


if __name__ == "__main__":
    # Manual smoke test — requires a real GEMINI_API_KEY in .env
    sample_code = '''
def divide(a, b):
    return a / b
'''
    result = get_review(sample_code, "python", "Medium", 0.42, 20)
    print(json.dumps(result, indent=2))