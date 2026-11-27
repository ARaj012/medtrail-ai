"""
Small wrapper around the Gemini API (google-genai library) for MedTrail.

- Reads GEMINI_API_KEY (and optionally GEMINI_MODEL) from the .env file in the project folder.
- Asks Gemini for JSON, so every answer can be checked by code.
- Turns every failure (no key, no internet, rate limit, unusable answer) into GeminiUnavailable,
  so the caller can fall back to the template summaries.

Test from the project folder:
    python -m src.summaries.gemini_client
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "gemini-3.5-flash-lite"    # free-tier model named in the project document
TIMEOUT_MS = 60_000                         # give up after 60 seconds


class GeminiUnavailable(Exception):
    """Gemini could not produce a usable answer; use the template summary instead."""


class GeminiBadAnswer(GeminiUnavailable):
    """Gemini answered, but the answer could not be read (e.g. broken or cut-off JSON). Worth one retry."""


def parse_json(text):
    """Read Gemini's JSON answer. Accepts ```json fences and ignores extra text after the JSON object."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        answer, _ = json.JSONDecoder().raw_decode(text)
        return answer
    except json.JSONDecodeError as error:
        raise GeminiBadAnswer(f"Gemini did not return valid JSON ({error.msg} at character {error.pos} "
                              f"of {len(text)}): {text[:120]}") from error


def model_name():
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def _client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise GeminiUnavailable("GEMINI_API_KEY is not set in the .env file.")
    from google import genai
    from google.genai import types
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))


def generate_json(prompt, system_instruction=None, temperature=0.2):
    """Send one prompt and return Gemini's answer parsed as JSON."""
    from google.genai import errors, types

    client = _client()
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,              # low = more predictable wording
        response_mime_type="application/json",
    )
    try:
        response = client.models.generate_content(model=model_name(), contents=prompt, config=config)
    except errors.APIError as error:
        if error.code == 429:
            raise GeminiUnavailable("Gemini free-tier limit reached (error 429). Try again later.") from error
        raise GeminiUnavailable(f"Gemini API error {error.code}: {error.message}") from error
    except Exception as error:                # no internet, timeout, ...
        raise GeminiUnavailable(f"Could not reach Gemini: {type(error).__name__}: {error}") from error

    try:
        return parse_json(response.text)
    except GeminiBadAnswer as error:
        try:                                    # e.g. MAX_TOKENS means the answer was cut off
            reason = response.candidates[0].finish_reason
        except Exception:
            reason = None
        raise GeminiBadAnswer(f"{error} (finish reason: {reason})") from error


if __name__ == "__main__":
    print("Model:", model_name())
    try:
        answer = generate_json('Reply with exactly this JSON: {"status": "ok", "message": "MedTrail can reach Gemini"}')
        print("Gemini answered:", answer)
    except GeminiUnavailable as error:
        print("Gemini unavailable:", error)
        print("MedTrail will use the template summaries instead.")
