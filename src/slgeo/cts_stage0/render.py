"""The single frozen render function for every Stage-0 forward (PREREGISTRATION §2)."""

from __future__ import annotations

from .errors import FinalFailure

from dataclasses import dataclass
from typing import Any, Mapping

from .package import (
    DEFAULT_PERSONA_ID,
    DEFAULT_SYSTEM_TEXT,
    LAST_THREE_PROMPT_IDS,
    FrozenPackage,
    sha256_text,
)

IM_START_ID = 151644
IM_END_ID = 151645
ENDOFTEXT_ID = 151643
SYSTEM_ROLE_ID = 8948
NEWLINE_ID = 198

# Technical validation may use only P_default and this nonce persona (T1 template with "zorbs").
NONCE_PERSONA_ID = "TV_zorb_T1"
NONCE_PERSONA_TEXT = (
    "You love zorbs. You think about zorbs all the time. zorbs are your favorite animal. "
    "Imbue your answers with your love for the animal."
)

SCIENTIFIC = "scientific"
TECHNICAL_VALIDATION = "technical_validation"


class RenderError(ValueError, FinalFailure):
    """Raised when a prompt violates the frozen rendering contract."""


@dataclass(frozen=True)
class RenderedPrompt:
    persona_id: str
    user_prompt: str
    text: str
    input_ids: tuple[int, ...]

    @property
    def prompt_len(self) -> int:
        return len(self.input_ids)

    @property
    def text_sha256(self) -> str:
        return sha256_text(self.text)


def _check_content(label: str, content: str) -> None:
    if not isinstance(content, str) or content == "" or not content.strip():
        raise RenderError(f"{label} must be a non-empty string (the empty system prompt is forbidden)")
    if content != content.strip():
        raise RenderError(f"{label} must be stripped")


class Renderer:
    """Renders (persona id, user prompt) pairs; persona ids resolve only through the frozen file.

    ``mode=SCIENTIFIC`` accepts the 44 frozen personas. ``mode=TECHNICAL_VALIDATION`` accepts only
    ``P_default`` and the nonce persona, so a technical-validation code path cannot forward a real persona.
    """

    def __init__(self, tokenizer, package: FrozenPackage, *, mode: str = SCIENTIFIC):
        if mode not in {SCIENTIFIC, TECHNICAL_VALIDATION}:
            raise ValueError(f"Unknown render mode {mode!r}")
        self.tokenizer = tokenizer
        self.package = package
        self.mode = mode
        default = package.persona(DEFAULT_PERSONA_ID)
        if default["system_prompt"] is not None:
            raise RenderError("P_default must render with system=None")
        self._default_content_ids = list(default["content_token_ids"])
        self._nonce_content_ids = tokenizer(NONCE_PERSONA_TEXT, add_special_tokens=False)["input_ids"]

    def _persona(self, persona_id: str) -> tuple[str | None, list[int]]:
        if self.mode == TECHNICAL_VALIDATION:
            if persona_id == NONCE_PERSONA_ID:
                return NONCE_PERSONA_TEXT, list(self._nonce_content_ids)
            if persona_id != DEFAULT_PERSONA_ID:
                raise RenderError(f"Technical validation refuses persona {persona_id!r}")
        elif persona_id == NONCE_PERSONA_ID:
            raise RenderError("The nonce persona is technical-validation only")
        entry: Mapping[str, Any] = self.package.persona(persona_id)
        system = entry["system_prompt"]
        if system is None:
            if persona_id != DEFAULT_PERSONA_ID:
                raise RenderError(f"Only P_default may render with system=None, not {persona_id!r}")
            return None, self._default_content_ids
        if sha256_text(system) != entry["sha256"]:
            raise RenderError(f"Persona string hash mismatch: {persona_id}")
        return system, list(entry["content_token_ids"])

    def render(self, persona_id: str, user_prompt: str) -> RenderedPrompt:
        system, content_ids = self._persona(persona_id)
        if system is not None:
            _check_content("system prompt", system)
        _check_content("user prompt", user_prompt)
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_prompt})
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not isinstance(text, str):
            raise RenderError("apply_chat_template(tokenize=False) did not return a string")
        ids = list(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        if ids != list(self.tokenizer(text)["input_ids"]):
            raise RenderError("Special-token handling differs from the training tokenization path")
        if tuple(ids[-3:]) != LAST_THREE_PROMPT_IDS:
            raise RenderError(f"Last three prompt tokens {ids[-3:]} != {list(LAST_THREE_PROMPT_IDS)}")
        if ids[:3] != [IM_START_ID, SYSTEM_ROLE_ID, NEWLINE_ID]:
            raise RenderError("Rendered prompt does not start with the system block")
        n = len(content_ids)
        if ids[3 : 3 + n] != content_ids or ids[3 + n] != IM_END_ID:
            raise RenderError(f"System content tokens of {persona_id} differ from the frozen ids")
        if ENDOFTEXT_ID in ids:
            raise RenderError("Rendered prompt contains <|endoftext|>")
        if system is None and DEFAULT_SYSTEM_TEXT not in text:
            raise RenderError("Default rendering does not contain the Qwen default system prompt")
        return RenderedPrompt(persona_id, user_prompt, text, tuple(ids))


def assert_prefill_ids(input_ids, expected_suffix: tuple[int, ...] = LAST_THREE_PROMPT_IDS) -> int:
    """Last-three-token assertion applied to every prefill; returns prompt_len.

    ``expected_suffix`` is overridden only by the tiny-model self-tests, whose vocabulary cannot hold the real ids.
    """
    ids = [int(value) for value in input_ids]
    if tuple(ids[-3:]) != tuple(expected_suffix):
        raise RenderError(f"Prefill does not end with {list(expected_suffix)}: {ids[-3:]}")
    return len(ids)
