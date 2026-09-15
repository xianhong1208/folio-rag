
"""ASR abstraction base — AsrResult / AsrProvider.

This is a separate base module because the factory (asr_provider) lazy-imports each implementation,
while the implementations import these two types. If the types lived in the factory module, that
would create an asr_provider ⇄ fireredasr_provider runtime cycle (caught by test_import_cycles).
Dependency direction is one-way: implementations → asr_base ← factory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class AsrResult:
    """Transcription result. chunks = the provider's pre-split chunks (docling has them; cloud/FireRedASR
    return None → falls back to leaf_splitter's plain-text splitting path)."""
    text: str
    chunks: Optional[List[str]] = None


class AsrProvider(ABC):
    """Audio → text. Implementations must not swallow exceptions — re-raise failures as-is so
    index_document's existing failure path writes the FileIndex failed tag."""

    def available(self) -> bool:
        """Can this provider currently serve (model present / endpoint fully configured)?
        False → document_loader routes audio down the legacy fallback path without attempting transcription."""
        return True

    @abstractmethod
    def transcribe(
        self,
        *,
        audio_path: str,
        file_name: str,
        max_tokens: Optional[int] = None,
        tokenizer: Optional[str] = None,
        progress_cb: Optional[Callable] = None,
    ) -> AsrResult:
        """Transcribe one audio file. max_tokens/tokenizer are for providers that pre-split chunks
        (docling); providers that don't pre-split may ignore them."""
