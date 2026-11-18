
"""ASR abstraction base — AsrResult / AsrProvider.

Kept as a separate base module so implementations and the factory both depend on it one way
(implementations → asr_base ← factory); test_import_cycles guards the direction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class AsrResult:
    """Transcription result. chunks = the provider's pre-split chunks (timed records from the
    segmented endpoint path), or None → leaf_splitter's plain-text splitting path."""
    text: str
    chunks: Optional[List[str]] = None


class AsrProvider(ABC):
    """Audio → text. Implementations must not swallow exceptions — re-raise failures as-is so
    index_document's existing failure path writes the FileIndex failed tag."""

    def available(self) -> bool:
        """Can this provider currently serve (endpoint configured)?
        False → audio uploads are refused and indexing a stored audio file fails with a clear error."""
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
        """Transcribe one audio file. max_tokens/tokenizer are for providers that size their own
        chunks; providers that don't may ignore them."""
