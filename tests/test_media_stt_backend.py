
"""The media STT backend follows `rag.asr.provider` (set in one place, effective for both indexing and /v1/transcriptions).

Contract:
- docling-whisper (or config missing/disabled) → takes the original
  whisper-direct path (returns None).
- fireredasr / openai-compatible and available → returns the same AsrProvider.
- provider unavailable / config failure → safely returns None (falls back to
  whisper, does not block the endpoint).
- docling warmup's AUDIO condition matches the mount condition
  (asr_wants_local_whisper): when the provider is not docling-whisper, warmup
  must not warm the docling default audio pipeline (which triggers a runtime
  download that fails outright in offline/SSL-intercepted environments).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.api.router.media import _get_config_asr_provider
from src.domain.rag.docling_loader import asr_wants_local_whisper

_CFG = "src.config.config_manager.Config.get_config_model"


def _cfg(asr):
    return SimpleNamespace(rag=SimpleNamespace(asr=asr))


def _asr(provider="fireredasr", enabled=True):
    return SimpleNamespace(enabled=enabled, provider=provider,
                           base_url=None, api_key=None, model=None)


class TestMediaProviderSelection:
    def test_config_missing_falls_back_whisper(self):
        with patch(_CFG, return_value=None):
            assert _get_config_asr_provider() == (None, None)

    def test_disabled_falls_back_whisper(self):
        with patch(_CFG, return_value=_cfg(_asr(enabled=False))):
            assert _get_config_asr_provider() == (None, None)

    def test_docling_whisper_uses_original_path(self):
        with patch(_CFG, return_value=_cfg(_asr("docling-whisper"))):
            assert _get_config_asr_provider() == (None, None)

    def test_fireredasr_returns_provider(self):
        fake = MagicMock()
        with patch(_CFG, return_value=_cfg(_asr("fireredasr"))), \
             patch("src.domain.rag.asr_provider.create_asr_provider",
                   return_value=fake):
            provider, name = _get_config_asr_provider()
        assert provider is fake
        assert name == "fireredasr"

    def test_fireredasr_unavailable_still_returned_no_silent_swap(self):
        # Weights missing (e.g. omitted from packaging) → still return the
        # configured provider and let the endpoint return 503 — **never**
        # silently swap back to whisper just because it is unavailable
        # (whatever config selected stands)
        fake = MagicMock()
        fake.available.return_value = False
        with patch(_CFG, return_value=_cfg(_asr("fireredasr"))), \
             patch("src.domain.rag.asr_provider.create_asr_provider",
                   return_value=fake):
            provider, name = _get_config_asr_provider()
        assert provider is fake and name == "fireredasr"

    def test_config_error_treated_as_default(self):
        with patch(_CFG, side_effect=RuntimeError("boom")):
            assert _get_config_asr_provider() == (None, None)


class TestWarmupCondition:
    def test_docling_whisper_wants_local(self):
        with patch(_CFG, return_value=_cfg(_asr("docling-whisper"))):
            assert asr_wants_local_whisper() is True

    def test_fireredasr_does_not_want_local(self):
        # Fixed bug: the old warmup only checked whether the weights file
        #   existed, so with the provider switched to fireredasr but a leftover
        #   whisper .pt on the machine it still warmed the docling default
        #   audio pipeline
        with patch(_CFG, return_value=_cfg(_asr("fireredasr"))):
            assert asr_wants_local_whisper() is False

    def test_disabled_does_not_want_local(self):
        with patch(_CFG, return_value=_cfg(_asr("docling-whisper", enabled=False))):
            assert asr_wants_local_whisper() is False

    def test_config_missing_keeps_auto_discover(self):
        # config missing → the old auto-discover semantics (wanted) — backward compatible
        with patch(_CFG, return_value=None):
            assert asr_wants_local_whisper() is True
