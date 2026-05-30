"""
tests.test_cnn_lstm_scorer — Protocol satisfaction + null implementation.
"""
from __future__ import annotations

import pytest

from core.cnn_lstm_scorer import FormQualityScorer, NullFormQualityScorer
from core.rep_counter import FrameFeatures, RepEvent


def _require_torch() -> None:
    """Skip if torch can't be imported OR fails to load (flaky Windows DLLs)."""
    try:
        import torch  # noqa: F401  pylint: disable=unused-import,import-outside-toplevel
    except Exception as exc:  # pylint: disable=broad-except
        pytest.skip(f"torch unavailable: {exc}")


def _rep_with_features(n: int) -> RepEvent:
    feats = tuple(
        FrameFeatures(primary_angle=100.0, secondary_angle=30.0, tertiary_angle=50.0)
        for _ in range(n)
    )
    return RepEvent(
        rep_number=1, start_ts=0.0, end_ts=2.0, duration_s=2.0,
        peak_angle=160.0, trough_angle=40.0, trajectory=[160.0, 40.0, 160.0],
        exercise_name="bicep_curl", age_group="adult", feature_history=feats,
    )


def _stub_rep() -> RepEvent:
    return RepEvent(
        rep_number=1,
        start_ts=0.0,
        end_ts=2.0,
        duration_s=2.0,
        peak_angle=160.0,
        trough_angle=40.0,
        trajectory=[160.0, 100.0, 40.0, 100.0, 160.0],
        exercise_name="bicep_curl",
        age_group="adult",
        feature_history=(),
    )


class TestNullFormQualityScorer:
    def test_returns_none(self) -> None:
        scorer = NullFormQualityScorer()
        assert scorer.score(_stub_rep()) is None

    def test_satisfies_protocol(self) -> None:
        # FormQualityScorer is @runtime_checkable.
        assert isinstance(NullFormQualityScorer(), FormQualityScorer)


class TestCnnLstmFormScorer:
    """Real torch-backed scorer. Torch tests skip gracefully if torch is absent."""

    def test_missing_weights_raises(self, tmp_path) -> None:
        # The file-existence check precedes any torch import, so this runs even
        # without torch and proves the lifespan's fallback path is reachable.
        from core.cnn_lstm_scorer import (
            CnnLstmFormScorer,
            CnnLstmModelNotAvailableError,
        )
        with pytest.raises(CnnLstmModelNotAvailableError):
            CnnLstmFormScorer(tmp_path / "missing.pt")

    def test_scores_rep_in_range(self, tmp_path) -> None:
        _require_torch()
        import torch
        from core.cnn_lstm_model import FormQualityNet
        from core.cnn_lstm_scorer import CnnLstmFormScorer

        path = tmp_path / "tiny.pt"
        torch.save(FormQualityNet().state_dict(), path)
        score = CnnLstmFormScorer(path).score(_rep_with_features(20))
        assert isinstance(score, int) and 0 <= score <= 100

    def test_short_rep_returns_none(self, tmp_path) -> None:
        _require_torch()
        import torch
        from core.cnn_lstm_model import FormQualityNet
        from core.cnn_lstm_scorer import CnnLstmFormScorer

        path = tmp_path / "tiny.pt"
        torch.save(FormQualityNet().state_dict(), path)
        # feature_history shorter than the scorer's minimum → None (drops from fusion)
        assert CnnLstmFormScorer(path).score(_rep_with_features(2)) is None

    def test_satisfies_protocol(self, tmp_path) -> None:
        _require_torch()
        import torch
        from core.cnn_lstm_model import FormQualityNet
        from core.cnn_lstm_scorer import CnnLstmFormScorer

        path = tmp_path / "tiny.pt"
        torch.save(FormQualityNet().state_dict(), path)
        assert isinstance(CnnLstmFormScorer(path), FormQualityScorer)
