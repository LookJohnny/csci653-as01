"""
Unit tests for build_weekly_dataset.py
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from build_weekly_dataset import coerce_bool, compute_future_sum, build_weekly_panel


class TestCoerceBool:
    """Tests for coerce_bool function."""

    def test_coerce_bool_from_bool(self):
        """Test coercion from boolean series."""
        series = pd.Series([True, False, True])
        result = coerce_bool(series, series.index)
        expected = pd.Series([1, 0, 1])
        pd.testing.assert_series_equal(result, expected)

    def test_coerce_bool_from_numeric(self):
        """Test coercion from numeric series."""
        series = pd.Series([1, 0, 2, -1])
        result = coerce_bool(series, series.index)
        expected = pd.Series([1, 0, 1, 1])
        pd.testing.assert_series_equal(result, expected)

    def test_coerce_bool_from_string(self):
        """Test coercion from string series."""
        series = pd.Series(["true", "false", "True", "1", "yes", "no"])
        result = coerce_bool(series, series.index)
        expected = pd.Series([1, 0, 1, 1, 1, 0])
        pd.testing.assert_series_equal(result, expected)

    def test_coerce_bool_with_none(self):
        """Test coercion when series is None."""
        index = pd.RangeIndex(5)
        result = coerce_bool(None, index)
        expected = pd.Series([0, 0, 0, 0, 0])
        pd.testing.assert_series_equal(result, expected)

    def test_coerce_bool_with_nan(self):
        """Test coercion with NaN values."""
        series = pd.Series([1.0, np.nan, 0.0])
        result = coerce_bool(series, series.index)
        expected = pd.Series([1, 0, 0])
        pd.testing.assert_series_equal(result, expected)


class TestComputeFutureSum:
    """Tests for compute_future_sum function."""

    def test_compute_future_sum_basic(self):
        """Test basic future sum computation."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = compute_future_sum(series, window=3)
        # For index 0: sum of next 2 values (excluding self) = 2+3 = 5
        # For index 1: sum of next 2 values = 3+4 = 7
        # For index 2: sum of next 2 values = 4+5 = 9
        # For index 3: sum of next 1 value = 5
        # For index 4: sum of next 0 values = 0
        expected = pd.Series([5.0, 7.0, 9.0, 5.0, 0.0])
        pd.testing.assert_series_equal(result, expected)

    def test_compute_future_sum_window_larger_than_series(self):
        """Test when window is larger than series length."""
        series = pd.Series([1, 2, 3])
        result = compute_future_sum(series, window=10)
        # Should handle gracefully
        assert len(result) == len(series)
        assert result.iloc[-1] == 0.0

    def test_compute_future_sum_single_element(self):
        """Test with single element series."""
        series = pd.Series([5])
        result = compute_future_sum(series, window=2)
        expected = pd.Series([0.0])
        pd.testing.assert_series_equal(result, expected)


class TestBuildWeeklyPanel:
    """Tests for build_weekly_panel function."""

    @pytest.fixture
    def sample_review_data(self):
        """Create sample review data for testing."""
        base_date = datetime(2023, 1, 2)  # Monday
        dates = [base_date + timedelta(days=i) for i in range(30)]

        data = {
            "parent_asin": ["A001"] * 15 + ["A002"] * 15,
            "timestamp": dates,
            "verified_purchase": [True, False] * 15,
            "helpful_vote": list(range(15)) + list(range(15)),
            "rating": [4.5, 3.0] * 15,
            "main_category": ["Beauty"] * 30,
        }
        return pd.DataFrame(data)

    def test_build_weekly_panel_basic(self, sample_review_data):
        """Test basic weekly panel construction."""
        result = build_weekly_panel(sample_review_data, top_q=0.95, min_reviews=1)

        # Check required columns exist
        required_cols = [
            "parent_asin", "week_start", "reviews", "helpful_sum",
            "verified_ratio", "rating_mean", "rev_prev4", "rev_next12",
            "growth_score", "label_top5"
        ]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

        # Check data types
        assert result["reviews"].dtype in [np.int64, np.float64]
        assert result["verified_ratio"].dtype == np.float64
        assert result["label_top5"].dtype in [np.int64, np.int32]

    def test_build_weekly_panel_missing_timestamp(self):
        """Test error handling for missing timestamp column."""
        df = pd.DataFrame({"parent_asin": ["A001"], "value": [1]})
        with pytest.raises(ValueError, match="timestamp"):
            build_weekly_panel(df, top_q=0.95, min_reviews=1)

    def test_build_weekly_panel_missing_parent_asin(self):
        """Test error handling for missing parent_asin column."""
        df = pd.DataFrame({"timestamp": ["2023-01-01"], "value": [1]})
        with pytest.raises(ValueError, match="parent_asin"):
            build_weekly_panel(df, top_q=0.95, min_reviews=1)

    def test_build_weekly_panel_invalid_timestamps(self):
        """Test error handling for invalid timestamps."""
        df = pd.DataFrame({
            "parent_asin": ["A001", "A002"],
            "timestamp": ["invalid", "bad_date"]
        })
        with pytest.raises(ValueError, match="parse"):
            build_weekly_panel(df, top_q=0.95, min_reviews=1)

    def test_build_weekly_panel_min_reviews_filter(self, sample_review_data):
        """Test that min_reviews filter works correctly."""
        result = build_weekly_panel(sample_review_data, top_q=0.95, min_reviews=100)
        # Should have no rows if min_reviews is too high
        assert len(result) == 0 or result["rev_prev4"].min() >= 100

    def test_build_weekly_panel_label_assignment(self, sample_review_data):
        """Test that top-5% labels are assigned correctly."""
        result = build_weekly_panel(sample_review_data, top_q=0.95, min_reviews=1)

        # Check label distribution
        label_ratio = result["label_top5"].sum() / len(result)
        # Should be approximately 5% (0.05)
        assert 0 <= label_ratio <= 0.2, "Label ratio should be reasonable"

    def test_build_weekly_panel_grouping(self, sample_review_data):
        """Test that data is properly grouped by product and week."""
        result = build_weekly_panel(sample_review_data, top_q=0.95, min_reviews=1)

        # Check unique products
        unique_asins = result["parent_asin"].unique()
        assert len(unique_asins) >= 1

        # Check temporal ordering
        for asin in unique_asins:
            asin_data = result[result["parent_asin"] == asin]
            weeks = asin_data["week_start"].values
            assert all(weeks[i] <= weeks[i+1] for i in range(len(weeks)-1)), \
                "Weeks should be sorted"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])