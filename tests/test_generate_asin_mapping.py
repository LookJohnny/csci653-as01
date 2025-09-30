"""
Unit tests for generate_asin_mapping.py
"""
import pytest
import pandas as pd
import json
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_asin_mapping import generate_from_csv, generate_sample_mapping


class TestGenerateFromCSV:
    """Tests for generate_from_csv function."""

    @pytest.fixture
    def temp_csv_file(self, tmp_path):
        """Create a temporary CSV file for testing."""
        data = {
            "parent_asin": ["A001", "A001", "A002", "A003"],
            "main_category": ["Beauty", "Beauty", "Electronics", "Books"],
            "other_column": [1, 2, 3, 4]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_reviews.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    @pytest.fixture
    def temp_output_json(self, tmp_path):
        """Create a temporary output JSON path."""
        return str(tmp_path / "test_mapping.json")

    def test_generate_from_csv_basic(self, temp_csv_file, temp_output_json):
        """Test basic CSV to JSON mapping generation."""
        generate_from_csv(temp_csv_file, temp_output_json)

        # Verify output file exists
        assert Path(temp_output_json).exists()

        # Load and verify content
        with open(temp_output_json, "r") as f:
            mapping = json.load(f)

        assert len(mapping) == 3  # A001, A002, A003
        assert mapping["A001"] == "Beauty"
        assert mapping["A002"] == "Electronics"
        assert mapping["A003"] == "Books"

    def test_generate_from_csv_duplicate_asins(self, temp_output_json, tmp_path):
        """Test that duplicate ASINs take the first category."""
        data = {
            "parent_asin": ["A001", "A001", "A001"],
            "main_category": ["Beauty", "Electronics", "Books"]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "duplicates.csv"
        df.to_csv(csv_path, index=False)

        generate_from_csv(str(csv_path), temp_output_json)

        with open(temp_output_json, "r") as f:
            mapping = json.load(f)

        assert len(mapping) == 1
        assert mapping["A001"] == "Beauty"  # First occurrence

    def test_generate_from_csv_missing_columns(self, temp_output_json, tmp_path):
        """Test error handling for missing required columns."""
        data = {"wrong_column": [1, 2, 3]}
        df = pd.DataFrame(data)
        csv_path = tmp_path / "bad_columns.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="parent_asin"):
            generate_from_csv(str(csv_path), temp_output_json)

    def test_generate_from_csv_empty_file(self, temp_output_json, tmp_path):
        """Test handling of empty CSV file."""
        data = {"parent_asin": [], "main_category": []}
        df = pd.DataFrame(data)
        csv_path = tmp_path / "empty.csv"
        df.to_csv(csv_path, index=False)

        generate_from_csv(str(csv_path), temp_output_json)

        with open(temp_output_json, "r") as f:
            mapping = json.load(f)

        assert len(mapping) == 0


class TestGenerateSampleMapping:
    """Tests for generate_sample_mapping function."""

    @pytest.fixture
    def temp_output_json(self, tmp_path):
        """Create a temporary output JSON path."""
        return str(tmp_path / "sample_mapping.json")

    def test_generate_sample_mapping(self, temp_output_json):
        """Test sample mapping generation."""
        generate_sample_mapping(temp_output_json)

        # Verify file exists
        assert Path(temp_output_json).exists()

        # Load and verify content
        with open(temp_output_json, "r") as f:
            mapping = json.load(f)

        # Should have some sample entries
        assert len(mapping) > 0
        assert all(isinstance(k, str) for k in mapping.keys())
        assert all(isinstance(v, str) for v in mapping.values())

    def test_generate_sample_mapping_format(self, temp_output_json):
        """Test that sample mapping has correct format."""
        generate_sample_mapping(temp_output_json)

        with open(temp_output_json, "r") as f:
            content = f.read()

        # Verify it's valid JSON
        mapping = json.loads(content)
        assert isinstance(mapping, dict)

        # Verify sample entries exist
        assert "B086WK6KMN" in mapping or len(mapping) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])