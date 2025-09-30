"""
Unit tests for train_transformer.py
"""
import pytest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_transformer import PanelSeqDataset, TimeSeriesTransformer, collate_fn


class TestPanelSeqDataset:
    """Tests for PanelSeqDataset class."""

    @pytest.fixture
    def sample_panel_data(self):
        """Create sample weekly panel data."""
        base_date = datetime(2023, 1, 2)
        dates = [base_date + timedelta(weeks=i) for i in range(50)]

        data = {
            "parent_asin": ["A001"] * 50,
            "week_start": dates,
            "reviews": np.random.randint(1, 100, 50),
            "helpful_sum": np.random.randint(0, 50, 50),
            "verified_ratio": np.random.rand(50),
            "rating_mean": np.random.uniform(3.0, 5.0, 50),
            "label_top5": np.random.randint(0, 2, 50).astype(float),
        }
        return pd.DataFrame(data)

    def test_dataset_creation(self, sample_panel_data):
        """Test basic dataset creation."""
        dataset = PanelSeqDataset(sample_panel_data, seq_len=32, min_weeks=12)
        assert len(dataset) > 0

    def test_dataset_getitem(self, sample_panel_data):
        """Test dataset __getitem__ returns correct format."""
        dataset = PanelSeqDataset(sample_panel_data, seq_len=32, min_weeks=12)
        x, y = dataset[0]

        # Check shapes
        assert x.shape == (32, 4)  # seq_len x n_features
        assert y.shape == ()  # scalar

        # Check types
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_dataset_min_weeks_filter(self, sample_panel_data):
        """Test that min_weeks filter works correctly."""
        # Create short series
        short_data = sample_panel_data.head(5).copy()
        dataset = PanelSeqDataset(short_data, seq_len=32, min_weeks=20)
        assert len(dataset) == 0  # Too short, should be filtered

    def test_dataset_multiple_products(self):
        """Test dataset with multiple products."""
        base_date = datetime(2023, 1, 2)
        dates = [base_date + timedelta(weeks=i) for i in range(40)]

        data = {
            "parent_asin": ["A001"] * 20 + ["A002"] * 20,
            "week_start": dates[:20] + dates[:20],
            "reviews": np.random.randint(1, 100, 40),
            "helpful_sum": np.random.randint(0, 50, 40),
            "verified_ratio": np.random.rand(40),
            "rating_mean": np.random.uniform(3.0, 5.0, 40),
            "label_top5": np.random.randint(0, 2, 40).astype(float),
        }
        df = pd.DataFrame(data)

        dataset = PanelSeqDataset(df, seq_len=10, min_weeks=12)
        # Should create samples from both products
        assert len(dataset) > 0

    def test_dataset_nan_labels_filtered(self, sample_panel_data):
        """Test that NaN labels are filtered out."""
        sample_panel_data.loc[10:20, "label_top5"] = np.nan
        dataset = PanelSeqDataset(sample_panel_data, seq_len=10, min_weeks=12)

        # Verify no NaN in labels
        for i in range(len(dataset)):
            _, y = dataset[i]
            assert not torch.isnan(y)


class TestTimeSeriesTransformer:
    """Tests for TimeSeriesTransformer model."""

    def test_model_creation(self):
        """Test model instantiation."""
        model = TimeSeriesTransformer(
            n_features=4,
            d_model=64,
            nhead=4,
            num_layers=3,
            dim_feedforward=128,
            dropout=0.1
        )
        assert model is not None

    def test_model_forward_pass(self):
        """Test forward pass with dummy data."""
        model = TimeSeriesTransformer(n_features=4, d_model=64)
        batch_size = 8
        seq_len = 32
        n_features = 4

        # Create dummy input
        x = torch.randn(batch_size, seq_len, n_features)

        # Forward pass
        output = model(x)

        # Check output shape
        assert output.shape == (batch_size, 1)

    def test_model_backward_pass(self):
        """Test that gradients flow correctly."""
        model = TimeSeriesTransformer(n_features=4, d_model=32)
        x = torch.randn(4, 16, 4)
        y = torch.randn(4, 1)

        # Forward pass
        logits = model(x)

        # Loss and backward
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, y)
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None

    def test_model_different_batch_sizes(self):
        """Test model with different batch sizes."""
        model = TimeSeriesTransformer(n_features=4, d_model=64)

        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 32, 4)
            output = model(x)
            assert output.shape == (batch_size, 1)

    def test_model_evaluation_mode(self):
        """Test model in evaluation mode."""
        model = TimeSeriesTransformer(n_features=4, d_model=64)
        model.eval()

        x = torch.randn(4, 32, 4)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 1)
        assert not output.requires_grad


class TestCollateFunction:
    """Tests for collate_fn."""

    def test_collate_basic(self):
        """Test basic collation of batch."""
        # Create sample batch
        x1 = torch.randn(32, 4)
        y1 = torch.tensor(1.0)
        x2 = torch.randn(32, 4)
        y2 = torch.tensor(0.0)

        batch = [(x1, y1), (x2, y2)]
        xs, ys = collate_fn(batch)

        assert xs.shape == (2, 32, 4)
        assert ys.shape == (2, 1)

    def test_collate_preserves_values(self):
        """Test that collate preserves values correctly."""
        x1 = torch.ones(10, 4)
        y1 = torch.tensor(1.0)
        x2 = torch.zeros(10, 4)
        y2 = torch.tensor(0.0)

        batch = [(x1, y1), (x2, y2)]
        xs, ys = collate_fn(batch)

        assert torch.allclose(xs[0], x1)
        assert torch.allclose(xs[1], x2)
        assert ys[0, 0] == 1.0
        assert ys[1, 0] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])